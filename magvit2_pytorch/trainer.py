from pathlib import Path
from functools import partial
from contextlib import contextmanager, nullcontext

import torch
from torch import nn
from torch.nn import Module
from torch.utils.data import Dataset, random_split

from beartype import beartype
from beartype.typing import Optional, Literal, Union

from magvit2_pytorch.optimizer import get_optimizer

from magvit2_pytorch.magvit2_pytorch import VideoTokenizer

from magvit2_pytorch.data import (
    VideoDataset,
    ImageDataset,
    DataLoader,
    video_tensor_to_gif
)

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from einops import rearrange

from ema_pytorch import EMA

# constants

VideosOrImagesLiteral = Union[
    Literal['videos'],
    Literal['images']
]

DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(
    find_unused_parameters = True
)

# helpers

def exists(v):
    return v is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

# class

class VideoTokenizerTrainer(Module):
    @beartype
    def __init__(
        self,
        model: VideoTokenizer,
        *,
        batch_size: int,
        num_train_steps: int,
        learning_rate: float = 1e-5,
        grad_accum_every: int = 1,
        apply_gradient_penalty_every: int = 4,
        max_grad_norm: Optional[float] = None,
        dataset: Optional[Dataset] = None,
        dataset_folder: Optional[str] = None,
        dataset_type: VideosOrImagesLiteral = 'videos',
        checkpoints_folder = './checkpoints',
        results_folder = './results',
        random_split_seed = 42,
        valid_frac = 0.05,
        validate_every_step = 100,
        checkpoint_every_step = 100,
        num_frames = 17,
        use_wandb_tracking = False,
        discr_start_after_step = 0.,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        dataset_kwargs: dict = dict()
    ):
        super().__init__()

        self.use_wandb_tracking = use_wandb_tracking

        if use_wandb_tracking:
            accelerate_kwargs['log_with'] = 'wandb'

        if 'kwargs_handlers' not in accelerate_kwargs:
            accelerate_kwargs['kwargs_handlers'] = [DEFAULT_DDP_KWARGS]

        # instantiate accelerator

        self.accelerator = Accelerator(**accelerate_kwargs)

        # model and exponentially moving averaged model

        self.model = model

        if self.is_main:
            self.ema_model = EMA(
                model,
                include_online_model = False,
                **ema_kwargs
            )

        dataset_kwargs.update(channels = model.channels)

        # dataset

        if not exists(dataset):
            if dataset_type == 'videos':
                dataset_klass = VideoDataset
                dataset_kwargs = {**dataset_kwargs, 'num_frames': num_frames}
            else:
                dataset_klass = ImageDataset

            assert exists(dataset_folder)
            dataset = dataset_klass(dataset_folder, image_size = model.image_size, **dataset_kwargs)

        # splitting dataset for validation

        assert 0 <= valid_frac < 1.

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(dataset))
            valid_size = len(dataset) - train_size
            dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))

            self.print(f'training with dataset of {len(dataset)} samples and validating with randomly splitted {len(valid_dataset)} samples')
        else:
            valid_dataset = dataset
            self.print(f'training with shared training and valid dataset of {len(dataset)} samples')

        # dataset and dataloader

        self.dataset = dataset
        self.dataloader = DataLoader(dataset, shuffle = True, drop_last = True, batch_size = batch_size)

        self.valid_dataset = valid_dataset
        self.valid_dataloader = DataLoader(valid_dataset, shuffle = True, drop_last = True, batch_size = batch_size)

        self.validate_every_step = validate_every_step
        self.checkpoint_every_step = checkpoint_every_step

        # optimizers

        self.optimizer = get_optimizer(model.parameters(), lr = learning_rate, **optimizer_kwargs)
        self.discr_optimizer = get_optimizer(model.discr_parameters(), lr = learning_rate, **optimizer_kwargs)

        # training related params

        self.batch_size = batch_size

        self.num_train_steps = num_train_steps
        self.grad_accum_every = grad_accum_every
        self.max_grad_norm = max_grad_norm

        self.apply_gradient_penalty_every = apply_gradient_penalty_every

        # prepare for maybe distributed

        (
            self.model,
            self.dataloader,
            self.optimizer,
            self.discr_optimizer
        ) = self.accelerator.prepare(
            self.model,
            self.dataloader,
            self.optimizer,
            self.discr_optimizer
        )

        # only use adversarial training after a certain number of steps

        self.discr_start_after_step = discr_start_after_step

        # multiscale discr losses

        self.has_multiscale_discrs = self.model.has_multiscale_discrs
        self.multiscale_discr_optimizers = []

        for ind, discr in enumerate(self.model.multiscale_discrs):
            multiscale_optimizer = get_optimizer(discr.parameters(), lr = learning_rate, **optimizer_kwargs)

            self.multiscale_discr_optimizers.append(multiscale_optimizer)

        if self.has_multiscale_discrs:
            self.multiscale_discr_optimizers = self.accelerator.prepare(*self.multiscale_discr_optimizers)

        # checkpoints and sampled results folder

        checkpoints_folder = Path(checkpoints_folder)
        results_folder = Path(results_folder)

        checkpoints_folder.mkdir(parents = True, exist_ok = True)
        results_folder.mkdir(parents = True, exist_ok = True)

        assert checkpoints_folder.is_dir()
        assert results_folder.is_dir()

        self.checkpoints_folder = checkpoints_folder
        self.results_folder = results_folder

        # keep track of train step

        self.register_buffer('step', torch.tensor(0))

        # move ema to the proper device

        self.ema_model.to(self.device)

    @contextmanager
    @beartype
    def trackers(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        hps: Optional[dict] = None
    ):
        assert self.use_wandb_tracking

        self.accelerator.init_trackers(project_name, config = hps)

        if exists(run_name):
            self.accelerator.trackers[0].run.name = run_name

        yield
        self.accelerator.end_training()

    def log(self, **data_kwargs):
        self.accelerator.log(data_kwargs, step = self.step.item())

    @property
    def device(self):
        return self.unwrapped_model.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def print(self, msg):
        return self.accelerator.print(msg)

    @property
    def ema_tokenizer(self):
        return self.ema_model.ema_model

    def tokenize(self, *args, **kwargs):
        return self.ema_tokenizer.tokenize(*args, **kwargs)

    def save(self, path, overwrite = True):
        path = Path(path)
        assert overwrite or not path.exists()

        pkg = dict(
            model = self.unwrapped_model.state_dict(),
            ema_model = self.ema_model.state_dict(),
            optimizer = self.optimizer.state_dict(),
            discr_optimizer = self.discr_optimizer.state_dict()
        )

        for ind, opt in enumerate(self.multiscale_discr_optimizers):
            pkg[f'multiscale_discr_optimizer_{ind}'] = opt.state_dict()

        torch.save(pkg, str(path))

    def load(self, path):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path))

        self.model.load_state_dict(pkg['model'])
        self.ema_model.load_state_dict(pkg['ema_model'])
        self.optimizer.load_state_dict(pkg['optimizer'])
        self.discr_optimizer.load_state_dict(pkg['discr_optimizer'])

        for ind, opt in enumerate(self.multiscale_discr_optimizers):
            opt.load_state_dict(pkg[f'multiscale_discr_optimizer_{ind}'])

    def train_step(self, dl_iter):
        self.model.train()

        step = self.step.item()

        # determine whether to train adversarially

        train_adversarially = self.model.use_gan and (step + 1) > self.discr_start_after_step

        adversarial_loss_weight = 0. if not train_adversarially else None
        multiscale_adversarial_loss_weight = 0. if not train_adversarially else None

        # main model

        self.optimizer.zero_grad()

        for grad_accum_step in range(self.grad_accum_every):

            is_last = grad_accum_step == (self.grad_accum_every - 1)
            context = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

            data, *_ = next(dl_iter)

            with self.accelerator.autocast(), context():
                loss, loss_breakdown = self.model(
                    data,
                    return_loss = True,
                    adversarial_loss_weight = adversarial_loss_weight,
                    multiscale_adversarial_loss_weight = multiscale_adversarial_loss_weight
                )

                self.accelerator.backward(loss / self.grad_accum_every)

        self.log(
            total_loss = loss.item(),
            recon_loss = loss_breakdown.recon_loss.item(),
            perceptual_loss = loss_breakdown.perceptual_loss.item(),
            adversarial_gen_loss = loss_breakdown.adversarial_gen_loss.item(),
        )

        self.print(f'recon loss: {loss_breakdown.recon_loss.item():.3f}')

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()

        # update ema model

        self.wait()

        if self.is_main:
            self.ema_model.update()

        self.wait()

        # if adversarial loss is turned off, continue

        if not train_adversarially:
            self.step.add_(1)
            return

        # discriminator and multiscale discriminators

        self.discr_optimizer.zero_grad()

        if self.has_multiscale_discrs:
            for multiscale_discr_optimizer in self.multiscale_discr_optimizers:
                multiscale_discr_optimizer.zero_grad()

        apply_gradient_penalty = not (step % self.apply_gradient_penalty_every)

        for grad_accum_step in range(self.grad_accum_every):

            is_last = grad_accum_step == (self.grad_accum_every - 1)
            context = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

            data, *_ = next(dl_iter)

            with self.accelerator.autocast(), context():
                discr_loss, discr_loss_breakdown = self.model(
                    data,
                    return_discr_loss = True,
                    apply_gradient_penalty = apply_gradient_penalty
                )

                self.accelerator.backward(discr_loss / self.grad_accum_every)

        self.log(discr_loss = discr_loss_breakdown.discr_loss.item())

        if apply_gradient_penalty:
            self.log(gradient_penalty = discr_loss_breakdown.gradient_penalty.item())

        self.print(f'discr loss: {discr_loss_breakdown.discr_loss.item():.3f}')

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.model.discr_parameters(), self.max_grad_norm)

            if self.has_multiscale_discrs:
                for multiscale_discr in self.model.multiscale_discrs:
                    self.accelerator.clip_grad_norm_(multiscale_discr.parameters(), self.max_grad_norm)

        self.discr_optimizer.step()

        if self.has_multiscale_discrs:
            for multiscale_discr_optimizer in self.multiscale_discr_optimizers:
                multiscale_discr_optimizer.step()

        # update train step

        self.step.add_(1)

    @torch.no_grad()
    def valid_step(
        self,
        dl_iter,
        save_recons = True,
        num_save_recons = 1
    ):
        self.ema_model.eval()

        recon_loss = 0.
        ema_recon_loss = 0.

        valid_videos = []
        recon_videos = []

        for _ in range(self.grad_accum_every):
            valid_video, = next(dl_iter)
            valid_video = valid_video.to(self.device)

            with self.accelerator.autocast():
                loss, _ = self.unwrapped_model(valid_video, return_recon_loss_only = True)
                ema_loss, ema_recon_video = self.ema_model(valid_video, return_recon_loss_only = True)

            recon_loss += loss / self.grad_accum_every
            ema_recon_loss += ema_loss / self.grad_accum_every

            if valid_video.ndim == 4:
                valid_video = rearrange(valid_video, 'b c h w -> b c 1 h w')

            valid_videos.append(valid_video.cpu())
            recon_videos.append(ema_recon_video.cpu())

        self.log(
            valid_recon_loss = recon_loss.item(),
            valid_ema_recon_loss = ema_recon_loss.item()
        )

        self.print(f'validation recon loss {recon_loss:.3f}')
        self.print(f'validation EMA recon loss {ema_recon_loss:.3f}')

        if not save_recons:
            return

        valid_videos = torch.cat(valid_videos)
        recon_videos = torch.cat(recon_videos)

        recon_videos.clamp_(min = 0., max = 1.)

        valid_videos, recon_videos = map(lambda t: t[:num_save_recons], (valid_videos, recon_videos))

        real_and_recon = rearrange([valid_videos, recon_videos], 'n b c f h w -> c f (b h) (n w)')

        validate_step = self.step.item() // self.validate_every_step

        sample_path = str(self.results_folder / f'sampled.{validate_step}.gif')

        video_tensor_to_gif(real_and_recon, str(sample_path))

        self.print(f'sample saved to {str(sample_path)}')

    def train(self):

        step = self.step.item()

        dl_iter = cycle(self.dataloader)
        valid_dl_iter = cycle(self.valid_dataloader)

        while step < self.num_train_steps:
            self.print(f'step {step}')

            self.train_step(dl_iter)

            self.wait()

            if self.is_main and not (step % self.validate_every_step):
                self.valid_step(valid_dl_iter)

            self.wait()

            if self.is_main and not (step % self.checkpoint_every_step):
                checkpoint_num = step // self.checkpoint_every_step
                checkpoint_path = self.checkpoints_folder / f'checkpoint.{checkpoint_num}.pt'
                self.save(str(checkpoint_path))

            self.wait()

            step += 1
