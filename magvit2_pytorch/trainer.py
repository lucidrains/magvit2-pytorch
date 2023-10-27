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
    DataLoader
)

from accelerate import Accelerator

from ema_pytorch import EMA

# constants

VideosOrImagesLiteral = Union[
    Literal['videos'],
    Literal['images']
]

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
        grad_accum_every: int = 1,
        apply_gradient_penalty_every: int = 4,
        max_grad_norm: Optional[float] = None,
        dataset: Optional[Dataset] = None,
        dataset_folder: Optional[str] = None,
        dataset_type: VideosOrImagesLiteral = 'videos',
        random_split_seed = 42,
        valid_frac = 0.05,
        validate_every_step = 100,
        num_frames = 17,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        dataset_kwargs: dict = dict()
    ):
        super().__init__()

        self.accelerator = Accelerator(**accelerate_kwargs)

        # model and exponentially moving averaged model

        self.model = model
        self.ema_model = EMA(model, **ema_kwargs)

        # dataset

        if not exists(dataset):
            if dataset_type == 'videos':
                dataset_klass = VideoDataset
                dataset_kwargs = {**dataset_kwargs, 'num_frames': num_frames}
            else:
                dataset_klass = ImageDatset

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

        # optimizers

        self.optimizer = get_optimizer(model.parameters(), **optimizer_kwargs)
        self.discr_optimizer = get_optimizer(model.discr_parameters(), **optimizer_kwargs)

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

        # keep track of train step

        self.register_buffer('step', torch.tensor(0))

    def print(self, msg):
        return self.accelerator.print(msg)

    def train_step(self, dl_iter):
        self.model.train()

        step = self.step.item()

        # main model

        for _ in range(self.grad_accum_every):
            data, *_ = next(dl_iter)

            loss, loss_breakdown = self.model(
                data,
                return_loss = True
            )

            self.accelerator.backward(loss / self.grad_accum_every)

        self.print(f'loss: {loss.item():.3f}')

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()

        # update ema model

        self.ema_model.update()

        # discriminator

        apply_gradient_penalty = not (step % self.apply_gradient_penalty_every)

        for _ in range(self.grad_accum_every):
            data, *_ = next(dl_iter)

            discr_loss, discr_loss_breakdown = self.model(
                data,
                return_discr_loss = True,
                apply_gradient_penalty = apply_gradient_penalty
            )

            self.accelerator.backward(discr_loss / self.grad_accum_every)

        self.print(f'discr loss: {discr_loss.item():.3f}')

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.discr_model.parameters(), self.max_grad_norm)

        self.discr_optimizer.step()
        self.discr_optimizer.zero_grad()

        # update train step

        self.step.add_(1)

    @torch.no_grad()
    def valid_step(self, dl_iter):
        self.ema_model.eval()

        recon_loss = 0.

        for _ in range(self.grad_accum_every):
            valid_data, = next(dl_iter)

            loss, recon_video = self.ema_model(
                valid_data,
                return_recon_loss_only = True
            )

            recon_loss += loss / self.grad_accum_every

        self.print(f'validation loss {recon_loss:.3f}')

    def train(self):

        step = self.step.item()

        dl_iter = cycle(self.dataloader)
        valid_dl_iter = cycle(self.valid_dataloader)

        while step < self.num_train_steps:

            self.train_step(dl_iter)

            if not (step % self.validate_every_step):
                self.valid_step(valid_dl_iter)

            step += 1
