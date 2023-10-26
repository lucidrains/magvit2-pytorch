import torch
from torch import nn
from torch.nn import Module
from torch.utils.data import Dataset

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
            dataset_klass = VideoDataset if dataset_type == 'videos' else ImageDatset
            assert exists(dataset_folder)
            dataset = dataset_klass(dataset_folder, image_size = model.image_size, **dataset_kwargs)

        self.dataset = dataset
        self.dataloader = DataLoader(dataset, shuffle = True, batch_size = batch_size)

        # optimizers

        self.optimizer = get_optimizer(model.parameters(), **optimizer_kwargs)
        self.discr_optimizer = get_optimizer(model.discr_parameters(), **optimizer_kwargs)

        # training related params

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

        self.register_buffer('step', torch.tensor(0.))

    def print(self, msg):
        return self.accelerator.print(msg)

    def train(self):

        step = self.step.item()
        dl_iter = cycle(self.dataloader)

        while step < self.num_train_steps:

            # main model

            for _ in range(self.grad_accum_every):
                data, *_ = next(dl_iter)

                loss, loss_breakdown = self.model(
                    data,
                    return_loss = True
                )

                self.accelerator.backward(loss / self.grad_accum_every)

            self.print(f'loss: {loss.item()}')

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

            self.print(f'discr loss: {discr_loss.item()}')

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.discr_model.parameters(), self.max_grad_norm)

            self.discr_optimizer.step()
            self.discr_optimizer.zero_grad()

            # update training steps

            step += 1
            self.step.copy_(step)
