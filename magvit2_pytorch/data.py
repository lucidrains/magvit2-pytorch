from pathlib import Path
from functools import partial

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader

import cv2
from PIL import Image
from torchvision import transforms as T, utils

from beartype import beartype
from beartype.typing import Tuple, List
from beartype.door import is_bearable

import numpy as np

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def pair(val):
    return val if isinstance(val, tuple) else (val, val)

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def cast_num_frames(t, *, frames):
    f = t.shape[-3]

    if f == frames:
        return t

    if f > frames:
        return t[..., :frames, :, :]

    return pad_at_dim(t, (0, frames - f), dim = -3)

def convert_image_to_fn(img_type, image):
    if not exists(img_type) or image.mode == img_type:
        return image

    return image.convert(img_type)

def append_if_no_suffix(path: str, suffix: str):
    path = Path(path)

    if path.suffix == '':
        path = path.parent / (path.name + suffix)

    assert path.suffix == suffix, f'{str(path)} needs to have suffix {suffix}'

    return str(path)

# channel to image mode mapping

CHANNEL_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}

# image related helpers fnuctions and dataset

class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels = 3,
        convert_image_to = None,
        exts = ['jpg', 'jpeg', 'png']
    ):
        super().__init__()
        folder = Path(folder)
        assert folder.is_dir(), f'{str(folder)} must be a folder containing images'
        self.folder = folder

        self.image_size = image_size

        exts = exts + [ext.upper() for ext in exts]
        self.paths = [p for ext in exts for p in folder.glob(f'**/*.{ext}')]

        print(f'{len(self.paths)} training samples found at {folder}')

        if exists(channels) and not exists(convert_image_to):
            convert_image_to = CHANNEL_TO_MODE.get(channels)

        self.transform = T.Compose([
            T.Lambda(partial(convert_image_to_fn, convert_image_to)),
            T.Resize(image_size, antialias = True),
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# tensor of shape (channels, frames, height, width) -> gif

# handle reading and writing gif

def seek_all_images(img: Tensor, channels = 3):
    mode = CHANNEL_TO_MODE.get(channels)

    assert exists(mode), f'channels {channels} invalid'

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# tensor of shape (channels, frames, height, width) -> gif

@beartype
def video_tensor_to_gif(
    tensor: Tensor,
    path: str,
    duration = 120,
    loop = 0,
    optimize = True
):
    path = append_if_no_suffix(path, '.gif')
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(str(path), save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

# gif -> (channels, frame, height, width) tensor

def gif_to_tensor(
    path: str,
    channels = 3,
    transform = T.ToTensor()
):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)

# handle reading and writing mp4

def video_to_tensor(
    path: str,              # Path of the video to be imported
    num_frames = -1,        # Number of frames to be stored in the output tensor
    crop_size = None
) -> Tensor:                # shape (1, channels, frames, height, width)

    video = cv2.VideoCapture(path)

    frames = []
    check = True

    while check:
        check, frame = video.read()

        if not check:
            continue

        if exists(crop_size):
            frame = crop_center(frame, *pair(crop_size))

        frames.append(rearrange(frame, '... -> 1 ...'))

    frames = np.array(np.concatenate(frames[:-1], axis = 0))  # convert list of frames to numpy array
    frames = rearrange(frames, 'f h w c -> c f h w')

    frames_torch = torch.tensor(frames).float()

    frames_torch /= 255.
    frames_torch = frames_torch.flip(dims = (0,)) # BGR -> RGB format

    return frames_torch[:, :num_frames, :, :]

@beartype
def tensor_to_video(
    tensor: Tensor,        # Pytorch video tensor
    path: str,             # Path of the video to be saved
    fps = 25,              # Frames per second for the saved video
    video_format = 'MP4V'
):
    path = append_if_no_suffix(path, '.mp4')

    tensor = tensor.cpu()

    num_frames, height, width = tensor.shape[-3:]

    fourcc = cv2.VideoWriter_fourcc(*video_format) # Changes in this line can allow for different video formats.
    video = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    frames = []

    for idx in range(num_frames):
        numpy_frame = tensor[:, idx, :, :].numpy()
        numpy_frame = np.uint8(rearrange(numpy_frame, 'c h w -> h w c'))
        video.write(numpy_frame)

    video.release()

    cv2.destroyAllWindows()

    return video

def crop_center(
    img: Tensor,
    cropx: int,      # Length of the final image in the x direction.
    cropy: int       # Length of the final image in the y direction.
) -> Tensor:
    y, x, c = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:(starty + cropy), startx:(startx + cropx), :]

# video dataset

class VideoDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels = 3,
        num_frames = 17,
        force_num_frames = True,
        exts = ['gif', 'mp4']
    ):
        super().__init__()
        folder = Path(folder)
        assert folder.is_dir(), f'{str(folder)} must be a folder containing videos'
        self.folder = folder

        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in folder.glob(f'**/*.{ext}')]

        print(f'{len(self.paths)} training samples found at {folder}')

        self.transform = T.Compose([
            T.Resize(image_size, antialias = True),
            T.CenterCrop(image_size)
        ])

        # functions to transform video path to tensor

        self.gif_to_tensor = partial(gif_to_tensor, channels = self.channels, transform = self.transform)
        self.mp4_to_tensor = partial(video_to_tensor, crop_size = self.image_size)

        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        ext = path.suffix
        path_str = str(path)

        if ext == '.gif':
            tensor = self.gif_to_tensor(path_str)
        elif ext == '.mp4':
            tensor = self.mp4_to_tensor(path_str)
            frames = tensor.unbind(dim = 1)
            tensor = torch.stack([*map(self.transform, frames)], dim = 1)
        else:
            raise ValueError(f'unknown extension {ext}')

        return self.cast_num_frames_fn(tensor)

# override dataloader to be able to collate strings

def collate_tensors_and_strings(data):
    if is_bearable(data, List[Tensor]):
        return (torch.stack(data),)

    data = zip(*data)
    output = []

    for datum in data:
        if is_bearable(datum, Tuple[Tensor, ...]):
            datum = torch.stack(datum)
        elif is_bearable(datum, Tuple[str, ...]):
            datum = list(datum)
        else:
            raise ValueError('detected invalid type being passed from dataset')

        output.append(datum)

    return tuple(output)

def DataLoader(*args, **kwargs):
    return PytorchDataLoader(*args, collate_fn = collate_tensors_and_strings, **kwargs)
