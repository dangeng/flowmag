import os
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation, resize
from torchvision.transforms import RandomResizedCrop, Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, CenterCrop, Resize, ColorJitter

from einops import rearrange

class RepeatDataset(Dataset):
    def __init__(self, dataset, factor=10):
        self.dataset = dataset
        self.factor = factor

    def __getitem__(self, idx):
        idx = idx % len(self.dataset)
        return self.dataset[idx]

    def __len__(self):
        return self.factor * len(self.dataset)


class TrainingFramesDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.frame_names = sorted(os.listdir(self.root))

    def transform_frame(self, frame):

        # Crop to H and W multiple of 8 (for RAFT)
        # n.b. technically this is only necessary for training
        c, h, w = frame.shape
        frame = frame[:,:(h//8)*8,:(w//8)*8]
        return frame

    def __getitem__(self, idx):
        fname = self.frame_names[idx]

        frame = Image.open(self.root / fname)
        frame = to_tensor(frame)

        # # Transform tensors
        frame = self.transform_frame(frame)

        return frame

    def __len__(self):
        return len(self.frame_names)

class FramesDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.frame_names = sorted(os.listdir(self.root))

    def transform_frame(self, frame):

        # Crop to H and W multiple of 8 (for RAFT)
        # n.b. technically this is only necessary for training
        c, h, w = frame.shape
        frame = frame[:,:(h//8)*8,:(w//8)*8]
        return frame

    def __getitem__(self, idx):
        fname = self.frame_names[idx]

        frame = Image.open(self.root / fname)
        frame = to_tensor(frame)

        return frame

    def __len__(self):
        return len(self.frame_names)

class TestTimeAdaptDataset(Dataset):
    def __init__(self, root, mode='first', length=None):
        '''
        args:
            root: (string) path to directory of frames
            mode: ['first', 'random'] how to sample frames
                first: always samples first frame + idx^th frame
                random: randomly samples two frames
        '''
        self.root = Path(root)
        self.frame_names = sorted(os.listdir(self.root))
        self.mode = mode
        self.im_size = 512
        self.scale = 1.1    # stretching scale

        self.geom_transform = Compose([
            RandomRotation(5),
        ])
        self.color_transform = ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.3)

        if length is not None:
            self.length = length
        else:
            self.length = len(self.frame_names)

    def transform_frames(self, frames):
        c, t, h, w = frames.shape

        # Apply geometric transforms on all frames
        frames = rearrange(frames, 'c t h w -> (c t) h w')
        frames = self.geom_transform(frames)
        frames = rearrange(frames, '(c t) h w -> c t h w', c=c, t=t)

        # scale
        new_h = h * self.scale**np.random.uniform(-1, 1)    # rescale for stretch aug
        new_w = w * self.scale**np.random.uniform(-1, 1)
        new_h = int(new_h)
        new_w = int(new_w)
        frames = resize(frames, (new_h, new_w))

        # Pad out so edges are at least im_size.
        to_pad_h = max(self.im_size - new_h, 0)
        to_pad_w = max(self.im_size - new_w, 0)
        pad_l_h = to_pad_h // 2
        pad_r_h = to_pad_h - pad_l_h
        pad_l_w = to_pad_w // 2
        pad_r_w = to_pad_w - pad_l_w
        frames = F.pad(frames, (pad_l_w, pad_r_w, pad_l_h, pad_r_h))
        _, _, padded_h, padded_w = frames.shape
        
        # crop to (im_size, im_size)
        ch = self.im_size
        cw = self.im_size
        ct = np.random.randint(padded_h-ch+1)
        cl = np.random.randint(padded_w-cw+1)
        frames = frames[:,:,ct:ct+ch,cl:cl+cw]

        # Apply color transforms (must have c=3, treating t as batch)
        frames = rearrange(frames, 'c t h w -> t c h w')
        frames = self.color_transform(frames)
        frames = rearrange(frames, 't c h w -> c t h w')

        return frames

    def __getitem__(self, idx):
        idx = idx % len(self.frame_names)

        if self.mode == 'first':
            idx0 = 0
            idx1 = idx
        elif self.mode == 'random':
            idx0 = np.random.randint(self.length)
            idx1 = np.random.randint(self.length)
        else:
            raise NotImplementedError

        frame0 = Image.open(self.root / self.frame_names[idx0])
        frame1 = Image.open(self.root / self.frame_names[idx1])

        frame0 = to_tensor(frame0)
        frame1 = to_tensor(frame1)

        frames = torch.stack([frame0, frame1], dim=1)

        # Transform tensors
        frames = self.transform_frames(frames)

        return frames

    def __len__(self):
        return self.length

class FlowMagDataset(Dataset):
    def __init__(self, data_root, split, aug=False, img_size=256):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size

        # There is not valid set, only a test set
        if self.split == 'valid':
            self.split = 'test'

        # Get frame metadata
        self.frameA_dir = self.data_root / self.split / 'frameA'
        self.frameB_dir = self.data_root / self.split / 'frameB'
        with open(self.data_root / f'{self.split}_fn.json', 'r') as f:
            self.fnames = json.load(f)

        # Make augmentations
        if aug:
            self.transform = Compose([
                RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                RandomHorizontalFlip(.5),
                RandomVerticalFlip(.5),
                RandomRotation(15),
            ])
            self.color_transform = ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.3)
        else:
            self.transform = Compose([
                Resize(img_size),
                CenterCrop(img_size),
            ])
            self.color_transform = nn.Identity()


    def transform_frames(self, frames):
        c, t, h, w = frames.shape

        # Apply geometric transforms on all frames
        frames = rearrange(frames, 'c t h w -> (c t) h w')
        frames = self.transform(frames)
        frames = rearrange(frames, '(c t) h w -> c t h w', c=c, t=t)

        # Apply color transforms (must have c=3, treating t as batch)
        frames = rearrange(frames, 'c t h w -> t c h w')
        frames = self.color_transform(frames)
        frames = rearrange(frames, 't c h w -> c t h w')

        return frames

    
    def __getitem__(self, idx):
        # Load both frameA and frameB
        image_paths = [self.frameA_dir / self.fnames[idx], self.frameB_dir / self.fnames[idx]]
        images = [Image.open(path) for path in image_paths]
        images = [to_tensor(im) for im in images]
        frames = torch.stack(images, dim=1)

        frames = self.transform_frames(frames)

        info = {'fname': self.fnames[idx]}

        return frames, info


    def __len__(self):
        return len(self.fnames)


def get_dataloader(config, split):
    if split == 'train':
        aug = config.data.aug
    else:
        aug = False

    dataset = FlowMagDataset(config.data.dataroot, split=split, aug=aug, img_size=config.data.im_size)

    return dataset
