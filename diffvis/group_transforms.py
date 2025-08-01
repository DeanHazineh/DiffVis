import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from torch.utils.data import Dataset
from einops import rearrange


reverse_transform = transforms.Compose(
    [
        transforms.Lambda(lambda t: permute_dimensions(t)),  # CHW to HWC
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: np.clip(t, 0, 1)),
    ]
)


def permute_dimensions(data):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    data = np.swapaxes(np.swapaxes(data, -3, -1), -3, -2)
    return data


class GroupTransforms:
    def __call__(self, sample):
        raise NotImplementedError("This method should be overridden in subclasses")


class JointRandomFlip(GroupTransforms):
    def __init__(self, horizontal_p=0.5, vertical_p=0.5):
        self.horizontal_p = horizontal_p
        self.vertical_p = vertical_p

    def __call__(self, sample):
        flip_hor = torch.rand(1)
        flip_ver = torch.rand(1)

        for idx in range(len(sample)):
            if flip_hor < self.horizontal_p:
                sample[idx] = torch.flip(sample[idx], [2])  # Assuming array is CxHxW
            if flip_ver < self.vertical_p:
                sample[idx] = torch.flip(sample[idx], [1])  # Assuming array is CxHxW

        return sample


class JointToTensor(GroupTransforms):
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, sample):
        return [
            (
                rearrange(item, "h w c -> c h w")
                if torch.is_tensor(item)
                else self.to_tensor(item)
            )
            for item in sample
        ]


class JointRandomCrop(GroupTransforms):
    def __init__(self, new_sz):
        self.new_sz = new_sz

    def __call__(self, sample):
        if self.new_sz is None:
            return sample

        i, j, h, w = transforms.RandomCrop.get_params(
            sample[0], output_size=self.new_sz
        )
        return [transforms.functional.crop(item, i, j, h, w) for item in sample]


class JointResize(GroupTransforms):
    def __init__(self, new_sz):
        if new_sz is not None:
            self.resize_fn = transforms.Resize(new_sz, antialias=True)
        else:
            self.resize_fn = lambda x: x

    def __call__(self, sample):
        return [self.resize_fn(item) for item in sample]


class ZeroPad:
    def __init__(self, pad_size):
        self.pad_size = pad_size

    def __call__(self, img):
        padby = self.pad_size
        if padby is None:
            return img
        else:
            return F.pad(img, pad=(padby[0], padby[0], padby[1], padby[1]))


class JointZeroPad:
    def __init__(self, pad_size):
        self.apply_pad = ZeroPad(pad_size)

    def __call__(self, sample):
        return [self.apply_pad(item) for item in sample]
