from torch.utils.data import Dataset
from natsort import natsorted
import os
from torchvision import transforms
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from dflat.render import hsi_to_rgb
from diffvis.group_transforms import *


def random_crop(data, crop_size):
    if crop_size is None:
        return [dat[:, :, :] for dat in data]

    height, width, c = data[0].shape
    if crop_size > height or crop_size > width:
        raise ValueError("Crop size must be smaller than the dimensions of the data.")

    x_start = np.random.randint(0, height - crop_size + 1)
    y_start = np.random.randint(0, width - crop_size + 1)
    return [
        dat[x_start : x_start + crop_size, y_start : y_start + crop_size, :]
        for dat in data
    ]


class Metadiff(Dataset):
    def __init__(
        self,
        root_dir,
        patchsize=64,
        patch_normalize=True,
        scale_shift_preprocess=True,
        data_fields=["mgs", "hsi"],
        eager_mode=True,
        use_aug=True,
        include=[],
        dtype="float32",
        max_sigma_percent=0.0,
        limit_data=None,
    ):
        self.df = data_fields
        self.patchsize = patchsize
        self.root_dir = root_dir
        self.eager_mode = eager_mode
        self.max_sigma_percent = max_sigma_percent

        if dtype == "float64":
            self.dtype = torch.float64
        elif dtype == "float32":
            self.dtype = torch.float32
        elif dtype == "float16":
            self.dtype = torch.float16
        else:
            raise ValueError("Invalid Datatype")

        patterns_to_exclude = ("_vflip.h5", "_hflip.h5", "_vhflip.h5")
        self.fnames = natsorted(
            [
                f
                for f in os.listdir(root_dir)
                if os.path.isfile(os.path.join(root_dir, f))
            ]
        )
        if not use_aug:
            self.fnames = [
                f for f in self.fnames if not f.endswith(patterns_to_exclude)
            ]
        self.fnames = [
            s for s in self.fnames if any(s.startswith(sub) for sub in include)
        ]
        if limit_data is not None:
            self.fnames = self.fnames[:limit_data]
        print(self.fnames)

        if eager_mode:
            self.data = []
            for fname in tqdm(self.fnames, desc="Loading data into RAM"):
                filepath = os.path.join(self.root_dir, fname)
                with h5py.File(filepath, "r") as hdf:
                    self.data.append([np.array(hdf[key]) for key in self.df])

        transform_list = [
            JointToTensor(),
            (
                transforms.Lambda(
                    lambda sample: (
                        [item / (torch.max(item) + 1e-6) for item in sample]
                    )
                )
                if patch_normalize
                else None
            ),
            (
                transforms.Lambda(lambda sample: ([(item * 2) - 1 for item in sample]))
                if scale_shift_preprocess
                else None
            ),
        ]
        transform_list = [t for t in transform_list if t is not None]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        if self.eager_mode:
            fid = str(idx)
            data = random_crop(self.data[idx], self.patchsize)
        else:
            fid = self.fnames[idx]
            filepath = self.root_dir + fid
            data = self._load_matching_crops(filepath)

        non_hsi_indices = [index for index, item in enumerate(self.df) if item != "hsi"]
        scale_noise = np.random.beta(1, 3, 1)
        for index in non_hsi_indices:
            meas = data[index]
            meanval = np.mean(meas)
            use_sigma = meanval * scale_noise * self.max_sigma_percent
            noise = np.random.randn(*meas.shape) * use_sigma
            noisy_meas = meas + noise
            noisy_meas = np.clip(noisy_meas, 0, noisy_meas.max())
            data[index] = noisy_meas

        data = self.transform(data)
        data = [dat.to(self.dtype) for dat in data]
        data = dict(zip(self.df, data))
        data["context"] = ""
        data["name"] = fid
        return data

    def _load_matching_crops(self, h5file):
        crop_size = self.patchsize
        df = self.df

        with h5py.File(h5file, "r") as hf:
            if self.patchsize is None:
                return [np.array(hf[key]) for key in df]

            dat = hf[df[0]]
            height, width, _ = dat.shape
            if crop_size > height or crop_size > width:
                raise ValueError(
                    "Crop size must be smaller than the dimensions of the data."
                )

            x_start = np.random.randint(0, height - crop_size + 1)
            y_start = np.random.randint(0, width - crop_size + 1)
            return [
                hf[key][
                    x_start : x_start + crop_size,
                    y_start : y_start + crop_size,
                    :,
                ]
                for key in df
            ]


if __name__ == "__main__":
    root_dir = "/home/deanhazineh/ssd4tb_mounted/DiffVis_Official/diffvis/prerendered_ARAD_mixed/"
    dataset = Metadiff(
        root_dir,
        patchsize=64,
        patch_normalize=True,
        scale_shift_preprocess=True,
        # data_fields=["hsi", "rgb", "mgs"],
        data_fields=["mgs", "hsi"],
        eager_mode=False,
        use_aug=False,
        include=["ARAD_valid"],
        max_sigma_percent=0.2,
    )

    sample = dataset[0]
    hsi = sample["hsi"].to(torch.float32)
    mgs = sample["mgs"].to(torch.float32)
    # # rgb = sample["rgb"].to(torch.float32)

    print(hsi.shape, hsi.min(), hsi.max(), hsi.dtype)
    print(mgs.shape, mgs.min(), mgs.max(), mgs.dtype)
    # # print(rgb.shape, rgb.min(), rgb.max(), rgb.dtype)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(reverse_transform(mgs))
    ax[1].imshow(reverse_transform(hsi)[:, :, 10])
    plt.show()
