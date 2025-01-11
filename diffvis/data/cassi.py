from torch.utils.data import Dataset
from natsort import natsorted
import os
from torchvision import transforms
import h5py
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from diffvis.data.group_transforms import *


def random_crop(hsi, crop_size):
    height, width, c = hsi.shape
    if crop_size > height or crop_size > width:
        raise ValueError("Crop size must be smaller than the dimensions of the data.")

    x_start = np.random.randint(0, height - crop_size + 1)
    y_start = np.random.randint(0, width - crop_size + 1)
    return np.array(
        hsi[x_start : x_start + crop_size, y_start : y_start + crop_size, :]
    )


def deshift(meas, step=2, nc=28):
    ch, h, w = meas.shape
    assert ch == 1, "grayscale measurement assertion."
    output = torch.cat([meas[:, :, step * i : step * i + h] for i in range(nc)], dim=0)
    return output


# def shift(inputs, step=2):
#     nC, row, col = inputs.shape
#     output = torch.zeros(
#         nC, row, col + (nC - 1) * step, dtype=inputs.dtype, device=inputs.device
#     )
#     for i in range(nC):
#         output[i, :, step * i : step * i + col] = inputs[i]
#     return output


def shift(inputs, step=2):
    # This is optimized to be a lot faster than looping
    nC, row, col = inputs.shape
    cols_out = col + (nC - 1) * step
    output = torch.zeros(nC, row, cols_out, dtype=inputs.dtype, device=inputs.device)
    base_indices = torch.arange(col, device=inputs.device).expand(nC, -1)
    shift_indices = (
        base_indices + torch.arange(nC, device=inputs.device).unsqueeze(1) * step
    )
    output.scatter_(2, shift_indices.unsqueeze(1).expand(-1, row, -1), inputs)
    return output


class CASSI(Dataset):
    def __init__(
        self,
        root_dir,
        cropsize=256,
        patchsize=64,
        maskpath="/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/datasets/mask.mat",
        maskkey="mask",
        use_aug=True,
        eager_mode=True,
        patch_normalize=True,
        scale_shift_preprocess=True,
        include=[],
        mask_product=False,
        dtype="float32",
        catmask=False,
        random_mask=False,
    ):
        self.root_dir = root_dir
        self.cropsize = cropsize
        self.patchsize = patchsize
        self.eager_mode = eager_mode
        self.use_aug = use_aug
        self.mask_product = mask_product
        self.catmask = catmask
        self.random_mask = random_mask

        if dtype == "float64":
            self.dtype = torch.float64
        elif dtype == "float32":
            self.dtype = torch.float32
        elif dtype == "float16":
            self.dtype = torch.float16
        else:
            raise ValueError("Invalid Datatype")

        self.fnames = natsorted([f for f in os.listdir(root_dir)])
        self.fnames = [
            s for s in self.fnames if any(s.startswith(sub) for sub in include)
        ]
        print(self.fnames)

        if eager_mode:
            self.data = []
            for fname in tqdm(self.fnames, desc="Loading data into RAM"):
                filepath = os.path.join(self.root_dir, fname)
                with h5py.File(filepath, "r") as hdf:
                    self.data.append(np.array(hdf["hsi"]))

        self.mask = torch.tensor(loadmat(maskpath)[maskkey][None], dtype=self.dtype)
        self.to_tensor = transforms.ToTensor()
        self.aug = JointRandomFlip()
        self.patchify = JointRandomCrop([patchsize, patchsize])
        transform_list = [
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
            hsi = random_crop(self.data[idx], self.cropsize)
        else:
            fid = self.fnames[idx]
            filepath = self.root_dir + fid
            hsi = self._load_hsi(filepath)  # cropped

        hsi = self.to_tensor(hsi).to(dtype=self.dtype)
        hsi = hsi / hsi.max()
        if self.use_aug:
            hsi = self.aug([hsi])[0]

        if self.random_mask:
            mask = torch.rand(*hsi.shape[-2:]).round().to(dtype=self.dtype)[None]
        else:
            mask = self.mask

        meas = torch.sum(shift(hsi * mask), axis=0, keepdim=True)
        meas = meas / meas.max()
        dsmeas = deshift(meas)
        if self.mask_product:
            dsmeas = dsmeas * mask

        hsi, dsmeas, meas, mask = self.patchify([hsi, dsmeas, meas, mask])
        hsi, dsmeas, meas = self.transform([hsi, dsmeas, meas])
        if self.catmask:
            dsmeas = torch.cat([dsmeas, mask], axis=0)

        data = dict(zip(["hsi", "mgs", "rmeas", "mask"], [hsi, dsmeas, meas, mask]))
        data["context"] = ""
        return data

    def _load_hsi(self, h5file):
        with h5py.File(h5file, "r") as hf:
            hsi = hf["hsi"]
            return random_crop(hsi, self.cropsize)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dflat.render import hsi_to_rgb

    dataset = CASSI(
        root_dir="/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/datasets/CASSI_Dataset_450_650/",
        cropsize=256,
        patchsize=64,
        maskpath="/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/datasets/mask.mat",
        maskkey="mask",
        use_aug=True,
        eager_mode=False,
        patch_normalize=True,
        scale_shift_preprocess=True,
        include=["ARAD_train", "CAVE_"],
        mask_product=False,
        dtype="float32",
        catmask=True,
        random_mask=False,
    )

    sample = dataset[0]
    hsi = sample["hsi"]
    mgs = sample["mgs"]
    rmeas = sample["rmeas"]
    mask = sample["mask"]
    print(hsi.shape, mgs.shape)

    # gt_rgb = hsi_to_rgb(reverse_transform(hsi), np.linspace(450e-9, 650e-9, 28))

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(gt_rgb)
    # ax[1].imshow(reverse_transform(rmeas))
    # plt.show()
