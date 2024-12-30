import os
from scipy.io import loadmat
import numpy as np
import torch
import torch.nn.functional as F

import pickle
from natsort import natsorted
import pandas as pd
import random
from torchvision import transforms

from diffvis.diffusion import initialize_diffusion_model, DiffSSI
from dflat.render import hsi_to_rgb, general_convolve
from diffvis.diffusion.eval_utils import *
from diffvis.data import reverse_transform, permute_dimensions


def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


##############################################################
set_seeds(42)
datpath = "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/datasets/ARAD1k_repackaged/"
root_dir = (
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/models/metalens_arad256_models/"
)
psf_path = "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/metasurfaces/L4s_lens_psf_compact32.pickle"
with open(psf_path, "rb") as file:
    data = pickle.load(file)
    psf = data["psf_int"]
    psf = psf / np.max(psf)
    wl = data["wl"]
    lam = wl
psf = torch.tensor(psf, dtype=torch.float32).to(device="cuda")

patch_normalize = True
patch_size = [64, 64]
stride_size = None
variance_n = 2
gloop = 1
gscale = 10
eta = 1.0
ddim_steps = 50

modeln_list = ["rgb", "rgb3", "mrgb", "mrgb3", "mgs"]
cond_list = ["rgb", "rgb3", "mrgb", "mrgb3", "mgs"]
mode_list = ["rgb", "rgb3", "rgb", "rgb3", "gs"]

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize([256, 256]),
    ]
)

for modeln, cond, mode in zip(modeln_list, cond_list, mode_list):
    savepath = root_dir + f"Samples/{modeln}/"
    os.makedirs(savepath, exist_ok=True)

    model_path = root_dir + modeln + "/config.yaml"
    model_ckpt = root_dir + modeln + "/model_snapshots/ckpt_last.ckpt"
    diffusion_model = initialize_diffusion_model(model_path, model_ckpt).to("cuda")

    use_kernel = psf[None] if cond not in ["rgb", "rgb3"] else None
    sampler = DiffSSI(
        diffusion_model,
        n_steps=ddim_steps,
        ddim_eta=eta,
        gloop=gloop,
        gscale=gscale,
        kernel=use_kernel,
        mode=mode,
        lam=lam,
        patch_normalize=patch_normalize,
    )

    fnames = natsorted(os.listdir(datpath))
    fnames = [f for f in fnames if "_valid" in f]
    for f in fnames:
        f = f.strip(".mat")
        fid = f.strip("ARAD_valid_")
        outpath = savepath + f"{fid}.pickle"

        datfile = os.path.join(datpath, f)
        hsi = loadmat(datfile)["hsi"]
        hsi = transform(hsi).to("cuda")
        hsi = hsi / hsi.max()
        mhsi = general_convolve(hsi, psf, rfft=True)
        mgs = torch.sum(mhsi, dim=0, keepdim=True)
        mgs = mgs / torch.max(mgs)
        mrgb = hsi_to_rgb(
            mhsi,
            lam,
            tensor_ordering=True,
            raw=True,
            projection="Basler_Bayer",
            normalize=True,
        )
        rgb = hsi_to_rgb(
            hsi,
            lam,
            tensor_ordering=True,
            raw=True,
            projection="Basler_Bayer",
            normalize=True,
        )
        mrgb3 = hsi_to_rgb(
            mhsi,
            lam,
            tensor_ordering=True,
            raw=False,
            projection="Basler_Bayer",
            normalize=True,
        )
        rgb3 = hsi_to_rgb(
            hsi,
            lam,
            tensor_ordering=True,
            raw=False,
            projection="Basler_Bayer",
            normalize=True,
        )
        meas = {"mgs": mgs, "mrgb": mrgb, "rgb": rgb, "mrgb3": mrgb3, "rgb3": rgb3}

        if os.path.exists(outpath):
            print(f"Loading previous save {outpath}")
            with open(os.path.join(savepath, f"{fid}.pickle"), "rb") as fhandle:
                data = pickle.load(fhandle)
                ims = data["est"][None]
                ims_ref = data["ref_est"][None]
        else:
            print(f"Drawing new sample saveto: {outpath}")
            ims, _, losses = sampler.sample(
                meas[cond],
                use_guidance=True,
                variance_n=variance_n,
                patch_size=patch_size,
                stride_size=stride_size,
            )
            ims_ref, _, losses_ref = sampler.sample(
                meas[cond],
                use_guidance=False,
                variance_n=variance_n,
                patch_size=patch_size,
                stride_size=stride_size,
            )

        # Compute Evaluation
        gt_hsi = hsi.cpu().numpy()
        gt_hsi = gt_hsi / np.mean(gt_hsi, axis=(-1, -2, -3), keepdims=True)
        est = ims[-1]
        est /= np.mean(est, axis=(-1, -2, -3), keepdims=True)
        avg_est = np.mean(est, axis=0, keepdims=True)
        ref_est = ims_ref[-1]
        ref_est /= np.mean(ref_est, axis=(-1, -2, -3), keepdims=True)
        avg_ref_est = np.mean(ref_est, axis=0, keepdims=True)

        # compute metrics
        out = compute_eval(est, gt_hsi, drop_perc=[0, 1, 5])
        out_ref = compute_eval(ref_est, gt_hsi, drop_perc=[0, 1, 5])

        out["gt_hsi"] = hsi.cpu().numpy()
        out["est"] = est
        out["avg_est"] = avg_est
        out["ref_est"] = ref_est
        out["avg_ref_est"] = avg_ref_est
        out["datname"] = fid
        out["ccond"] = meas[cond]
        out_ref["datname"] = "ref" + fid

        with open(os.path.join(savepath, f"{fid}.pickle"), "wb") as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(savepath, f"{fid}_no_guidance.pickle"), "wb") as handle:
            pickle.dump(out_ref, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Repackage each simulation into a final data
        data_list = []
        filenames = natsorted(os.listdir(savepath))
        for fname in filenames:
            if fname.endswith(".csv"):
                continue
            with open(os.path.join(savepath, fname), "rb") as handle:
                data = pickle.load(handle)
                datname = data["datname"]
                new_row = {
                    "datname": data["datname"],
                    "ssim": data["ssim"],
                    "sam": data["sam"],
                    "sdi": data["sdi"],
                    "ppsnr": data["ppsnr"],
                    "spsnr": data["spsnr"],
                }
                data_list.append(new_row)
        df = pd.DataFrame(data_list)
        df.set_index("datname", inplace=True)

        output_filename = root_dir + f"Samples/{modeln}_summary.csv"
        df.to_csv(output_filename)
        print(f"Data saved to {output_filename}")
