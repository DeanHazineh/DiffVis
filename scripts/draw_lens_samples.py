import os
from scipy.io import loadmat
import numpy as np
import torch
import h5py
import pickle
from natsort import natsorted
import pandas as pd
import random
from natsort import natsorted
from torchvision import transforms

from dflat.render import hsi_to_rgb, general_convolve

from diffvis.diffusion import initialize_diffusion_model, DiffSSI
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
datpath = "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/datasets/ARAD1k_repackaged/"
model_root = (
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/models/lens_sweep_models/"
)
model_names = [
    "AIF/",
    "L1/",
    "L2/",
    "L4v2/",
    "L4s/",
    "L8/",
]

mode = "gs"
patch_normalize = True
patch_size = [64, 64]
variance_n = 2
gloop = 1
gscale = 10
eta = 1.0
stride_size = None

set_seeds(42)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize([256, 256]),
    ]
)
fnames = natsorted(os.listdir(datpath))
fnames = [f for f in fnames if "_valid" in f]
for modeln in model_names:
    savepath = model_root + "Samples/" + modeln
    os.makedirs(savepath, exist_ok=True)

    psf_path = model_root + modeln + "psf_compact32.pickle"
    if os.path.exists(psf_path):
        with open(psf_path, "rb") as file:
            data = pickle.load(file)
            psf = data["psf_int"]
            psf = psf / np.max(psf)
            wl = data["wl"]
        psf = torch.tensor(psf, dtype=torch.float32).to(device="cuda")
    else:
        psf = None

    model_path = model_root + modeln + "config.yaml"
    ckpt_path = model_root + modeln + "model_snapshots/ckpt_last.ckpt"
    diffusion_model = initialize_diffusion_model(model_path, ckpt_path).to("cuda")
    lam = np.linspace(400e-9, 700e-9, 31)
    sampler = DiffSSI(
        diffusion_model,
        n_steps=50,
        ddim_eta=eta,
        gloop=gloop,
        gscale=gscale,
        kernel=psf[None] if psf is not None else None,
        mode=mode,
        lam=lam,
        patch_normalize=patch_normalize,
    )

    for f in fnames[:1]:
        f = f.strip(".mat")
        fid = f.strip("ARAD_valid_")
        outpath = savepath + f"{fid}.pickle"

        datfile = os.path.join(datpath, f)
        hsi = loadmat(datfile)["hsi"]
        hsi = transform(hsi).to("cuda")
        hsi = hsi / hsi.max()

        if psf is None:
            cond = torch.sum(hsi, dim=0, keepdim=True)
            cond = cond / torch.max(cond)
        else:
            mhsi = general_convolve(hsi, psf, rfft=True)
            cond = torch.sum(mhsi, dim=0, keepdim=True)
            cond = cond / torch.max(cond)

        if os.path.exists(outpath):
            print(f"Loading previous save {outpath}")
            with open(os.path.join(savepath, f"{fid}.pickle"), "rb") as fhandle:
                data = pickle.load(fhandle)
                ims = data["est"][None]
                ims_ref = data["ref_est"][None]
        else:
            ims, _, losses = sampler.sample(
                cond,
                use_guidance=True,
                variance_n=variance_n,
                patch_size=patch_size,
                stride_size=stride_size,
            )
            ims_ref, _, losses_ref = sampler.sample(
                cond,
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
        out["ccond"] = cond
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
        output_filename = model_root + "Samples/" + modeln[:-1] + "_summary.csv"
        df.to_csv(output_filename)
        print(f"Data saved to {output_filename}")
