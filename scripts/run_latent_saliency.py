import os
import pickle
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm
from scipy.io import loadmat
from torchvision import transforms
from dflat.render import hsi_to_rgb, general_convolve
from diffvis.diffusion.sampler_guided_ddim import DiffSSI
from diffvis.diffusion import initialize_diffusion_model
from diffvis.data.metavis import Metadiff

# Paths
BASE_PATH = "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/"
DATAPATH = os.path.join(BASE_PATH, "datasets/metalens_prerendered/")
PSFPATH = os.path.join(BASE_PATH, "metasurfaces/")
MODELPATH = os.path.join(BASE_PATH, "models/lens_sweep_models/")
SAVEBASE = os.path.join(BASE_PATH, "latent_saliency_output/")

# Dataset and Model Configurations
DATASET_FOLDS = [
    os.path.join(DATAPATH, subfolder)
    for subfolder in [
        "prerendered_L8s_256/",
        "prerendered_L4s_256/",
        "prerendered_L4v2_256/",
        "prerendered_L2_256/",
        "prerendered_L1_256/",
        "prerendered_AIF_256/",
    ]
]

PSF_PATHS = [
    os.path.join(PSFPATH, file)
    for file in [
        "L8s_lens_psf_compact32.pickle",
        "L4s_lens_psf_compact32.pickle",
        "L4v2_lens_psf_compact32.pickle",
        "L2_lens_psf_compact32.pickle",
        "L1_lens_psf_compact32.pickle",
    ]
] + ["_NONE_"]

MODEL_DIRS = [
    os.path.join(MODELPATH, subfolder)
    for subfolder in ["L8/", "L4s/", "L4v2/", "L2/", "L1/", "AIF/"]
]

NAMES = ["L8/", "L4s/", "L4v2/", "L2/", "L1/", "AIF/"]

# Loop through each configuration
for dataset_path, psf_path, model_dir, name in zip(
    DATASET_FOLDS, PSF_PATHS, MODEL_DIRS, NAMES
):
    # Initialize Dataset
    dataset = Metadiff(
        root_dir=dataset_path,
        patchsize=64,
        patch_normalize=True,
        scale_shift_preprocess=False,
        data_fields=["mgs"],
        use_aug=False,
        include=["ARAD_valid"],
        dtype="float32",
    )

    # Load PSF
    if psf_path == "_NONE_":
        psf = None
    else:
        with open(psf_path, "rb") as file:
            data = pickle.load(file)
            psf = torch.tensor(
                data["psf_int"] / np.max(data["psf_int"]), dtype=torch.float32
            ).to("cuda")[None]

    # Initialize Model
    model_config = os.path.join(model_dir, "config.yaml")
    ckpt_path = os.path.join(model_dir, "model_snapshots/ckpt_last.ckpt")
    diffusion_model = initialize_diffusion_model(
        model_config, ckpt_path, grad_checkpoint_override=False
    ).to("cuda")

    sampler = DiffSSI(
        diffusion_model,
        n_steps=50,
        ddim_eta=0.0,
        gloop=1,  # this is not used because we draw samples with use_guidance = False
        gscale=10,
        kernel=psf,
        mode="gs",
    )

    # Prepare Save Path
    savepath = os.path.join(SAVEBASE, name)
    os.makedirs(savepath, exist_ok=True)

    dl = iter(dataset)
    x_start = torch.randn((1, 31, 64, 64))
    variance_n, patch_size, repeat, res = 1, [64, 64], 20, 1

    for i in range(repeat):
        try:
            sample = next(dl)
        except StopIteration:
            dl = iter(dataset)
            sample = next(dl)

        mgs = sample["mgs"]
        saveto = os.path.join(savepath, f"store_{i}.pkl")

        if os.path.exists(saveto):
            print(f"Skipping {saveto}")
            continue

        # Generate Reference Images
        ims_ref, _, _ = sampler.sample(
            mgs,
            x_start=x_start,
            use_guidance=False,
            variance_n=variance_n,
            patch_size=patch_size,
        )

        # Generate Perturbed Images
        lx = len(np.arange(8, 56, res))
        out = np.zeros((lx, lx, 1, 1, 31, 64, 64))
        for ir, r in enumerate(np.arange(8, 56, res)):
            for ic, c in enumerate(np.arange(8, 56, res)):
                meas = deepcopy(mgs)
                meas[:, r, c] = 0.0

                im, _, _ = sampler.sample(
                    meas,
                    x_start=x_start,
                    use_guidance=False,
                    variance_n=variance_n,
                    patch_size=patch_size,
                )
                out[ir, ic] = im

        # Save Results
        with open(saveto, "wb") as f:
            pickle.dump({"ims_ref": ims_ref, "out": out, "meas": meas}, f)

        print(f"Saved {saveto}")
