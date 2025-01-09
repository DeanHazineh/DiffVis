from diffvis.diffusion import initialize_training

configs = [
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/models/metalens_arad256_models/mgs/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/models/metalens_arad256_models/mgs_nopn/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/models/metalens_arad256_models/mgs_patch32/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/models/metalens_arad256_models/mrgb/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/models/metalens_arad256_models/mrgb3/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/models/metalens_arad256_models/rgb/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/models/metalens_arad256_models/rgb3/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/models/metalens_arad256_models/mrgb/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/models/lens_sweep_models/AIF/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/models/lens_sweep_models/L1/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/models/lens_sweep_models/L2/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/models/lens_sweep_models/L4v2/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/models/lens_sweep_models/L4s/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/models/lens_sweep_models/L8/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/models/icvl_model/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/priormodels/mgs/UNet/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/priormodels/mgs/TSANet/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/priormodels/mgs/MST/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/priormodels/mgs/HDNet/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/priormodels/mgs/DGSMP/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/priormodels/mgs/DAUHST/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/priormodels/mrgb/DAUHST/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/priormodels/mrgb/DGSMP/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/priormodels/mrgb/HDNet/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/priormodels/mrgb/MST/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/priormodels/mrgb/TSANet/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/priormodels/mrgb/UNet/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/priormodels/rgb/DAUHST/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/priormodels/rgb/DGSMP/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/priormodels/rgb/HDNet/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/priormodels/rgb/MST/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/priormodels/rgb/TSANet/config.yaml",
    "/home/deanhazineh/ssd4tb_mounted/DiffVis/scripts/priormodels/rgb/UNet/config.yaml",
]

for config in configs:
    trainer = initialize_training(config, override_eager=True)
    trainer.fit()
