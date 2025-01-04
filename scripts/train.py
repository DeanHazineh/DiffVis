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
]

for config in configs:
    trainer = initialize_training(config, override_eager=True)
    trainer.fit()
