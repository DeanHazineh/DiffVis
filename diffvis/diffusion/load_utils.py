from omegaconf import OmegaConf
import importlib
import torch


def initialize_training(
    config_path,
    model_ckpt_path=None,
    override_train_maxsteps=None,
    override_eager=False,
):
    config = OmegaConf.load(config_path)

    config_trainer = config.trainer
    config_train_dat = config.train_dataset
    config_valid_dat = (
        config.valid_dataset if "valid_dataset" in config.keys() else None
    )

    ## used for quick testing:
    if override_eager:
        config_train_dat.params.eager_mode = False

    # Instantiate the model
    model = initialize_diffusion_model(config_path, model_ckpt_path)
    model = model.to("cuda")

    # Load the datasets
    train_dataset = instantiate_from_config(config_train_dat)
    valid_dataset = (
        instantiate_from_config(config_valid_dat)
        if config_valid_dat is not None
        else None
    )

    # Initialize the trainer
    print(f"Trainer Module: {config_trainer.target}")
    trainer_params = config_trainer.get("params", dict())
    if override_train_maxsteps is not None:
        trainer_params["max_steps"] = override_train_maxsteps

    trainer = get_obj_from_str(config_trainer["target"])(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        **trainer_params,
    )

    return trainer


# The weighted training is not used in this version
def initialize_scale_training(config_path, model_ckpt_path=None):
    config = OmegaConf.load(config_path)

    # Instantiate the model
    model = initialize_diffusion_model(config_path, model_ckpt_path)
    model = model.to("cuda")

    # Load the datasets
    config_train_dat = config.train_dataset
    train_prob = []
    train_datasets = []
    for di in config_train_dat.keys():
        dat = config_train_dat[di]
        train_prob.append(dat["prob"])
        dat = instantiate_from_config(dat)
        train_datasets.append(dat)

    valid_dataset = instantiate_from_config(config.valid_dataset)

    # Initialize the trainer
    config_trainer = config.trainer
    print(f"Trainer Module: {config_trainer.target}")
    trainer_params = config_trainer.get("params", dict())
    trainer = get_obj_from_str(config_trainer["target"])(
        model=model,
        train_datasets=train_datasets,
        train_prob=train_prob,
        valid_dataset=valid_dataset,
        **trainer_params,
    )

    return trainer


def initialize_diffusion_model(
    config_path, ckpt_path=None, strict=True, grad_checkpoint_override=False
):
    # Get the config yaml and initalize the full object
    config = OmegaConf.load(config_path)
    config_model = config.model

    if grad_checkpoint_override:
        config_model.params.unet_config.params.use_checkpoint = True

    print(f"Target Module: {config_model.target}")
    diffusion_model = instantiate_from_config(config_model)

    # If given, initialize strict from a checkpoint
    if ckpt_path is not None:
        print(f"Loading from checkpoint {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = diffusion_model.load_state_dict(sd, strict=strict)
        print(
            f"Restored with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )

    return diffusion_model


def instantiate_from_config(config_model, ckpt_path=None, strict=False):
    if not "target" in config_model:
        raise KeyError("Expected key `target` to instantiate.")
    target_str = config_model["target"]
    loaded_module = get_obj_from_str(target_str)(**config_model.get("params", dict()))

    # Get model checkpoint
    if ckpt_path is not None and ckpt_path != "None":
        print(
            f"Target: {config_model['target']} Loading from checkpoint {ckpt_path} as strict={strict}"
        )
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]

        # Allow to load from model checkpoint for transfer learning
        sd = {key.replace("model.", ""): value for key, value in sd.items()}

        # # (Note: When we use SD/compvis Latent Diffusion Model checkpoints, we need to fix the names)
        # # Having this check in all cases wont hurt since the key tag wont be met
        # sd = {
        #     key.replace("model.diffusion_model.", ""): value
        #     for key, value in sd.items()
        # }
        # sd = {key.replace("first_stage_model.", ""): value for key, value in sd.items()}

        missing, unexpected = loaded_module.load_state_dict(sd, strict=strict)
        print(
            f"Restored {target_str} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        print(missing)
        print(unexpected)

    return loaded_module


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)

    return getattr(importlib.import_module(module, package=None), cls)
