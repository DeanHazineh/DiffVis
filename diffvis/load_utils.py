from omegaconf import OmegaConf
import importlib
import torch
import os


# Update so all path should be relative to diffvis root
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def to_root_path(path):
    if path is None or path == "None":
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(SRC_ROOT, path)


def initialize_training(
    config_path,
    model_ckpt_path=None,
    override_eager=False,
):
    config_path = to_root_path(config_path)
    config = OmegaConf.load(config_path)

    config_trainer = config.trainer
    config_train_dat = config.train_dataset
    config_valid_dat = (
        config.valid_dataset if "valid_dataset" in config.keys() else None
    )

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
    trainer = get_obj_from_str(config_trainer["target"])(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        **trainer_params,
    )

    return trainer


def initialize_diffusion_model(
    config_path, ckpt_path=None, strict=True, grad_checkpoint_override=False
):
    # Get the config yaml and initalize the full object
    config_path = to_root_path(config_path)
    config = OmegaConf.load(config_path)
    config_model = config.model

    # Optionally override gradient checkpointing setting before loading the model
    if grad_checkpoint_override:
        config_model.params.unet_config.params.use_checkpoint = True

    print(f"Target Module: {config_model.target}")
    diffusion_model = instantiate_from_config(config_model)

    # If given, initialize strict from a checkpoint
    ckpt_path = to_root_path(ckpt_path)
    if ckpt_path is not None:
        print(f"Loading from checkpoint {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = diffusion_model.load_state_dict(sd, strict=strict)
        print(
            f"Restored with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )

    return diffusion_model


def instantiate_from_config(config_model, ckpt_path=None, strict=False):
    if not "target" in config_model:
        raise KeyError("Expected key `target` to instantiate.")
    target_str = config_model["target"]
    print(target_str)
    loaded_module = get_obj_from_str(target_str)(**config_model.get("params", dict()))

    # Get model checkpoint
    ckpt_path = to_root_path(ckpt_path)
    if ckpt_path is not None and ckpt_path != "None":
        print(
            f"Target: {config_model['target']} Loading from checkpoint {ckpt_path} as strict={strict}"
        )
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        sd = {key.replace("model.", ""): value for key, value in sd.items()}

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
