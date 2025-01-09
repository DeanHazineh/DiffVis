import pytorch_lightning as pl
from diffvis.diffusion.load_utils import get_obj_from_str, instantiate_from_config
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from .HDNet import FDL


class Loss_MRAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / label
        return torch.mean(error)


class Loss_RMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = (outputs - label) ** 2
        return torch.sqrt(torch.mean(error))


class Loss_MAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        return torch.mean(torch.abs((outputs - label)))


class Loss_RMSE_FDL(nn.Module):
    def __init__(self):
        super().__init__()
        self.FDL_loss = FDL(
            loss_weight=0.7,
            alpha=2.0,
            patch_factor=4,
            ave_spectrum=True,
            log_matrix=True,
            batch_matrix=True,
        ).cuda()
        self.alpha = 0.7
        self.RMSE_loss = Loss_RMSE()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        return self.RMSE_loss(outputs, label) + self.alpha * self.FDL_loss(
            outputs, label
        )


class ModelWrapper(pl.LightningModule):
    def __init__(
        self, network_config, input_key, xcond_key, ccond_key, loss, trainable
    ):
        super().__init__()
        self.input_key = input_key
        self.xcond_key = xcond_key
        self.ccond_key = ccond_key
        self.loss = loss
        self.use_dtype = torch.float32
        self.__instantiate_network(network_config, trainable)

        if self.loss == "MRAE":
            self.loss_fn = Loss_MRAE()
        elif self.loss == "RMSE":
            self.loss_fn = Loss_RMSE()
        elif self.loss == "MAE":
            self.loss_fn = Loss_MAE()
        elif self.loss == "RMSE_FDL":
            self.loss_fn = Loss_RMSE_FDL()

    def __instantiate_network(self, config, trainable):
        model = instantiate_from_config(
            config, ckpt_path=config["ckpt_path"], strict=False
        )
        if not trainable:
            model = model.eval()
            for param in model.parameters():
                param.requires_grad = False

        self.model = model
        return

    def training_step(self, batch):
        # matching the keywords used in the diffusion code although xcond will not be used
        xinput, xcond, ccond = self.get_input(batch)
        loss = self.forward(xinput, xcond, ccond)
        return loss

    @torch.no_grad()
    def get_input(self, batch):
        xinput = batch[self.input_key].to(dtype=self.use_dtype, device=self.device)
        xcond = None
        ccond = batch[self.ccond_key].to(dtype=self.use_dtype, device=self.device)
        return (xinput, xcond, ccond)

    def forward(self, x, xcond=None, ccond=None):
        model_output = self.model(ccond)
        pred_loss = self.loss_fn(model_output, x)
        return pred_loss

    def sample(self, ccond):
        if not torch.is_tensor(ccond):
            ccond = torch.tensor(ccond, device="cuda", dtype=torch.float32)
        ccond = ccond / torch.amax(ccond, axis=(-1, -2, -3), keepdim=True)

        model_out = self.model(ccond)
        model_out = model_out / torch.amax(
            model_out, axis=(-1, -2, -3), keepdim=True
        )  # Renormalize HSI to range of 0 and 1

        model_out = torch.clamp(model_out, 0, 1)
        return model_out.detach().cpu().numpy()
