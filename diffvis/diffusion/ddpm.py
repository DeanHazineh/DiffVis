from abc import ABC, abstractmethod
from functools import partial
import random
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from einops import rearrange

from .load_utils import get_obj_from_str, instantiate_from_config
from .ddpm_utils import *


class GaussianDiffusion(pl.LightningModule, ABC):
    def __init__(
        self,
        unet_config,
        xcond_stage_config,
        ccond_stage_config,
        noise_channels,
        input_key,
        xcond_key,
        ccond_key,
        use_xcond=False,
        use_ccond=False,
        cfg_dropout=0.00,
        use_cfg=False,
        cfg_gamma=1.0,
        timesteps=1000,
        beta_schedule="quadratic",  # compvis/stability refers to this scheduler as "linear"
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        loss_type="l1",
        unet_trainable=True,
        xcond_stage_trainable=False,
        ccond_stage_trainable=False,
        use_fp16=False,
    ):
        super().__init__()
        self.input_key = input_key
        self.xcond_key = xcond_key
        self.ccond_key = ccond_key
        self._loss_type = loss_type
        self._seed_channels = noise_channels
        self._use_xcond = use_xcond
        self._use_ccond = use_ccond
        self.use_cfg = use_cfg
        self.cfg_dropout = cfg_dropout
        self.cfg_gamma = cfg_gamma
        self.use_dtype = torch.float16 if use_fp16 else torch.float32
        if use_cfg:
            assert use_xcond or use_ccond, "CFG requires some form of guidance."
        self.bstart = linear_start
        self.bend = linear_end
        self.bcos = cosine_s
        self.beta_schedule = beta_schedule

        self.__instantiate_denoise_model(unet_config, unet_trainable)
        self.__register_schedule(
            timesteps, beta_schedule, linear_start, linear_end, cosine_s
        )
        self.__instantiate_xcond_stage(xcond_stage_config, xcond_stage_trainable)
        self.__instantiate_ccond_stage(ccond_stage_config, ccond_stage_trainable)

    def __instantiate_denoise_model(self, config, trainable):
        model = instantiate_from_config(
            config, ckpt_path=config["ckpt_path"], strict=False
        )
        if not trainable:
            model = model.eval()
            model.train = disabled_train
            for param in model.parameters():
                param.requires_grad = False

        self.model = model
        return

    def __register_schedule(
        self, timesteps, beta_schedule, linear_start, linear_end, cosine_s
    ):
        self.timesteps = timesteps

        # Define the beta schedule
        betas = make_beta_schedule(
            beta_schedule, timesteps, linear_start, linear_end, cosine_s
        ).to(self.use_dtype)
        assert betas.ndim == 1, "betas must be 1-D"
        assert (0 < betas).all() and (betas <= 1).all(), "betas must be in (0..1]"
        self.betas = betas

        # Define alphas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.alphas_cumprod = alphas_cumprod

        one_minus_alphas_cumprod = 1.0 - alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(one_minus_alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_coeff1 = (
            torch.sqrt(alphas_cumprod_prev) * betas / one_minus_alphas_cumprod
        )
        self.posterior_coeff2 = (
            torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (one_minus_alphas_cumprod)
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (one_minus_alphas_cumprod)
        )
        self.posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance)
        # self.posterior_log_variance_clipped = torch.log(
        #     torch.cat(
        #         (self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:])
        #     )
        # ) # Snippet from improved diffusion initialization

        return

    def __instantiate_xcond_stage(self, config, trainable):
        cond_type = config["target"]

        if cond_type == "__is_unconditional__":
            print(
                f"Training {self.__class__.__name__} without cross-attention conditioning."
            )
            self.xcond_stage_model = None

        elif cond_type == "__is_identity__":
            print("Using identity as the cross-attention stage.")
            self.xcond_stage_model = identity

        else:
            print(
                f"Attempting to import cross-attention conditioning model: {cond_type}"
            )
            model = instantiate_from_config(
                config, ckpt_path=config["ckpt_path"], strict=False
            )
            if not trainable:
                model = model.eval()
                model.train = disabled_train
                for param in model.parameters():
                    param.requires_grad = False
            self.xcond_stage_model = model

        return

    def __instantiate_ccond_stage(self, config, trainable):
        cond_type = config["target"]

        if cond_type == "__is_unconditional__":
            print(
                f"Training {self.__class__.__name__} without concatenation conditioning."
            )
            self.ccond_stage_model = None

        elif cond_type == "__is_identity__":
            print("Using identity as the concatenation transform.")
            self.ccond_stage_model = identity

        else:
            print(f"Attempting to import cond model: {cond_type}")
            model = instantiate_from_config(
                config, ckpt_path=config["ckpt_path"], strict=False
            )
            if not trainable:
                model = model.eval()
                model.train = disabled_train
                for param in model.parameters():
                    param.requires_grad = False
            self.ccond_stage_model = model

        return

    def forward(self, x, xcond=None, ccond=None):
        t = torch.randint(
            0, self.timesteps, (x.shape[0],), device=self.device, dtype=torch.long
        )

        if self._use_xcond and xcond is not None:
            xcond = self.xcond_stage_model(xcond)

        if self._use_ccond and ccond is not None:
            ccond = self.ccond_stage_model(ccond)

        # Dropout for CFG training
        if self.use_cfg:
            dp = torch.rand((x.shape[0],), device="cuda")[:, None, None, None]
            if ccond is not None:
                ccond = torch.where(
                    dp < self.cfg_dropout, torch.zeros_like(ccond), ccond
                )
            if xcond is not None:
                xcond = torch.where(
                    dp < self.cfg_dropout, torch.zeros_like(xcond), xcond
                )

        return self.p_losses(x, t, xcond_embd=xcond, ccond_embd=ccond)

    def training_step(self, batch):
        xinput, xcond, ccond = self.get_input(batch)
        loss = self.forward(xinput, xcond, ccond)
        return loss

    @torch.no_grad()
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def sample(
        self,
        image_size,
        batch_size=16,
        variance_n=1,
        return_intermediate=True,
        x_start=None,
        xcond=None,
        ccond=None,
        clip=True,  # Generally we always use this for DDPM
    ):
        device = next(self.model.parameters()).device
        if x_start is None:
            x_start = torch.randn(
                (variance_n * batch_size, self._seed_channels, *image_size),
                dtype=self.use_dtype,
                device=device,
            )
        else:
            x_start = x_start.to(dtype=self.use_dtype, device=self.device)
            x_start = self.reshape_batched_variance(x_start, variance_n)

        if self._use_ccond:
            assert ccond is not None
            ccond = ccond.to(dtype=self.use_dtype, device=self.device)
            ccond = self.reshape_batched_variance(ccond, variance_n)
            ccond = self.ccond_stage_model(ccond)
        else:
            ccond = None

        if self._use_xcond:
            assert xcond is not None
            if (
                isinstance(xcond, float)
                or isinstance(xcond, np.ndarray)
                or isinstance(xcond, torch.Tensor)
            ):
                xcond = xcond.to(dtype=self.use_dtype, device=self.device)
            xcond = self.xcond_stage_model(xcond).to(dtype=self.use_dtype)
            xcond = self.reshape_batched_variance(xcond, variance_n)
        else:
            xcond = None

        out, x0s = self.p_sample_loop(
            x_start,
            xcond,
            ccond,
            return_intermediate,
            clip,
        )
        out = rearrange(out, "t (v b) c h w -> t b v c h w", b=batch_size, v=variance_n)
        x0s = rearrange(x0s, "t (v b) c h w -> t b v c h w", b=batch_size, v=variance_n)

        return out.numpy(), x0s.numpy()

    @torch.no_grad()
    def p_sample_loop(
        self,
        im,
        xcond_embd=None,
        ccond_embd=None,
        return_intermediate=True,
        clip=True,
    ):
        device = next(self.model.parameters()).device

        batch_size = im.shape[0]
        img_shape = im.shape[1:]  # Get the shape of each image tensor
        total_steps = self.timesteps if return_intermediate else 1
        imgs = torch.zeros((total_steps, batch_size, *img_shape), device="cpu")
        x0s = torch.zeros((total_steps, batch_size, *img_shape), device="cpu")

        step_idx = 0
        for i in tqdm(
            reversed(range(self.timesteps)),
            desc="sampling loop time step",
            total=self.timesteps,
        ):

            x0t, im = self.p_sample(
                im,
                torch.full((im.shape[0],), i, device=device, dtype=torch.long),
                i,
                xcond_embd,
                ccond_embd,
                clip=clip,
                return_pred_start=True,
            )
            if (i == 0) or return_intermediate:
                imgs[step_idx] = im.cpu()
                x0s[step_idx] = x0t.cpu()
                step_idx += 1

        return imgs, x0s

    @staticmethod
    def reshape_batched_variance(x, variance_n):
        x = x.unsqueeze(0).repeat(variance_n, *([1] * x.ndim))
        x = x.view((-1, *x.shape[2:]))
        return x

    @abstractmethod
    def get_input(self, batch):
        """Training_step can be kept the same for LDM vs pixel-space ddpm.
        We can instead override get_input() in the child latent class to work in the latent space for LDM.
        This must return a list for use in training step."""
        pass

    @abstractmethod
    def p_losses(self, x_start, t, noise=None, xcond_embd=None, ccond_embd=None):
        pass

    @abstractmethod
    def p_sample(
        self,
        x,
        t,
        t_index,
        xcond_embd=None,
        ccond_embd=None,
        clip=True,
        return_pred_start=False,
    ):
        pass


class DDPM(GaussianDiffusion):
    def __init__(
        self,
        prediction_type: str = ModelMeanType.EPSILON,
        variance_type: str = ModelVarType.FIXED_SMALL,
        kmin_snr=5.0,
        use_min_snr=True,
        mask_loss=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert (
            prediction_type in ModelMeanType.__members__.values()
        ), "'prediction_type' must be a member of ModelMeanType."
        assert (
            variance_type in ModelVarType.__members__.values()
        ), "'variance_type' must be a member of ModelVarType."

        self.variance_type = variance_type
        self.prediction_type = prediction_type
        self.kmin_snr = kmin_snr
        self.use_min_snr = use_min_snr

        # Modification for Patch DiffSSI: Mask_loss to central region
        if mask_loss is None:
            self.mask = None
        else:
            h, H = mask_loss
            mask = torch.zeros((H, H), dtype=torch.float32, device="cuda")
            start = (H - h) // 2
            end = start + h
            mask[start:end, start:end] = 1.0
            self.mask = mask

        if self.prediction_type != "epsilon" and self.use_cfg:
            raise ValueError("CFG is only implemented for epsilon prediction.")

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_coeff1 = extract(self.posterior_coeff1, t, x_t.shape)
        posterior_coeff2 = extract(self.posterior_coeff2, t, x_t.shape)

        posterior_mean = posterior_coeff1 * x_start + posterior_coeff2 * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        return posterior_mean, posterior_variance

    def p_model_mean_variance(self, model_output, x, t, clip_denoised=False):
        x_shape = x.shape
        # Get the model output and possibly the variance
        if self.variance_type in ["learned", "learned_range"]:
            assert (
                model_output.shape[1] == x_shape[1] * 2
            ), "Model output channels not consistent for learned variance."
            model_output, model_var = torch.split(model_output, x_shape[1], dim=1)

            if self.variance_type == "learned":
                posterior_variance_t = torch.exp(model_var)  # Learn log variance
            else:
                min_log = extract(self.posterior_log_variance_clipped, t, x_shape)
                max_log = extract(torch.log(self.betas), t, x_shape)
                frac = (model_var + 1) / 2
                posterior_variance_t = torch.exp(frac * max_log + (1 - frac) * min_log)

        elif self.variance_type in ["fixed_small", "fixed_large"]:
            if self.variance_type == "fixed_small":
                posterior_variance_t = extract(
                    self.posterior_variance,
                    t,
                    x_shape,
                )
            elif self.variance_type == "fixed_large":
                posterior_variance_t = extract(
                    torch.concat((self.posterior_variance[1:2], self.betas[1:])),
                    t,
                    x_shape,
                )

        else:
            raise ValueError("Unknown variance type error.")

        # Predict the mean and xstart
        if self.prediction_type == "previous_x":
            # (xprev - coef2*x_t) / coef1
            posterior_coeff1 = extract(self.posterior_coeff1, t, x_shape)
            posterior_coeff2 = extract(self.posterior_coeff2, t, x_shape)
            pred_xstart = (model_output - posterior_coeff2 * x) / posterior_coeff1
            if clip_denoised:
                pred_xstart = torch.clamp(pred_xstart, -1, 1)
            model_mean = model_output

        elif self.prediction_type in ["epsilon", "start_x"]:
            if self.prediction_type == "epsilon":
                sqrt_one_minus_alphas_cumprod_t = extract(
                    self.sqrt_one_minus_alphas_cumprod, t, x.shape
                )
                sqrt_recip_alphas_cumprod_t = extract(
                    self.sqrt_recip_alphas_cumprod, t, x.shape
                )
                pred_xstart = (
                    x - sqrt_one_minus_alphas_cumprod_t * model_output
                ) * sqrt_recip_alphas_cumprod_t
            elif self.prediction_type == "start_x":
                pred_xstart = model_output

            if clip_denoised:
                pred_xstart = torch.clamp(pred_xstart, -1, 1)

            model_mean, _ = self.q_posterior_mean_variance(pred_xstart, x, t)

        else:
            raise ValueError("Unknown mean prediction type error.")

        return model_mean, posterior_variance_t, pred_xstart

    def p_losses(self, x_start, t, noise=None, xcond_embd=None, ccond_embd=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start, t, noise)
        # if self.mask is not None:
        #     x_noisy = x_noisy * self.mask[None, None]
        loss = 0

        # min-snr
        loss_weight = torch.ones_like(t)
        if self.use_min_snr:
            kmin_snr = self.kmin_snr
            alphas = extract(self.sqrt_alphas_cumprod, t, t.shape)
            sigma = extract(self.sqrt_one_minus_alphas_cumprod, t, t.shape)
            snr = (alphas / sigma) ** 2
            loss_weight = torch.minimum(snr, kmin_snr * torch.ones_like(snr)) / snr

        # Get the model output
        model_output = self.model(
            (torch.cat((x_noisy, ccond_embd), dim=1) if self._use_ccond else x_noisy),
            t,
            context=xcond_embd,
        )

        # # Add vb loss for learned variance from improved diffusion paper
        if self.variance_type in ["learned", "learned_range"]:
            true_mean, _ = self.q_posterior_mean_variance(x_start, x_noisy, t)
            true_log_variance = extract(
                self.posterior_log_variance_clipped, t, x_noisy.shape
            )

            # Model should output double the dimensionality since variance is learned
            B, C, res_shape = *x_noisy.shape[:2], x_noisy.shape[2:]
            assert model_output.shape == (B, C * 2, *res_shape)
            model_output, model_var_values = torch.split(model_output, C, dim=1)

            # Learn the variance using the variational bound, but don't let
            # it affect our mean prediction.
            # the openai implementation adds a discretized_guassian_step for t=0
            frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
            pred_mean, pred_variance, _ = self.p_model_mean_variance(
                frozen_out, x_noisy, t
            )
            kl = normal_kl(
                true_mean, true_log_variance, pred_mean, torch.log(pred_variance)
            )
            kl = mean_flat(kl) / np.log(2.0)
            decoder_nll = -discretized_gaussian_log_likelihood(
                x_start, means=pred_mean, log_scales=0.5 * torch.log(pred_variance)
            )
            decoder_nll = mean_flat(decoder_nll) / np.log(2.0)
            loss_term = torch.where((t == 0), decoder_nll, kl)
            # loss_term = torch.where(
            #     torch.isclose(t, torch.zeros_like(t), atol=0.5), decoder_nll, kl
            # )
            loss = loss + torch.mean(loss_term)

        # compute loss wrt target
        loss_type = self._loss_type
        if loss_type == "l1":
            loss_fn = partial(F.l1_loss, reduction="none")
        elif loss_type == "l2":
            loss_fn = partial(F.mse_loss, reduction="none")
        elif loss_type == "huber":
            loss_fn = partial(F.smooth_l1_loss, reduction="none")
        elif loss_type == "hybrid":
            loss_fn = partial(F.mse_loss, reduction="none")
        else:
            raise NotImplementedError()

        # Compute Noise Loss
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "previous_x":
            target = self.q_posterior_mean_variance(x_start, x_noisy, t)[0]
        elif self.prediction_type == "start_x":
            target = x_start
        else:
            raise NotImplementedError()

        pred_loss = loss_fn(model_output, target)
        if self.mask is not None:
            pred_loss = pred_loss * self.mask[None, None]

        # Optionally add in the xstart loss
        if loss_type == "hybrid":
            _, _, pred_start = self.p_model_mean_variance(
                model_output, x_noisy, t, clip_denoised=False
            )
            x0loss = F.l1_loss(pred_start, x_start, reduction="none")

            if self.mask is not None:
                x0loss = x0loss * self.mask[None, None]
            pred_loss = pred_loss + x0loss

        loss = loss + torch.mean(pred_loss * loss_weight[:, None, None, None])

        return loss

    @torch.no_grad()
    def p_sample(
        self,
        x,
        t,
        t_index,
        xcond_embd=None,
        ccond_embd=None,
        return_pred_start=False,
        clip=True,
    ):
        if self.mask is not None:
            x = x * self.mask[None, None]
        model_input = torch.cat((x, ccond_embd), dim=1) if self._use_ccond else x
        model_output = self.model(model_input, t, context=xcond_embd)

        # Classifier Free Guidance
        if self.use_cfg:
            xcond_embd = torch.zeros_like(xcond_embd) if self._use_xcond else None
            ccond_embd = torch.zeros_like(ccond_embd) if self._use_ccond else None
            model_input = torch.cat((x, ccond_embd), dim=1) if self._use_ccond else x
            model_output_uncond = self.model(model_input, t, context=xcond_embd)
            model_output = (
                1 - self.cfg_gamma
            ) * model_output_uncond + self.cfg_gamma * model_output

        pred_prev_sample, posterior_variance_t, pred_start = self.p_model_mean_variance(
            model_output, x, t, clip_denoised=clip
        )

        noise = torch.randn_like(x, dtype=self.use_dtype)
        nonzero_mask = (1 - (t == 0).float()).reshape(
            x.shape[0], *((1,) * (len(x.shape) - 1))
        )
        xtm1 = pred_prev_sample + nonzero_mask * posterior_variance_t**0.5 * noise

        # ### testing exact code match to sd v1 where they use log variance clipping
        # posterior_log_variance = extract(
        #     self.posterior_log_variance_clipped, t, x.shape
        # )

        if return_pred_start:
            return pred_start, xtm1
        else:
            return xtm1

    @torch.no_grad()
    def get_input(self, batch):
        """Gets inputs for Pixel Space Diffusion."""
        xinput = batch[self.input_key].to(dtype=self.use_dtype, device=self.device)

        if self._use_xcond:
            xcond = batch[self.xcond_key]
            assert isinstance(
                xcond, (float, np.ndarray, torch.Tensor)
            ), "Dtype Error Xcond."
            xcond = xcond.to(dtype=self.use_dtype, device=self.device)
        else:
            xcond = None

        if self._use_ccond:
            ccond = batch[self.ccond_key].to(dtype=self.use_dtype, device=self.device)
        else:
            ccond = None

        return (xinput, xcond, ccond)


class LatentDiffusion(DDPM):
    def __init__(
        self,
        first_stage_config,
        default_latent_scale=0.18,
        scale_from_first_batch=False,
        batch_encode=-1,
        embed_xcond=False,
        embed_ccond=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.instantiate_first_stage(first_stage_config)
        self.z_channels = first_stage_config.params.ddconfig["z_channels"]
        self.scale_from_first_batch = scale_from_first_batch
        self.batch_encode = batch_encode
        self.embed_xcond = embed_xcond
        self.embed_ccond = embed_ccond
        self.register_buffer("scale_factor", torch.tensor(default_latent_scale))
        self.register_buffer("cold_state", torch.tensor(0))

        return

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(
            config, ckpt_path=config["ckpt_path"], strict=False
        )
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

        return

    @torch.no_grad()
    def embd_in_latent(self, img):
        # B C H W
        # If the pixel space channel dimension is larger than 3 then we should rearrange
        bs, init_ch_dim = img.shape[0], img.shape[1]
        if np.mod(init_ch_dim, 3) != 0:
            # Calculate how many channels are needed to reach the next multiple of 3
            # Repeat the last channel 'needed_ch' times and concatenate it
            needed_ch = 3 - (init_ch_dim % 3)
            last_ch_repeated = img[:, -1:, :, :].repeat(1, needed_ch, 1, 1)
            img = torch.cat([img, last_ch_repeated], dim=1)

        ch_dim = img.shape[1]
        assert ch_dim % 3 == 0, "Channel dimension should be a multiple of 3"
        img = rearrange(img, "b (g c) h w -> (b g) c h w", b=bs, c=3, g=ch_dim // 3)

        # Too large batch dimension will crash memory so we can do it in loop
        batch_encode = min(self.batch_encode, img.shape[0])
        if batch_encode == -1:
            batch_encode = img.shape[0]
        n_batches = -(-img.size(0) // batch_encode)
        posterior_list = []
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_encode
            end_idx = min(start_idx + batch_encode, img.size(0))

            posterior = self.first_stage_model.encode(img[start_idx:end_idx])
            if not isinstance(posterior, torch.Tensor):
                posterior = posterior.sample().detach()
            posterior_list.append(posterior)

        posterior = torch.cat(posterior_list, dim=0)
        posterior = rearrange(
            posterior,
            "(b g) z h w -> b (g z) h w",
            g=ch_dim // 3,
            z=self.z_channels,
        )

        # If the model has not been run before, we can have the option to estimate a latent scale factor
        # from the first batch of the input data
        if self.scale_from_first_batch and self.cold_state == 0:
            self.cold_state.data = torch.tensor(1)
            self.scale_factor.data = 1 / posterior.std()
            print("using scale_factor: ", self.scale_factor)

        return posterior * self.scale_factor

    @torch.no_grad()
    def decode(self, z):
        # Similar to encode, we can batch the channel dimensions into groups
        # Too large batches will crash memory so we can also try to do it in a loop
        bs = z.shape[0]
        z = rearrange(z, "b (g z) h w -> (b g) z h w", b=bs, z=self.z_channels)

        batch_encode = min(self.batch_encode, z.shape[0])
        if batch_encode == -1:
            batch_encode = z.shape[0]
        n_batches = -(-z.size(0) // batch_encode)
        out = []
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_encode
            end_idx = min(start_idx + batch_encode, z.size(0))
            out.append(
                self.first_stage_model.decode(z[start_idx:end_idx] / self.scale_factor)
                .cpu()
                .numpy()
            )
        out = np.concatenate(out, axis=0)
        out = rearrange(out, "(b g) c h w -> b (g c) h w", c=3, b=bs)

        return out

    @torch.no_grad()
    def get_input(self, batch):
        xinput, xcond, ccond = super().get_input(batch)

        xinput = self.embd_in_latent(xinput)
        if self.embed_ccond and ccond is not None:
            ccond = self.embd_in_latent(ccond)
        if self.embed_xcond and xcond is not None:
            xcond = self.embd_in_latent(xcond)

        return (xinput, xcond, ccond)

    @torch.no_grad()
    def sample(
        self,
        latent_size,
        batch_size=16,
        variance_n=1,
        return_intermediate=False,
        x_start=None,
        xcond=None,
        ccond=None,
        return_latents=False,
    ):

        if ccond is not None and self.embed_ccond:
            ccond = self.embd_in_latent(
                ccond.to(dtype=self.use_dtype, device=self.device)
            )

        if xcond is not None and self.embed_xcond:
            xcond = self.embd_in_latent(
                xcond.to(dtype=self.use_dtype, device=self.device)
            )

        zt = super().sample(
            latent_size,
            batch_size,
            variance_n,
            return_intermediate,
            x_start,
            xcond,
            ccond,
        )
        zt = rearrange(zt, "t b v c h w -> t (b v) c h w", b=batch_size, v=variance_n)

        if return_intermediate:
            b = zt.shape[0]
            x_imgs = []
            for t in tqdm(
                range(b), desc="Decoding time sequences to pixel space", total=b
            ):
                x_imgs.append(self.decode(zt[t]))
            x_imgs = np.stack(x_imgs)
        else:
            print("Decoding the final timestep")
            zt = zt[-1]
            x_imgs = self.decode(zt)[None]

        x_imgs = rearrange(
            x_imgs, "t (b v) c h w -> t b v c h w", b=batch_size, v=variance_n
        )

        if return_latents:
            zt = rearrange(
                zt, "t (b v) c h w -> t b v c h w", b=batch_size, v=variance_n
            )
            return (x_imgs, zt)
        else:
            return x_imgs
