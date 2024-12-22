from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat
from torch.utils.checkpoint import checkpoint

from einops import rearrange
from tqdm import tqdm
import math
import pickle

from dflat.render import hsi_to_rgb, general_convolve
from diffvis.diffusion.ddpm_utils import extract


class TotalVariationLoss(nn.Module):
    def __init__(self):
        """
        Initializes the Total Variation Loss module.
        """
        super().__init__()

    def forward(self, input):
        """
        Computes the Total Variation Loss for a batch of images.

        Args:
            input (torch.Tensor): Input tensor of shape (b, c, h, w).

        Returns:
            torch.Tensor: The total variation loss (a scalar).
        """
        # Ensure input dimensions are valid
        if input.shape[2] < 2 or input.shape[3] < 2:
            raise ValueError("Height and width must be greater than 1 for TV loss.")

        # Compute differences between neighboring pixels
        diff_h = input[:, :, 1:, :] - input[:, :, :-1, :]  # Height differences
        diff_w = input[:, :, :, 1:] - input[:, :, :, :-1]  # Width differences

        # Compute the total variation loss
        loss_h = torch.abs(diff_h).mean()
        loss_w = torch.abs(diff_w).mean()

        # Total loss is the sum of height and width variations
        loss = loss_h + loss_w
        return loss


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        """
        Initializes the Huber Loss module.

        Args:
            delta (float): The threshold where the loss transitions from quadratic to linear.
        """
        super().__init__()
        self.delta = delta

    def forward(self, prediction, target):
        """
        Computes the Huber Loss.

        Args:
            prediction (torch.Tensor): The predicted values (shape: (C, H, W)).
            target (torch.Tensor): The ground truth values (shape: (C, H, W)).

        Returns:
            torch.Tensor: The Huber Loss.
        """
        error = prediction - target
        abs_error = torch.abs(error)

        quadratic = torch.where(
            abs_error <= self.delta,
            0.5 * error**2,
            self.delta * (abs_error - 0.5 * self.delta),
        )
        loss = quadratic.mean()  # Averaging over all pixels
        return loss


def generate_uncertainty_mask(uncertainty_matrix, percentage):
    percentage = max(0, min(100, percentage))
    total_pixels = uncertainty_matrix.size
    pixels_to_remove = int(total_pixels * percentage / 100)
    if pixels_to_remove == 0:
        flat_indices = []
    else:
        flat_indices = np.argsort(uncertainty_matrix.ravel())[-pixels_to_remove:]
    mask = np.ones_like(uncertainty_matrix, dtype=bool)
    np.put(mask, flat_indices, False)
    return mask.astype(np.float32)


def split_patches(image, patch_size, stride=None):
    # Expect input like [C, H, W] or [B, C, H, W]
    # Returns patched image of shape [G, C, h, w] or [G, B, C, h, w] depending on input dimensions
    # Where G is the number of patches
    img_shape = image.shape
    flag_tensor = torch.is_tensor(image)
    if not flag_tensor:
        image = torch.tensor(image, dtype=torch.float32)

    if stride is None:
        stride = patch_size

    # # zero-pad
    # hkH = (patch_size[0]-stride[0])//2
    # hkW = (patch_size[1]-stride[1])//2
    # rh = np.mod(img_shape[-2], stride[0])
    # rw = np.mod(img_shape[-1], stride[1])
    # image = F.pad(image, [hkW, rw + hkW, hkH, rh + hkH] + [0,0]*(len(img_shape)-2))

    # Unfold along height and width with the given patch size and stride
    patches_h = image.unfold(-2, patch_size[0], stride[0])
    patches_w = patches_h.unfold(-2, patch_size[1], stride[1])
    h, w = patches_w.shape[-4:-2]

    # Adjust the output format based on the dimensionality of the input
    if len(img_shape) == 3:  # Single image [C, H, W]
        patches = rearrange(patches_w, "c h w ph pw -> (h w) c ph pw")
    else:  # Batch of images [B, C, H, W]
        patches = rearrange(patches_w, "b c h w ph pw -> (h w) b c ph pw")

    if not flag_tensor:
        patches = patches.cpu().numpy()

    return patches, [h, w]


def combine_patches(patches, g, patch_size, stride=None):
    # B V C H W
    flag_tensor = torch.is_tensor(patches)
    if not flag_tensor:
        patches = torch.tensor(patches)

    patch_shape = patches.shape
    batch_mode = len(patch_shape) == 5
    if batch_mode:
        patches = rearrange(
            patches, "(g1 g2) b c ph pw -> g1 g2 b c ph pw", g1=g[0], g2=g[1]
        )
    else:
        patches = rearrange(
            patches, "(g1 g2) c ph pw -> g1 g2 c ph pw", g1=g[0], g2=g[1]
        )

    # Valid Crop
    if stride is None:
        stride = patch_size
    g1, g2 = patches.shape[0:2]
    ih, iw = g1 * stride[0], g2 * stride[1]
    dx = np.array(patch_size) - np.array(stride)
    ir = dx // 2
    er = -dx // 2
    er[0] = ih if er[0] == 0 else er[0]
    er[1] = iw if er[1] == 0 else er[1]

    inner_stitch = torch.cat([patches[i] for i in range(g1)], dim=-2)
    inner_stitch = torch.cat([inner_stitch[i] for i in range(g2)], dim=-1)
    return inner_stitch

    # # Inner Stitching
    # inner_stitch = patches[1:-1, 1:-1]
    # inner_stitch = inner_stitch[..., ir[0] : er[0], ir[1] : er[1]]
    # inner_stitch = torch.cat([inner_stitch[i] for i in range(g1 - 2)], dim=-2)
    # inner_stitch = torch.cat([inner_stitch[i] for i in range(g2 - 2)], dim=-1)

    # # outer stitching
    # tm = patches[0, 1:-1]
    # tm = tm[..., 0 : er[0], ir[1] : er[1]]
    # tm = torch.cat([tm[i] for i in range(g2 - 2)], dim=-1)
    # bm = patches[-1, 1:-1]
    # bm = bm[..., ir[0] :, ir[1] : er[1]]
    # bm = torch.cat([bm[i] for i in range(g2 - 2)], dim=-1)
    # unfolded = torch.cat([tm, inner_stitch, bm], axis=-2)

    # lt = patches[0, 0, :, :, 0 : er[0], 0 : er[1]]
    # lm = patches[1:-1, 0]
    # lm = lm[..., ir[0] : er[0], 0 : er[1]]
    # lm = torch.cat([lm[i] for i in range(g1 - 2)], dim=-2)
    # lb = patches[-1, 0, :, :, ir[0] :, 0 : er[1]]
    # left = torch.cat([lt, lm, lb], axis=-2)
    # unfolded = torch.cat([left, unfolded], axis=-1)

    # rt = patches[0, -1, :, :, 0 : er[0], ir[1] :]
    # rm = patches[1:-1, -1]
    # rm = rm[..., ir[0] : er[0], ir[1] :]
    # rm = torch.cat([rm[i] for i in range(g1 - 2)], dim=-2)
    # rb = patches[-1, -1, :, :, ir[0] :, ir[1] :]
    # right = torch.cat([rt, rm, rb], axis=-2)
    # unfolded = torch.cat([unfolded, right], axis=-1)

    # if not flag_tensor:
    #     unfolded = unfolded.cpu().numpy()

    # return unfolded


def generate_patch_masks(image_shape, patch_size, stride=None):
    """
    Generates binary masks for each patch in an image.

    :param image_shape: The shape of the image as a tuple (C, H, W).
    :param patch_size: A list or tuple with two elements [patch_height, patch_width].
    :returns: A tensor containing masks for each patch. Shape [G, C, H, W].
    """
    C, H, W = image_shape[-3:]
    if stride is None:
        stride = patch_size

    dx = np.array(patch_size) - np.array(stride)
    dh, dw = dx[0], dx[1]

    num_patches_h = (H - dh) // stride[0]
    num_patches_w = (W - dw) // stride[1]
    num_patches = num_patches_w * num_patches_h
    masks = torch.zeros((num_patches, 1, H, W), dtype=torch.float32)

    stride_height, stride_width = stride
    patch_idx = 0
    for i in range(num_patches_h):
        for j in range(num_patches_w):

            if i == 0:
                y1, y2 = 0, dh // 2 + stride_height
            elif i == num_patches_h - 1:
                y1, y2 = dh // 2 + stride_height * i, H
            else:
                y1, y2 = dh // 2 + i * stride_height, dh // 2 + (i + 1) * stride_height

            if j == 0:
                x1, x2 = 0, dw // 2 + stride_width
            elif j == num_patches_w - 1:
                x1, x2 = dw // 2 + stride_width * j, W
            else:
                x1, x2 = dw // 2 + j * stride_width, dw // 2 + (j + 1) * stride_width

            masks[patch_idx, :, y1:y2, x1:x2] = 1
            patch_idx += 1

    return masks


def deshift(meas, step=2, nc=28):
    istensor = True
    if not torch.is_tensor(meas):
        meas = torch.tensor(meas, dtype=torch.float32)
        istensor = False

    ch, h, w = meas.shape
    assert ch == 1, "grayscale measurement assertion."
    output = torch.cat(
        [meas[:, :, step * i : step * i + w - (nc - 1) * step] for i in range(nc)],
        dim=0,
    )
    if not istensor:
        output = output.numpy()

    return output


def shift(inputs, step=2):
    if not torch.is_tensor(inputs):
        inputs = torch.tensor(inputs, dtype=torch.float32)

    nv, nC, row, col = inputs.shape[-4:]
    output = torch.zeros(
        nv, nC, row, col + (nC - 1) * step, dtype=inputs.dtype, device=inputs.device
    )
    for i in range(nC):
        output[:, i, :, step * i : step * i + col] = inputs[:, i]
    return output


class DDIM_Sampler(nn.Module):
    def __init__(
        self,
        diffusion_model,
        n_steps,
        ddim_scheme="uniform",
        ddim_eta=0,
    ):
        super().__init__()
        self.diffusion_model = diffusion_model.to("cuda")
        self.prediction_type = diffusion_model.prediction_type
        timesteps = self.diffusion_model.timesteps

        if ddim_scheme == "uniform":
            c = timesteps // n_steps
            # ddim_times = np.asarray(list(range(0, timesteps, c)))
            ddim_times = list(range(0, timesteps, c))
            if ddim_times[-1] != 999:
                ddim_times.append(999)
            ddim_times = np.array(ddim_times)
            # The original DDIM paper does not enforce starting at terminal T
            # but later works have shown that this is flawed so I modify it
        else:
            raise ValueError("Unknown ddim scheme input")

        self._initialize_schedule(
            self.diffusion_model.alphas_cumprod, ddim_times, ddim_eta
        )
        self.n_steps = len(ddim_times)
        self.ddim_times = ddim_times

    def _initialize_schedule(self, orig_alphas_cumprod, ddim_times, ddim_eta):
        alphas_cumprod = orig_alphas_cumprod[ddim_times].clone()
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        ddim_sigma = (
            ddim_eta
            * torch.sqrt((1 - alphas_cumprod_prev) / (1 - alphas_cumprod))
            * torch.sqrt(1 - alphas_cumprod / alphas_cumprod_prev)
        )

        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.ddim_sigma = ddim_sigma
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.ddim_coeff = torch.sqrt(1 - alphas_cumprod_prev - ddim_sigma**2)
        return

    def p_sample(self, x_t, t, ti, ccond, clip=True):
        ### Collect coefficients
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, ti, x_t.shape
        )
        sqrt_recip_alphas_cumprod_t = extract(
            self.sqrt_recip_alphas_cumprod, ti, x_t.shape
        )
        sqrt_alpha_cumprod_prev_t = torch.sqrt(
            extract(self.alphas_cumprod_prev, ti, x_t.shape)
        )
        ddim_coeff_t = extract(self.ddim_coeff, ti, x_t.shape)
        ddim_sigma_t = extract(self.ddim_sigma, ti, x_t.shape)

        ### Get noise/score
        with torch.no_grad():
            model_output = self.diffusion_model.model(
                torch.cat((x_t, ccond), dim=1), t, context=None
            )

        if self.prediction_type == "epsilon":
            epsilon = model_output
        elif self.prediction_type == "start_x":
            sqrt_alphas_cumprod_t = torch.sqrt(
                extract(self.alphas_cumprod, ti, x_t.shape)
            )
            epsilon = (
                x_t - model_output * sqrt_alphas_cumprod_t
            ) / sqrt_one_minus_alphas_cumprod_t
        else:
            raise ValueError("Unsupported model prediction type.")

        ### Get x0hat
        pred_xstart = (
            x_t - sqrt_one_minus_alphas_cumprod_t * epsilon
        ) * sqrt_recip_alphas_cumprod_t
        if clip:
            pred_xstart = torch.clamp(pred_xstart, -1.0, 1.0)

        ### Compute DDIM Step
        nonzero_mask = (1 - (t == 0).float()).reshape(
            x_t.shape[0], *((1,) * (len(x_t.shape) - 1))
        )
        noise = torch.randn_like(x_t)

        xtm1 = (
            sqrt_alpha_cumprod_prev_t * pred_xstart
            + ddim_coeff_t * epsilon
            + nonzero_mask * ddim_sigma_t * noise
        )

        return xtm1, pred_xstart


class DIFFSSI_CASSI(DDIM_Sampler):
    def __init__(
        self,
        diffusion_model,
        n_steps=50,
        ddim_scheme="uniform",
        ddim_eta=1.0,
        gtstart=999,
        gtend=0,
        gloop=10,
        gscale=1,
        mask=None,
        mask_path=None,
        mask_key=None,
        patch_normalize=True,
        patch_scale_guidance=True,
        orig_clamp=True,
        guidance_loss="L2",
        mask_product=False,
    ):
        super().__init__(diffusion_model, n_steps, ddim_scheme, ddim_eta)
        if mask_path is not None:
            assert mask is None, "If mask_path is provided, do not pass in a mask"
            assert (
                mask_key is not None
            ), "dictionary key name corresponding to mask must be provided."
            mask = loadmat(mask_path)[mask_key][None]
            mask = mask / mask.max()
        if mask is not None:
            assert len(mask.shape) == 3, "Expected kernel of shape [C, H, W]"
            self.mask = torch.tensor(mask, dtype=torch.float32, device="cuda")
        else:
            self.mask = None
            print("No masked is passed in. Running in maskless mode")

        self.gloop = gloop
        self.gscale = gscale
        self.gtstart = gtstart
        self.gtend = gtend
        self.variance_n = None
        self.batch_size = None
        self.patch_normalize = patch_normalize
        self.patch_scale_guidance = patch_scale_guidance
        self.orig_clamp = True
        self.guidance_loss = guidance_loss
        self.mask_product = mask_product

    def sample(
        self,
        measurement,
        nc=28,
        variance_n=1,
        split_n=None,
        return_intermediate=False,
        x_start=None,
        patch_size=[64, 64],
        stride_size=None,
        use_guidance=True,
        xcond=None,
        rescale_return=True,
        diffusion_batching=None,
    ):

        split_n = variance_n if split_n is None else int(split_n)
        assert (
            len(measurement.shape) == 3
        ), "measurement should be a rank 3 tensor [c h w] image."
        assert len(patch_size) == 2, "patchsize should be [h, w]"

        # convert measurement to shif-back condition
        measurement = torch.tensor(measurement, dtype=torch.float32, device="cuda")
        self.measurement = measurement

        ccond = deshift(measurement, nc=nc)
        if self.mask_product:
            print("Computing with Mask Product Condition")
            ccond = ccond * self.mask

        h, w = ccond.shape[-2:]
        ch = self.diffusion_model._seed_channels
        sh, sw = patch_size
        assert (
            h % sh == 0 and w % sw == 0
        ), "Height and width must be perfectly divisible by sh and sw"

        ccond = ccond.to(dtype=torch.float32, device="cuda")
        ccond, ccond_pn, g = self._forward_img2patch(ccond, patch_size, stride_size)
        batch_size = ccond.shape[0]

        # mask, _ = split_patches(self.mask, patch_size, stride_size)
        # ccond = torch.cat((ccond, mask), axis=1)
        # ccond_pn = torch.cat((ccond_pn, mask), axis=1)

        if x_start is None:
            x_start = torch.randn(
                (
                    variance_n * batch_size,
                    ch,
                    *patch_size,
                ),
                dtype=torch.float32,
                device="cuda",
            )
        else:
            x_start = x_start.to(dtype=torch.float32, device="cuda")

        # ccond = self.diffusion_model.ccond_stage_model(ccond)
        ccond = self.diffusion_model.reshape_batched_variance(ccond, variance_n)
        ccond_pn = self.diffusion_model.reshape_batched_variance(ccond_pn, variance_n)

        # if xcond is not None:
        #     xcond = xcond.to(dtype=torch.float32, device="cuda")[None]
        #     xcond = (xcond * 2) - 1
        #     xcond = self.diffusion_model.xcond_stage_model(xcond).detach()
        #     xcond = torch.tile(xcond, [ccond.shape[0], 1])

        self.batch_size = batch_size
        self.g = g
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.variance_n = split_n

        # Allow splitting of variance sampled draws into subgroups
        rep_calc = variance_n // split_n
        total_steps = self.n_steps if return_intermediate else 1
        imgs = torch.zeros((total_steps, variance_n, ch, h, w), device="cpu")
        x0s = torch.zeros((total_steps, variance_n, ch, h, w), device="cpu")
        self.hold_losses = []
        for repi in range(rep_calc):
            ilow = repi * split_n * batch_size
            ihigh = (repi + 1) * split_n * batch_size

            if diffusion_batching is None:
                self.diffusion_batching = int(ihigh - ilow)
            else:
                self.diffusion_batching = int(diffusion_batching)
            im = x_start[ilow:ihigh]
            use_ccond = ccond[ilow:ihigh]
            use_ccond_pn = ccond_pn[ilow:ihigh]
            use_xcond = xcond[ilow:ihigh] if xcond is not None else None
            progress_bar = tqdm(
                zip(reversed(range(self.n_steps)), reversed(self.ddim_times)),
                total=self.n_steps,
                desc="",
            )
            step_idx = 0

            for idx, i in progress_bar:
                im, x0 = self.p_sample(
                    im,
                    torch.full((im.shape[0],), i, dtype=torch.long, device="cuda"),
                    torch.full((im.shape[0],), idx, dtype=torch.long, device="cuda"),
                    use_ccond,
                    use_ccond_pn,
                    use_xcond,
                    use_guidance,
                )

                if (i == 0) or return_intermediate:
                    im_ = im.clone().detach()
                    x0_ = x0.clone().detach()

                    ## Apply LSQ patch scaling
                    if rescale_return:
                        _, x0_, cps = self._guidance_step(x0_)
                        _, im_ = self._guidance_step(im_)

                    im_ = self._reverse_patch2img(im_, split_n, batch_size)
                    x0_ = self._reverse_patch2img(x0_, split_n, batch_size)

                    imgs[step_idx, split_n * repi : split_n * (repi + 1)] = im_.cpu()
                    x0s[step_idx, split_n * repi : split_n * (repi + 1)] = x0_.cpu()
                    step_idx += 1

                progress_bar.set_description(f"Index {idx}, Time: {i}")

        np_hold_losses = np.stack(self.hold_losses)
        np_hold_losses = rearrange(
            np_hold_losses, "(g t) s -> t (g s)", g=variance_n // split_n, s=split_n
        )
        return imgs.numpy(), x0s.numpy(), np_hold_losses

    def p_sample(self, x_t, t, ti, ccond, ccond_pn, xcond, use_guidance):
        ### Collect coefficients
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, ti, x_t.shape
        )
        sqrt_recip_alphas_cumprod_t = extract(
            self.sqrt_recip_alphas_cumprod, ti, x_t.shape
        )
        sqrt_alpha_cumprod_prev_t = torch.sqrt(
            extract(self.alphas_cumprod_prev, ti, x_t.shape)
        )
        ddim_coeff_t = extract(self.ddim_coeff, ti, x_t.shape)
        ddim_sigma_t = extract(self.ddim_sigma, ti, x_t.shape)

        ### Get noise/score
        with torch.no_grad():
            ccond_ = ccond_pn if self.patch_normalize else ccond
            model_output = self.batched_model_inference(x_t, ccond_, t, xcond)

        if self.prediction_type == "epsilon":
            epsilon = model_output
        elif self.prediction_type == "start_x":
            sqrt_alphas_cumprod_t = torch.sqrt(
                extract(self.alphas_cumprod, ti, x_t.shape)
            )
            epsilon = (
                x_t - model_output * sqrt_alphas_cumprod_t
            ) / sqrt_one_minus_alphas_cumprod_t
        else:
            raise ValueError("Unsupported model prediction type.")

        xtp = x_t.clone()
        time = t.flatten()[0]
        if use_guidance and time <= self.gtstart and time >= self.gtend:
            use_gloop = self.gloop
            xtp.requires_grad_(True)
            # optimizer = optim.Adam([xtp], lr=self.gscale)

            for _ in range(use_gloop):
                if xtp.grad is not None:
                    xtp.grad.zero_()
                # optimizer.zero_grad()

                ccond_ = ccond_pn if self.patch_normalize else ccond
                model_output = self.batched_model_inference(xtp, ccond_, t, xcond)

                if self.prediction_type == "epsilon":
                    x0 = (
                        xtp - sqrt_one_minus_alphas_cumprod_t * model_output
                    ) * sqrt_recip_alphas_cumprod_t
                elif self.prediction_type == "start_x":
                    x0 = model_output

                x0 = torch.clamp(x0, -1, 1)
                losses, _ = self._guidance_step(x0)
                loss = torch.sum(losses)
                loss.backward()
                # optimizer.step()

                with torch.no_grad():
                    xtp -= self.gscale * xtp.grad / torch.norm(xtp.grad)
                    xtp = xtp.detach().requires_grad_(True)

            xtp = xtp.detach()

        ### Get x0hat and Compute DDIM Step
        pred_xstart = (
            xtp - sqrt_one_minus_alphas_cumprod_t * epsilon
        ) * sqrt_recip_alphas_cumprod_t
        pred_xstart = torch.clamp(pred_xstart, -1, 1)

        nonzero_mask = (1 - (t == 0).float()).reshape(
            x_t.shape[0], *((1,) * (len(x_t.shape) - 1))
        )
        noise = torch.randn_like(x_t)
        xtm1 = (
            sqrt_alpha_cumprod_prev_t * pred_xstart
            + ddim_coeff_t * epsilon
            + nonzero_mask * ddim_sigma_t * noise
        )

        # Save the projection loss for reference
        losses, _ = self._guidance_step(pred_xstart)
        self.hold_losses.append(losses[:, 0].detach().cpu().numpy())

        return xtm1, pred_xstart

    def batched_model_inference(self, x_t, ccond, t, xcond):
        b = self.diffusion_batching
        n = x_t.shape[0]
        outputs = []

        for i in range(0, n, b):
            end = min(i + b, n)
            batch_x_t = x_t[i:end]
            batch_ccond = ccond[i:end]
            batch_t = t[i:end]
            batch_xcond = xcond[i:end] if xcond is not None else None

            batch_input = torch.cat((batch_x_t, batch_ccond), dim=1)
            batch_output = self.diffusion_model.model(
                batch_input, batch_t, context=batch_xcond
            )

            outputs.append(batch_output)

        return torch.cat(outputs, dim=0)

    def _forward_img2patch(self, ccond, patch_size, stride_size):
        ccond = ccond / torch.amax(ccond)
        ccond, g = split_patches(ccond, patch_size, stride_size)

        ccond_pn = ccond / torch.amax(ccond, axis=(-1, -2, -3), keepdim=True)
        ccond_pn = (ccond_pn * 2) - 1
        ccond = (ccond * 2) - 1

        return ccond, ccond_pn, g

    def _guidance_step(self, x0, return_scales=False):
        measurement = self.measurement
        x0 = rearrange(
            x0, "(v b) c h w -> b v c h w", v=self.variance_n, b=self.batch_size
        )
        x0 = torch.clamp(x0, -1, 1)
        # x0 = torch.clamp(x0, -1, 1e3)
        x0 = (x0 + 1) / 2
        x0_resh = combine_patches(x0, self.g, self.patch_size, self.stride_size)

        patch_masks = generate_patch_masks(
            x0_resh.shape, self.patch_size, self.stride_size
        )

        A = torch.zeros(
            (self.batch_size, self.variance_n, *measurement.shape[-3:]),
            dtype=torch.float32,
            device="cuda",
        )
        for i in range(patch_masks.shape[0]):
            pm = patch_masks[i].to(dtype=torch.float32, device="cuda")
            Ai = shift(x0_resh * pm[None] * self.mask[None])
            Ai = torch.sum(Ai, axis=-3, keepdim=True)
            A[i] = Ai

        cps = []
        for vi in range(A.shape[1]):
            Ai = A[:, vi].view(self.batch_size, -1).T
            cp = torch.pinverse(Ai.T @ Ai) @ Ai.T @ measurement.flatten()
            cps.append(cp)

        cps = torch.stack(cps).T
        x0 = x0 * cps[:, :, None, None, None]
        x0_resc = combine_patches(x0, self.g, self.patch_size, self.stride_size)

        pmeas = shift(x0_resc * self.mask[None])
        pmeas = torch.sum(pmeas, axis=-3, keepdim=True)

        if self.guidance_loss == "L2":
            losses = torch.sum((measurement[None] - pmeas) ** 2, axis=(-1, -2, -3))
        elif self.guidance_loss == "L1":
            losses = torch.sum(torch.abs(measurement[None] - pmeas), axis=(-1, -2, -3))
        else:
            raise ValueError("unknown guidance loss ")

        if return_scales:
            return losses[:, None], x0, cps

        return losses[:, None], x0

    def _reverse_patch2img(self, hsi, v, b):
        # hsi = rearrange(hsi, "(v b) c h w -> b v c h w", v=v, b=b)
        hsi = combine_patches(hsi, self.g, self.patch_size, self.stride_size)
        # if self.orig_clamp:
        #     hsi = torch.clamp(hsi, -1, 1)
        # else:
        #     hsi = torch.clamp(hsi, -1, 1e3)

        # hsi = (hsi + 1) / 2
        # hsi = hsi / torch.amax(hsi, axis=(-1, -2, -3), keepdim=True)
        return hsi


class DIFFSSI_CASSI_Large(DDIM_Sampler):
    def __init__(
        self,
        diffusion_model,
        n_steps=50,
        ddim_scheme="uniform",
        ddim_eta=1.0,
        gtstart=999,
        gtend=0,
        gloop=10,
        gscale=1,
        mask=None,
        mask_path=None,
        mask_key=None,
        patch_normalize=True,
        patch_scale_guidance=True,
        guidance_loss="L2",
        mask_product=True,
    ):
        super().__init__(diffusion_model, n_steps, ddim_scheme, ddim_eta)
        if mask_path is not None:
            assert mask is None, "If mask_path is provided, do not pass in a mask"
            assert (
                mask_key is not None
            ), "dictionary key name corresponding to mask must be provided."
            mask = loadmat(mask_path)[mask_key][None]
            mask = mask / mask.max()
        if mask is not None:
            assert len(mask.shape) == 3, "Expected kernel of shape [C, H, W]"
            self.mask = torch.tensor(mask, dtype=torch.float32, device="cuda")
        else:
            self.mask = None
            print("No masked is passed in. Running in maskless mode")

        self.gloop = gloop
        self.gscale = gscale
        self.gtstart = gtstart
        self.gtend = gtend
        self.variance_n = None
        self.batch_size = None
        self.patch_normalize = patch_normalize
        self.patch_scale_guidance = patch_scale_guidance
        self.guidance_loss = guidance_loss
        self.mask_product = mask_product

    def sample(
        self,
        measurement,
        nc=28,
        variance_n=1,
        split_n=None,
        return_intermediate=False,
        x_start=None,
        patch_size=[64, 64],
        stride_size=None,
        use_guidance=True,
        xcond=None,
        rescale_return=True,
        diffusion_batching=16,
    ):
        split_n = variance_n if split_n is None else int(split_n)
        assert (
            len(measurement.shape) == 3
        ), "measurement should be a rank 3 tensor [c h w] image."
        assert len(patch_size) == 2, "patchsize should be [h, w]"

        # convert measurement to shif-back condition
        measurement = torch.tensor(measurement, dtype=torch.float32, device="cuda")
        self.measurement = measurement

        ccond = deshift(measurement, nc=nc)
        if self.mask_product:
            print("Computing with Mask Product Condition")
            ccond = ccond * self.mask

        h, w = ccond.shape[-2:]
        ch = self.diffusion_model._seed_channels
        sh, sw = patch_size
        assert (
            h % sh == 0 and w % sw == 0
        ), "Height and width must be perfectly divisible by sh and sw"

        ccond = ccond.to(dtype=torch.float32, device="cuda")
        ccond, ccond_pn, g = self._forward_img2patch(ccond, patch_size, stride_size)
        batch_size = ccond.shape[0]

        if x_start is None:
            x_start = torch.randn(
                (
                    variance_n * batch_size,
                    ch,
                    *patch_size,
                ),
                dtype=torch.float32,
                device="cuda",
            )
        else:
            x_start = x_start.to(dtype=torch.float32, device="cuda")

        ccond = self.diffusion_model.reshape_batched_variance(ccond, variance_n)
        ccond_pn = self.diffusion_model.reshape_batched_variance(ccond_pn, variance_n)

        # if xcond is not None:
        #     xcond = xcond.to(dtype=torch.float32, device="cuda")[None]
        #     xcond = (xcond * 2) - 1
        #     xcond = self.diffusion_model.xcond_stage_model(xcond).detach()
        #     xcond = torch.tile(xcond, [ccond.shape[0], 1])

        self.batch_size = batch_size
        self.g = g
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.variance_n = split_n

        # Allow splitting of variance sampled draws into subgroups
        rep_calc = variance_n // split_n
        total_steps = self.n_steps if return_intermediate else 1
        imgs = torch.zeros((total_steps, variance_n, ch, h, w), device="cpu")
        x0s = torch.zeros((total_steps, variance_n, ch, h, w), device="cpu")
        self.hold_losses = []
        for repi in range(rep_calc):
            ilow = repi * split_n * batch_size
            ihigh = (repi + 1) * split_n * batch_size

            if diffusion_batching is None:
                self.diffusion_batching = int(ihigh - ilow)
            else:
                self.diffusion_batching = int(diffusion_batching)
            im = x_start[ilow:ihigh]
            use_ccond = ccond[ilow:ihigh]
            use_ccond_pn = ccond_pn[ilow:ihigh]
            use_xcond = xcond[ilow:ihigh] if xcond is not None else None
            progress_bar = tqdm(
                zip(reversed(range(self.n_steps)), reversed(self.ddim_times)),
                total=self.n_steps,
                desc="",
            )
            step_idx = 0

            for idx, i in progress_bar:
                im, x0 = self.p_sample(
                    im,
                    torch.full((im.shape[0],), i, dtype=torch.long, device="cuda"),
                    torch.full((im.shape[0],), idx, dtype=torch.long, device="cuda"),
                    use_ccond,
                    use_ccond_pn,
                    use_xcond,
                    use_guidance,
                )

                if (i == 0) or return_intermediate:
                    im_ = im.clone().detach()
                    x0_ = x0.clone().detach()

                    ## Apply LSQ patch scaling
                    if rescale_return:
                        _, x0_ = self._guidance_step(x0_)
                        _, im_ = self._guidance_step(im_)

                    im_ = self._reverse_patch2img(im_, split_n, batch_size)
                    x0_ = self._reverse_patch2img(x0_, split_n, batch_size)

                    imgs[step_idx, split_n * repi : split_n * (repi + 1)] = im_.cpu()
                    x0s[step_idx, split_n * repi : split_n * (repi + 1)] = x0_.cpu()
                    step_idx += 1

                progress_bar.set_description(f"Index {idx}, Time: {i}")

        np_hold_losses = np.stack(self.hold_losses)
        np_hold_losses = rearrange(
            np_hold_losses, "(g t) s -> t (g s)", g=variance_n // split_n, s=split_n
        )
        return imgs.numpy(), x0s.numpy(), np_hold_losses

    def p_sample(self, x_t, t, ti, ccond, ccond_pn, xcond, use_guidance):
        ### Collect coefficients
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, ti, x_t.shape
        )
        sqrt_recip_alphas_cumprod_t = extract(
            self.sqrt_recip_alphas_cumprod, ti, x_t.shape
        )
        sqrt_alpha_cumprod_prev_t = torch.sqrt(
            extract(self.alphas_cumprod_prev, ti, x_t.shape)
        )
        ddim_coeff_t = extract(self.ddim_coeff, ti, x_t.shape)
        ddim_sigma_t = extract(self.ddim_sigma, ti, x_t.shape)

        ### Get noise/score
        with torch.no_grad():
            ccond_ = ccond_pn if self.patch_normalize else ccond
            model_output = self.batched_model_inference(x_t, ccond_, t, xcond)

        if self.prediction_type == "epsilon":
            epsilon = model_output
        elif self.prediction_type == "start_x":
            sqrt_alphas_cumprod_t = torch.sqrt(
                extract(self.alphas_cumprod, ti, x_t.shape)
            )
            epsilon = (
                x_t - model_output * sqrt_alphas_cumprod_t
            ) / sqrt_one_minus_alphas_cumprod_t
        else:
            raise ValueError("Unsupported model prediction type.")

        xtp = x_t.clone()
        time = t.flatten()[0]
        if use_guidance and time <= self.gtstart and time >= self.gtend:
            use_gloop = self.gloop
            xtp.requires_grad_(True)
            for _ in range(use_gloop):
                if xtp.grad is not None:
                    xtp.grad.zero_()
                ccond_ = ccond_pn if self.patch_normalize else ccond
                model_output = checkpoint(
                    self.batched_model_inference, xtp, ccond_, t, xcond
                )
                # model_output = self.batched_model_inference(xtp, ccond_, t, xcond)

                if self.prediction_type == "epsilon":
                    x0 = (
                        xtp - sqrt_one_minus_alphas_cumprod_t * model_output
                    ) * sqrt_recip_alphas_cumprod_t
                elif self.prediction_type == "start_x":
                    x0 = model_output

                x0 = torch.clamp(x0, -1, 1)
                losses, _ = checkpoint(self._guidance_step, x0)
                # losses, _ = self._guidance_step(x0)
                loss = torch.sum(losses)
                loss.backward()

                with torch.no_grad():
                    xtp -= self.gscale * xtp.grad / torch.norm(xtp.grad)
                    xtp = xtp.detach().requires_grad_(True)

            xtp = xtp.detach()

        ### Get x0hat and Compute DDIM Step
        pred_xstart = (
            xtp - sqrt_one_minus_alphas_cumprod_t * epsilon
        ) * sqrt_recip_alphas_cumprod_t
        pred_xstart = torch.clamp(pred_xstart, -1, 1)

        nonzero_mask = (1 - (t == 0).float()).reshape(
            x_t.shape[0], *((1,) * (len(x_t.shape) - 1))
        )
        noise = torch.randn_like(x_t)
        xtm1 = (
            sqrt_alpha_cumprod_prev_t * pred_xstart
            + ddim_coeff_t * epsilon
            + nonzero_mask * ddim_sigma_t * noise
        )

        # Save the projection loss for reference
        losses, _ = self._guidance_step(pred_xstart)
        self.hold_losses.append(losses[:, 0].detach().cpu().numpy())

        return xtm1, pred_xstart

    def batched_model_inference(self, x_t, ccond, t, xcond):
        b = self.diffusion_batching
        n = x_t.shape[0]
        outputs = []

        for i in range(0, n, b):
            end = min(i + b, n)
            batch_x_t = x_t[i:end]
            batch_ccond = ccond[i:end]
            batch_t = t[i:end]
            batch_xcond = xcond[i:end] if xcond is not None else None

            batch_input = torch.cat((batch_x_t, batch_ccond), dim=1)
            batch_output = self.diffusion_model.model(
                batch_input, batch_t, context=batch_xcond
            )

            outputs.append(batch_output)

        return torch.cat(outputs, dim=0)

    def _forward_img2patch(self, ccond, patch_size, stride_size):
        ccond = ccond / torch.amax(ccond)
        ccond, g = split_patches(ccond, patch_size, stride_size)

        ccond_pn = ccond / torch.amax(ccond, axis=(-1, -2, -3), keepdim=True)
        ccond_pn = (ccond_pn * 2) - 1
        ccond = (ccond * 2) - 1

        return ccond, ccond_pn, g

    def _guidance_step(self, x0, return_scales=False):
        measurement = self.measurement
        x0 = rearrange(
            x0, "(v b) c h w -> b v c h w", v=self.variance_n, b=self.batch_size
        )
        x0 = torch.clamp(x0, -1, 1)
        x0 = (x0 + 1) / 2

        # Patch Rescale
        x0 = self._patch_normalize(x0)

        losses = checkpoint(self.measurement_guidance, x0)
        return losses, x0

    def measurement_guidance(self, x0):
        measurement = self.measurement
        x0_resh = combine_patches(x0, self.g, self.patch_size, self.stride_size)
        pmeas = shift(x0_resh * self.mask[None])
        pmeas = torch.sum(pmeas, axis=-3, keepdim=True)
        if self.guidance_loss == "L2":
            losses = torch.sum((measurement[None] - pmeas) ** 2, axis=(-1, -2, -3))
        elif self.guidance_loss == "L1":
            losses = torch.sum(torch.abs(measurement[None] - pmeas), axis=(-1, -2, -3))
        else:
            raise ValueError("unknown guidance loss ")

        return losses[:, None]

    def _patch_normalize(self, x0, chunk_mult=4):
        g = self.g
        variance_n = self.variance_n
        patch_size = self.patch_size
        measurement = self.measurement
        out = rearrange(x0, "(g0 g1) v c h w -> g0 g1 v c h w", g0=g[0], g1=g[1])
        with torch.no_grad():
            split_mask, _ = split_patches(self.mask, self.patch_size, self.stride_size)
            split_mask = rearrange(
                split_mask, "(g0 g1) c h w -> g0 g1 c h w", g0=g[0], g1=g[1]
            )
            rc = out.shape[0] // chunk_mult
            cc = out.shape[1] // chunk_mult
            hold_cps = torch.zeros(
                (g[0], g[1], variance_n, 1, 1, 1), dtype=torch.float32, device="cuda"
            )
            for rci in range(rc):
                for cci in range(cc):
                    xoi = out[
                        rci * chunk_mult : (rci + 1) * chunk_mult,
                        cci * chunk_mult : (cci + 1) * chunk_mult,
                    ]
                    xoi = rearrange(
                        xoi,
                        "g0 g1 v c h w -> (g0 g1) v c h w",
                        g0=chunk_mult,
                        g1=chunk_mult,
                    )
                    ki = split_mask[
                        rci * chunk_mult : (rci + 1) * chunk_mult,
                        cci * chunk_mult : (cci + 1) * chunk_mult,
                    ]
                    ki = rearrange(
                        ki, "g0 g1 c h w -> (g0 g1) c h w", g0=chunk_mult, g1=chunk_mult
                    )
                    mi = measurement[
                        0,
                        rci
                        * chunk_mult
                        * patch_size[0] : (rci + 1)
                        * chunk_mult
                        * patch_size[0],
                        cci
                        * chunk_mult
                        * patch_size[1] : (cci + 1)
                        * chunk_mult
                        * patch_size[1],
                    ]

                    cps = self.block_lsq_patch_norm(
                        xoi, mi, ki, chunk_mult, masked=True
                    )
                    cps = rearrange(
                        cps,
                        "(g0 g1) v c h w -> g0 g1 v c h w",
                        g0=chunk_mult,
                        g1=chunk_mult,
                    )
                    hold_cps[
                        rci * chunk_mult : (rci + 1) * chunk_mult,
                        cci * chunk_mult : (cci + 1) * chunk_mult,
                    ] = cps

        out = out * hold_cps
        out = rearrange(out, " g0 g1 v c h w -> (g0 g1) v c h w", g0=g[0], g1=g[1])
        return out

    def _reverse_patch2img(self, hsi, v, b):
        # hsi = rearrange(hsi, "(v b) c h w -> b v c h w", v=v, b=b)
        hsi = combine_patches(hsi, self.g, self.patch_size, self.stride_size)
        # if self.orig_clamp:
        #     hsi = torch.clamp(hsi, -1, 1)
        # else:
        #     hsi = torch.clamp(hsi, -1, 1e3)

        # hsi = (hsi + 1) / 2
        # hsi = hsi / torch.amax(hsi, axis=(-1, -2, -3), keepdim=True)
        return hsi

    def block_lsq_patch_norm(self, xoi, mi, ki, gc, masked=True):
        py, px = self.patch_size[-2:]
        ys, xs = 0, 2 * 27
        fh = mi.shape[-2] + ys
        fw = mi.shape[-1] + xs  # sheer_step * wl
        A = torch.zeros(
            size=(*xoi.shape[0:2], 1, fh, fw), dtype=torch.float32, device="cuda"
        )
        for pi in range(xoi.shape[0]):
            # torch.Size([4, 2, 28, 64, 64]) torch.Size([4, 1, 1, 64, 64])
            Ai = shift(xoi[pi] * ki[None, pi])
            Ai = torch.sum(Ai, axis=-3, keepdim=True)
            ri, ci = divmod(pi, gc)
            A[
                pi,
                :,
                :,
                ri * py : (ri * py + py + ys),
                ci * px : (ci * px + px + xs),
            ] = Ai
        A = A[:, :, :, :, :-xs]

        # Apply mask to measurement chunk and A
        if masked:
            H, W = mi.shape[-2:]
            subcrop_mask = torch.zeros((1, H, W), dtype=torch.float32, device="cuda")
            subcrop_mask[
                :,
                :,
                xs:W,
            ] = 1.0
            A = A * subcrop_mask[None, None]
            mi = mi * subcrop_mask

            cps = []
            for vi in range(A.shape[1]):
                Ai = A[:, vi].reshape(xoi.shape[0], -1).T
                cp = torch.inverse(Ai.T @ Ai) @ Ai.T @ mi.flatten()
                cps.append(cp)
            cps = torch.stack(cps).T
        return cps[:, :, None, None, None].to(torch.float32)


class DiffSSIV2(DDIM_Sampler):
    def __init__(
        self,
        diffusion_model,
        n_steps,
        ddim_scheme="uniform",
        ddim_eta=1.0,
        gtstart=999,
        gtend=0,
        gloop=10,
        gscale=1,
        kernel=None,
        kernel_path=None,
        kernel_key=None,
        mode="gs",
        patch_normalize=True,
        patch_scale_guidance=True,
        sub_crop=False,
        lam=None,
        guidance_loss="L2",
    ):
        super().__init__(diffusion_model, n_steps, ddim_scheme, ddim_eta)
        assert mode in ["rgb", "gs"], "mode must be 'rgb' or 'gs'."
        if kernel_path is not None:
            assert kernel is None, "If kernel_path is provided, do not pass in a kernel"
            assert (
                kernel_key is not None
            ), "dictionary key name corresponding to kernel must be provided."
            with open(kernel_path, "rb") as f:
                data = pickle.load(f)
                kernel = data[kernel_key]
                kernel = kernel[None] / np.max(kernel)
        if kernel is not None:
            assert len(kernel.shape) == 4, "Expected kernel of shape [1, C, H, W]"
            self.kernel = torch.tensor(kernel, dtype=torch.float32, device="cuda")
        else:
            self.kernel = None
        if mode == "rgb":
            assert lam is not None, "Lambda must be provided for rgb mode."

        self.lam = lam
        self.sub_crop = sub_crop
        self.mode = mode
        self.gloop = gloop
        self.gscale = gscale
        self.gtstart = gtstart
        self.gtend = gtend
        self.variance_n = None
        self.batch_size = None
        self.patch_normalize = patch_normalize
        self.patch_scale_guidance = patch_scale_guidance
        self.orig_clamp = True
        self.guidance_loss = guidance_loss

    def sample(
        self,
        measurement,
        variance_n=1,
        split_n=None,
        return_intermediate=False,
        x_start=None,
        patch_size=[64, 64],
        stride_size=None,
        use_guidance=True,
        xcond=None,
        rescale_return=True,
        boundary_crop=0,
        diffusion_batching=None,
    ):
        assert len(measurement.shape) == 3, "measurement should be rank 3"
        if not torch.is_tensor(measurement):
            measurement = torch.tensor(measurement)
        measurement = measurement.to(dtype=torch.float32, device="cuda")
        self.measurement = measurement

        # Get a boundary crop mask for large image divisions
        H, W = measurement.shape[-2:]
        self.subcrop_mask = torch.zeros((1, H, W), dtype=torch.float32, device="cuda")
        self.subcrop_mask[
            :,
            boundary_crop : H - boundary_crop,
            boundary_crop : W - boundary_crop,
        ] = 1.0

        split_n = variance_n if split_n is None else int(split_n)
        assert len(patch_size) == 2, "patchsize should be [h, w]"
        h, w = measurement.shape[-2:]
        ch = self.diffusion_model._seed_channels
        sh, sw = patch_size
        assert (
            h % sh == 0 and w % sw == 0
        ), "Height and width must be perfectly divisible by sh and sw"

        ccond, ccond_pn, g = self._forward_img2patch(
            measurement, patch_size, stride_size
        )
        batch_size = ccond.shape[0]  # number patches

        if x_start is None:
            x_start = torch.randn(
                (
                    variance_n * batch_size,
                    ch,
                    *patch_size,
                ),
                dtype=torch.float32,
                device="cuda",
            )
        else:
            x_start = x_start.to(dtype=torch.float32, device="cuda")

        # ccond = self.diffusion_model.ccond_stage_model(ccond)
        ccond = self.diffusion_model.reshape_batched_variance(ccond, variance_n)
        ccond_pn = self.diffusion_model.reshape_batched_variance(ccond_pn, variance_n)

        if xcond is not None:
            xcond = xcond.to(dtype=torch.float32, device="cuda")[None]
            xcond = (xcond * 2) - 1
            xcond = self.diffusion_model.xcond_stage_model(xcond).detach()
            xcond = torch.tile(xcond, [ccond.shape[0], 1])

        self.batch_size = batch_size
        self.g = g
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.variance_n = split_n

        # Allow splitting of variance sampled draws into subgroups
        rep_calc = variance_n // split_n
        total_steps = self.n_steps if return_intermediate else 1
        imgs = torch.zeros((total_steps, variance_n, ch, h, w), device="cpu")
        x0s = torch.zeros((total_steps, variance_n, ch, h, w), device="cpu")
        self.hold_losses = []
        for repi in range(rep_calc):
            ilow = repi * split_n * batch_size
            ihigh = (repi + 1) * split_n * batch_size
            if diffusion_batching is None:
                self.diffusion_batching = int(ihigh - ilow)
            else:
                self.diffusion_batching = int(diffusion_batching)

            im = x_start[ilow:ihigh]
            use_ccond = ccond[ilow:ihigh]
            use_ccond_pn = ccond_pn[ilow:ihigh]
            use_xcond = xcond[ilow:ihigh] if xcond is not None else None
            progress_bar = tqdm(
                zip(reversed(range(self.n_steps)), reversed(self.ddim_times)),
                total=self.n_steps,
                desc="",
            )
            step_idx = 0

            for idx, i in progress_bar:
                im, x0 = self.p_sample(
                    im,
                    torch.full((im.shape[0],), i, dtype=torch.long, device="cuda"),
                    torch.full((im.shape[0],), idx, dtype=torch.long, device="cuda"),
                    use_ccond,
                    use_ccond_pn,
                    use_xcond,
                    use_guidance,
                )

                if (i == 0) or return_intermediate:
                    im_ = im.clone().detach()
                    x0_ = x0.clone().detach()

                    ## Apply LSQ patch scaling
                    if rescale_return:
                        _, im_ = self._guidance_step(im_)
                        _, x0_ = self._guidance_step(x0_)

                    im_ = self._reverse_patch2img(im_, split_n, batch_size)
                    x0_ = self._reverse_patch2img(x0_, split_n, batch_size)

                    imgs[step_idx, split_n * repi : split_n * (repi + 1)] = im_.cpu()
                    x0s[step_idx, split_n * repi : split_n * (repi + 1)] = x0_.cpu()
                    step_idx += 1

                progress_bar.set_description(f"Index {idx}, Time: {i}")

        np_hold_losses = np.concatenate(self.hold_losses, axis=1)
        print(np_hold_losses.shape)
        print(variance_n // split_n, split_n)
        np_hold_losses = rearrange(
            np_hold_losses,
            "g (t s) -> (g s) t",
            g=split_n,
            s=variance_n // split_n,
            # np_hold_losses, "(g t) s -> t (g s)", g=variance_n // split_n, s=split_n
        )
        return imgs.numpy(), x0s.numpy(), np_hold_losses

    def p_sample(self, x_t, t, ti, ccond, ccond_pn, xcond, use_guidance):
        ### Collect coefficients
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, ti, x_t.shape
        )
        sqrt_recip_alphas_cumprod_t = extract(
            self.sqrt_recip_alphas_cumprod, ti, x_t.shape
        )
        sqrt_alpha_cumprod_prev_t = torch.sqrt(
            extract(self.alphas_cumprod_prev, ti, x_t.shape)
        )
        ddim_coeff_t = extract(self.ddim_coeff, ti, x_t.shape)
        ddim_sigma_t = extract(self.ddim_sigma, ti, x_t.shape)

        ### Get noise/score
        with torch.no_grad():
            ccond_ = ccond_pn if self.patch_normalize else ccond
            model_output = self.batched_model_inference(x_t, ccond_, t, xcond)

        if self.prediction_type == "epsilon":
            epsilon = model_output
        elif self.prediction_type == "start_x":
            sqrt_alphas_cumprod_t = torch.sqrt(
                extract(self.alphas_cumprod, ti, x_t.shape)
            )
            epsilon = (
                x_t - model_output * sqrt_alphas_cumprod_t
            ) / sqrt_one_minus_alphas_cumprod_t
        else:
            raise ValueError("Unsupported model prediction type.")

        xtp = x_t.clone()
        time = t.flatten()[0]
        if use_guidance and time <= self.gtstart and time >= self.gtend:
            use_gloop = self.gloop
            xtp.requires_grad_(True)

            for _ in range(use_gloop):
                if xtp.grad is not None:
                    xtp.grad.zero_()

                ccond_ = ccond_pn if self.patch_normalize else ccond
                model_output = self.batched_model_inference(xtp, ccond_, t, xcond)

                if self.prediction_type == "epsilon":
                    x0 = (
                        xtp - sqrt_one_minus_alphas_cumprod_t * model_output
                    ) * sqrt_recip_alphas_cumprod_t
                elif self.prediction_type == "start_x":
                    x0 = model_output

                # losses, _ = self._guidance_step(x0)
                losses, _ = checkpoint(self._guidance_step, x0)
                loss = torch.sum(losses)
                loss.backward()

                with torch.no_grad():
                    xtp -= self.gscale * xtp.grad / torch.norm(xtp.grad)
                    xtp = xtp.detach().requires_grad_(True)

                torch.cuda.empty_cache()

            xtp = xtp.detach()

        ### Get x0hat and Compute DDIM Step
        pred_xstart = (
            xtp - sqrt_one_minus_alphas_cumprod_t * epsilon
        ) * sqrt_recip_alphas_cumprod_t
        pred_xstart = torch.clamp(pred_xstart, -1, 1)

        nonzero_mask = (1 - (t == 0).float()).reshape(
            x_t.shape[0], *((1,) * (len(x_t.shape) - 1))
        )
        noise = torch.randn_like(x_t)
        xtm1 = (
            sqrt_alpha_cumprod_prev_t * pred_xstart
            + ddim_coeff_t * epsilon
            + nonzero_mask * ddim_sigma_t * noise
        )

        # Save the projection loss for reference
        losses, _ = self._guidance_step(pred_xstart)
        self.hold_losses.append(losses[:, None].detach().cpu().numpy())
        return xtm1, pred_xstart

    def batched_model_inference(self, x_t, ccond, t, xcond):
        b = self.diffusion_batching
        n = x_t.shape[0]
        outputs = []

        for i in range(0, n, b):
            end = min(i + b, n)
            batch_x_t = x_t[i:end]
            batch_ccond = ccond[i:end]
            batch_t = t[i:end]
            batch_xcond = xcond[i:end] if xcond is not None else None

            batch_input = torch.cat((batch_x_t, batch_ccond), dim=1)
            batch_output = self.diffusion_model.model(
                batch_input, batch_t, context=batch_xcond
            )

            outputs.append(batch_output)

        return torch.cat(outputs, dim=0)

    def _forward_img2patch(self, ccond, patch_size, stride_size):
        ccond = ccond / torch.amax(ccond)
        ccond, g = split_patches(ccond, patch_size, stride_size)

        ccond_pn = ccond / torch.amax(ccond, axis=(-1, -2, -3), keepdim=True)
        ccond_pn = (ccond_pn * 2) - 1
        ccond = (ccond * 2) - 1

        return ccond, ccond_pn, g

    def _reverse_patch2img(self, hsi, v, b):
        # hsi = rearrange(hsi, "(v b) c h w -> b v c h w", v=v, b=b)
        hsi = combine_patches(hsi, self.g, self.patch_size, self.stride_size)
        # if self.orig_clamp:
        #     hsi = torch.clamp(hsi, -1, 1)
        # else:
        #     hsi = torch.clamp(hsi, -1, 1e3)
        # hsi = (hsi + 1) / 2
        # hsi = hsi / torch.amax(hsi, axis=(-1, -2, -3), keepdim=True)
        return hsi

    def _guidance_step(self, x0):
        measurement = self.measurement * self.subcrop_mask[None]
        x0 = rearrange(
            x0, "(v b) c h w -> b v c h w", v=self.variance_n, b=self.batch_size
        )
        x0 = torch.clamp(x0, -1, 1)
        x0 = (x0 + 1) / 2

        if self.patch_normalize:
            # x0 = self._patch_normalize(x0)
            x0 = checkpoint(self._patch_normalize, x0)

        ###
        g = self.g
        ch = 3 if self.mode == "rgb" else 1
        py, px = self.patch_size[-2:]
        if self.kernel is not None:
            ys, xs = self.kernel.shape[-2] // 2, self.kernel.shape[-1] // 2
        else:
            ys, xs = 0, 0
        fh = measurement.shape[-2] + 2 * ys
        fw = measurement.shape[-1] + 2 * xs
        meas = torch.zeros(
            size=(x0.shape[1], ch, fh, fw), dtype=torch.float32, device="cuda"
        )

        for pi in range(x0.shape[0]):
            if self.kernel is not None:
                out = general_convolve(x0[pi], self.kernel, rfft=True, mode="full")
            else:
                out = x0[pi]

            if self.mode == "gs":
                out = torch.sum(out, axis=-3, keepdim=True)
            elif self.mode == "rgb":
                out = hsi_to_rgb(out, self.lam, tensor_ordering=True, normalize=False)

            ri, ci = divmod(pi, g[-1])
            meas[
                :,
                :,
                ri * py : (ri * py + py + 2 * ys),
                ci * px : (ci * px + px + 2 * xs),
            ] = (
                meas[
                    :,
                    :,
                    ri * py : (ri * py + py + 2 * ys),
                    ci * px : (ci * px + px + 2 * xs),
                ]
                + out
            )

        meas = meas[:, :, ys:-ys, xs:-xs] if self.kernel is not None else meas
        meas = meas * self.subcrop_mask[None]

        if not self.patch_normalize:
            # TODO: Should swap this to mean
            meas = meas / torch.amax(meas, dim=(-1, -2, -3), keepdim=True)
            measurement = measurement / torch.max(measurement)

        if self.guidance_loss == "L2":
            losses = torch.sum((measurement - meas) ** 2, axis=(-1, -2, -3))
        elif self.guidance_loss == "L1":
            losses = torch.sum(torch.abs(measurement - meas), axis=(-1, -2, -3))
        else:
            raise ValueError("unknown guidance loss ")

        return losses, x0

    def _patch_normalize(self, x0):
        g = self.g
        measurement = self.measurement * self.subcrop_mask
        ch = 3 if self.mode == "rgb" else 1
        py, px = self.patch_size[-2:]

        if self.kernel is not None:
            ys, xs = self.kernel.shape[-2] // 2, self.kernel.shape[-1] // 2
        else:
            ys, xs = 0, 0
        fh = measurement.shape[-2] + 2 * ys
        fw = measurement.shape[-1] + 2 * xs
        A = torch.zeros(
            size=(*x0.shape[0:2], ch, fh, fw), dtype=torch.float32, device="cuda"
        )

        for pi in range(x0.shape[0]):
            if self.kernel is not None:
                out = general_convolve(x0[pi], self.kernel, rfft=True, mode="full")
            else:
                out = x0[pi]

            if self.mode == "gs":
                out = torch.sum(out, axis=-3, keepdim=True)
            elif self.mode == "rgb":
                out = hsi_to_rgb(out, self.lam, tensor_ordering=True, normalize=False)

            ri, ci = divmod(pi, g[-1])
            A[
                pi,
                :,
                :,
                ri * py : (ri * py + py + 2 * ys),
                ci * px : (ci * px + px + 2 * xs),
            ] = out
        A = A[:, :, :, ys:-ys, xs:-xs] if self.kernel is not None else A
        A = A * self.subcrop_mask

        cps = []
        for vi in range(A.shape[1]):
            Ai = A[:, vi].reshape(x0.shape[0], -1).T
            cp = torch.inverse(Ai.T @ Ai) @ Ai.T @ measurement.flatten()
            cps.append(cp)
        cps = torch.stack(cps).T

        x0 = x0 * cps[:, :, None, None, None].to(torch.float32)
        return x0


class DiffSSI_Large(DDIM_Sampler):
    def __init__(
        self,
        diffusion_model,
        n_steps,
        ddim_scheme="uniform",
        ddim_eta=1.0,
        gtstart=999,
        gtend=0,
        gloop=10,
        gscale=10,
        kernel=None,
        kernel_path=None,
        kernel_key=None,
        mode="gs",
        patch_normalize=True,
        lam=None,
        guidance_loss="L2",
        chunk_size=[256, 256],
    ):
        super().__init__(diffusion_model, n_steps, ddim_scheme, ddim_eta)
        assert mode in ["rgb", "gs"], "mode must be 'rgb' or 'gs'."
        if kernel_path is not None:
            assert kernel is None, "If kernel_path is provided, do not pass in a kernel"
            assert (
                kernel_key is not None
            ), "dictionary key name corresponding to kernel must be provided."
            with open(kernel_path, "rb") as f:
                data = pickle.load(f)
                kernel = data[kernel_key]
                kernel = kernel[None] / np.max(kernel)
        if kernel is not None:
            assert len(kernel.shape) == 4, "Expected kernel of shape [1, C, H, W]"
            self.kernel = torch.tensor(kernel, dtype=torch.float32, device="cuda")
        else:
            self.kernel = None
        if mode == "rgb":
            assert lam is not None, "Lambda must be provided for rgb mode."

        self.lam = lam
        self.mode = mode
        self.gloop = gloop
        self.gscale = gscale
        self.gtstart = gtstart
        self.gtend = gtend
        self.variance_n = None
        self.batch_size = None
        self.patch_normalize = patch_normalize
        self.orig_clamp = True
        self.guidance_loss = guidance_loss
        self.chunk_size = chunk_size

    def sample(
        self,
        measurement,
        variance_n=1,
        split_n=None,
        return_intermediate=False,
        x_start=None,
        patch_size=[64, 64],
        stride_size=None,
        use_guidance=True,
        xcond=None,
        rescale_return=True,
        diffusion_batching=None,
    ):
        assert len(measurement.shape) == 3, "measurement should be rank 3"
        if not torch.is_tensor(measurement):
            measurement = torch.tensor(measurement)
        measurement = measurement.to(dtype=torch.float32, device="cuda")
        self.measurement = measurement

        split_n = variance_n if split_n is None else int(split_n)
        assert len(patch_size) == 2, "patchsize should be [h, w]"
        h, w = measurement.shape[-2:]
        ch = self.diffusion_model._seed_channels
        sh, sw = patch_size
        assert (
            h % sh == 0 and w % sw == 0
        ), "Height and width must be perfectly divisible by sh and sw"

        ccond, ccond_pn, g = self._forward_img2patch(
            measurement, patch_size, stride_size
        )
        batch_size = ccond.shape[0]  # number patches

        if x_start is None:
            x_start = torch.randn(
                (
                    variance_n * batch_size,
                    ch,
                    *patch_size,
                ),
                dtype=torch.float32,
                device="cuda",
            )
        else:
            x_start = x_start.to(dtype=torch.float32, device="cuda")

        # ccond = self.diffusion_model.ccond_stage_model(ccond)
        ccond = self.diffusion_model.reshape_batched_variance(ccond, variance_n)
        ccond_pn = self.diffusion_model.reshape_batched_variance(ccond_pn, variance_n)

        if xcond is not None:
            xcond = xcond.to(dtype=torch.float32, device="cuda")[None]
            xcond = (xcond * 2) - 1
            xcond = self.diffusion_model.xcond_stage_model(xcond).detach()
            xcond = torch.tile(xcond, [ccond.shape[0], 1])

        self.batch_size = batch_size
        self.g = g
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.variance_n = split_n

        # Allow splitting of variance sampled draws into subgroups
        rep_calc = variance_n // split_n
        total_steps = self.n_steps if return_intermediate else 1
        imgs = torch.zeros((total_steps, variance_n, ch, h, w), device="cpu")
        x0s = torch.zeros((total_steps, variance_n, ch, h, w), device="cpu")
        self.hold_losses = []
        for repi in range(rep_calc):
            ilow = repi * split_n * batch_size
            ihigh = (repi + 1) * split_n * batch_size
            if diffusion_batching is None:
                self.diffusion_batching = int(ihigh - ilow)
            else:
                self.diffusion_batching = int(diffusion_batching)

            im = x_start[ilow:ihigh]
            use_ccond = ccond[ilow:ihigh]
            use_ccond_pn = ccond_pn[ilow:ihigh]
            use_xcond = xcond[ilow:ihigh] if xcond is not None else None
            progress_bar = tqdm(
                zip(reversed(range(self.n_steps)), reversed(self.ddim_times)),
                total=self.n_steps,
                desc="",
            )
            step_idx = 0

            for idx, i in progress_bar:
                im, x0 = self.p_sample(
                    im,
                    torch.full((im.shape[0],), i, dtype=torch.long, device="cuda"),
                    torch.full((im.shape[0],), idx, dtype=torch.long, device="cuda"),
                    use_ccond,
                    use_ccond_pn,
                    use_xcond,
                    use_guidance,
                )

                if (i == 0) or return_intermediate:
                    im_ = im.clone().detach()
                    x0_ = x0.clone().detach()

                    ## Apply LSQ patch scaling
                    if rescale_return:
                        _, im_ = self._guidance_step(im_)
                        _, x0_ = self._guidance_step(x0_)

                    im_ = self._reverse_patch2img(im_, split_n, batch_size)
                    x0_ = self._reverse_patch2img(x0_, split_n, batch_size)

                    imgs[step_idx, split_n * repi : split_n * (repi + 1)] = im_.cpu()
                    x0s[step_idx, split_n * repi : split_n * (repi + 1)] = x0_.cpu()
                    step_idx += 1

                progress_bar.set_description(f"Index {idx}, Time: {i}")

        np_hold_losses = np.concatenate(self.hold_losses, axis=1)
        print(np_hold_losses.shape)
        print(variance_n // split_n, split_n)
        np_hold_losses = rearrange(
            np_hold_losses,
            "g (t s) -> (g s) t",
            g=split_n,
            s=variance_n // split_n,
            # np_hold_losses, "(g t) s -> t (g s)", g=variance_n // split_n, s=split_n
        )
        return imgs.numpy(), x0s.numpy(), np_hold_losses

    def p_sample(self, x_t, t, ti, ccond, ccond_pn, xcond, use_guidance):
        ### Collect coefficients
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, ti, x_t.shape
        )
        sqrt_recip_alphas_cumprod_t = extract(
            self.sqrt_recip_alphas_cumprod, ti, x_t.shape
        )
        sqrt_alpha_cumprod_prev_t = torch.sqrt(
            extract(self.alphas_cumprod_prev, ti, x_t.shape)
        )
        ddim_coeff_t = extract(self.ddim_coeff, ti, x_t.shape)
        ddim_sigma_t = extract(self.ddim_sigma, ti, x_t.shape)

        ### Get noise/score
        with torch.no_grad():
            ccond_ = ccond_pn if self.patch_normalize else ccond
            model_output = self.batched_model_inference(x_t, ccond_, t, xcond)

        if self.prediction_type == "epsilon":
            epsilon = model_output
        elif self.prediction_type == "start_x":
            sqrt_alphas_cumprod_t = torch.sqrt(
                extract(self.alphas_cumprod, ti, x_t.shape)
            )
            epsilon = (
                x_t - model_output * sqrt_alphas_cumprod_t
            ) / sqrt_one_minus_alphas_cumprod_t
        else:
            raise ValueError("Unsupported model prediction type.")

        xtp = x_t
        time = t.flatten()[0]
        if use_guidance and time <= self.gtstart and time >= self.gtend:
            use_gloop = self.gloop
            xtp.requires_grad_(True)

            for _ in range(use_gloop):
                if xtp.grad is not None:
                    xtp.grad.zero_()

                ccond_ = ccond_pn if self.patch_normalize else ccond
                model_output = self.batched_model_inference(xtp, ccond_, t, xcond)

                if self.prediction_type == "epsilon":
                    x0 = (
                        xtp - sqrt_one_minus_alphas_cumprod_t * model_output
                    ) * sqrt_recip_alphas_cumprod_t
                elif self.prediction_type == "start_x":
                    x0 = model_output

                losses, _ = checkpoint(self._guidance_step, x0)
                loss = torch.sum(losses)
                loss.backward()

                with torch.no_grad():
                    xtp -= self.gscale * xtp.grad / torch.norm(xtp.grad)
                    xtp = xtp.detach().requires_grad_(True)

                torch.cuda.empty_cache()

            xtp = xtp.detach()

        ### Get x0hat and Compute DDIM Step
        pred_xstart = (
            xtp - sqrt_one_minus_alphas_cumprod_t * epsilon
        ) * sqrt_recip_alphas_cumprod_t
        pred_xstart = torch.clamp(pred_xstart, -1, 1)

        nonzero_mask = (1 - (t == 0).float()).reshape(
            x_t.shape[0], *((1,) * (len(x_t.shape) - 1))
        )
        noise = torch.randn_like(x_t)
        xtm1 = (
            sqrt_alpha_cumprod_prev_t * pred_xstart
            + ddim_coeff_t * epsilon
            + nonzero_mask * ddim_sigma_t * noise
        )

        # Save the projection loss for reference
        losses, _ = self._guidance_step(pred_xstart)
        self.hold_losses.append(losses[:, None].detach().cpu().numpy())
        return xtm1, pred_xstart

    def batched_model_inference(self, x_t, ccond, t, xcond):
        b = self.diffusion_batching
        n = x_t.shape[0]
        outputs = []

        for i in range(0, n, b):
            end = min(i + b, n)
            batch_x_t = x_t[i:end]
            batch_ccond = ccond[i:end]
            batch_t = t[i:end]
            batch_xcond = xcond[i:end] if xcond is not None else None

            batch_input = torch.cat((batch_x_t, batch_ccond), dim=1)
            batch_output = self.diffusion_model.model(
                batch_input, batch_t, context=batch_xcond
            )

            outputs.append(batch_output)

        return torch.cat(outputs, dim=0)

    def _forward_img2patch(self, ccond, patch_size, stride_size):
        ccond = ccond / torch.amax(ccond)
        ccond, g = split_patches(ccond, patch_size, stride_size)

        ccond_pn = ccond / torch.amax(ccond, axis=(-1, -2, -3), keepdim=True)
        ccond_pn = (ccond_pn * 2) - 1
        ccond = (ccond * 2) - 1

        return ccond, ccond_pn, g

    def _reverse_patch2img(self, hsi, v, b):
        # hsi = rearrange(hsi, "(v b) c h w -> b v c h w", v=v, b=b)
        hsi = combine_patches(hsi, self.g, self.patch_size, self.stride_size)
        # if self.orig_clamp:
        #     hsi = torch.clamp(hsi, -1, 1)
        # else:
        #     hsi = torch.clamp(hsi, -1, 1e3)
        # hsi = (hsi + 1) / 2
        # hsi = hsi / torch.amax(hsi, axis=(-1, -2, -3), keepdim=True)
        return hsi

    def _guidance_step(self, x0):
        measurement = self.measurement
        kernel = self.kernel
        g = self.g

        x0 = rearrange(
            x0, "(v b) c h w -> b v c h w", v=self.variance_n, b=self.batch_size
        )
        x0 = torch.clamp(x0, -1, 1)
        x0 = (x0 + 1) / 2

        if self.patch_normalize:
            x0 = self._patch_normalize(x0)

        ch = 3 if self.mode == "rgb" else 1
        py, px = self.patch_size[-2:]
        if kernel is not None:
            ys, xs = kernel.shape[-2] // 2, kernel.shape[-1] // 2
        else:
            ys, xs = 0, 0
        fh = measurement.shape[-2] + 2 * ys
        fw = measurement.shape[-1] + 2 * xs
        meas = torch.zeros(
            size=(x0.shape[1], ch, fh, fw), dtype=torch.float32, device="cuda"
        )

        for pi in range(x0.shape[0]):
            if kernel is not None:
                out = general_convolve(x0[pi], kernel, rfft=True, mode="full")
            else:
                out = x0[pi]

            if self.mode == "gs":
                out = torch.sum(out, axis=-3, keepdim=True)
            elif self.mode == "rgb":
                out = hsi_to_rgb(out, self.lam, tensor_ordering=True, normalize=False)

            ri, ci = divmod(pi, g[-1])
            meas[
                :,
                :,
                ri * py : (ri * py + py + 2 * ys),
                ci * px : (ci * px + px + 2 * xs),
            ] = (
                meas[
                    :,
                    :,
                    ri * py : (ri * py + py + 2 * ys),
                    ci * px : (ci * px + px + 2 * xs),
                ]
                + out
            )
        meas = meas[:, :, ys:-ys, xs:-xs] if kernel is not None else meas

        if not self.patch_normalize:
            # TODO: Should swap this to mean
            meas = meas / torch.amax(meas, dim=(-1, -2, -3), keepdim=True)
            measurement = measurement / torch.max(measurement)

        if self.guidance_loss == "L2":
            losses = torch.sum((measurement - meas) ** 2, axis=(-1, -2, -3))
        elif self.guidance_loss == "L1":
            losses = torch.sum(torch.abs(measurement - meas), axis=(-1, -2, -3))
        else:
            raise ValueError("unknown guidance loss ")

        return losses, x0

    def _patch_normalize(self, x0):
        # Input like B V C H W
        measurement = self.measurement
        chunk_size = self.chunk_size
        split_meas, _ = split_patches(measurement, chunk_size)

        patch_size = self.patch_size
        g = self.g
        nr_chunk, nr_within = (
            measurement.shape[-2] // chunk_size[-2],
            chunk_size[-2] // patch_size[-2],
        )
        nc_chunk, nc_within = (
            measurement.shape[-1] // chunk_size[-1],
            chunk_size[-1] // patch_size[-1],
        )

        out = rearrange(x0, "(g0 g1) v c h w -> g0 g1 v c h w", g0=g[0], g1=g[1])
        out = rearrange(
            out,
            "(nr_chunk nr_within) (nc_chunk nc_within) batch ch h w -> (nr_chunk nc_chunk) (nr_within nc_within) batch ch h w",
            nr_chunk=nr_chunk,
            nr_within=nr_within,
            nc_chunk=nc_chunk,
            nc_within=nc_within,
        )

        ### Do the least squares solve in loop
        all_cps = []
        for i in range(out.shape[0]):
            x0_chunk = out[i]  # b v c h w
            meas_chunk = split_meas[i]  # 1 c h
            cps = self.block_lsq_patch_norm(
                x0_chunk, meas_chunk, g=[nr_within, nc_within]
            )
            all_cps.append(cps)
        all_cps = torch.stack(all_cps)
        out = out * all_cps

        out = rearrange(
            out,
            "(nr_chunk nc_chunk) (nr_within nc_within) batch ch h w -> (nr_chunk nr_within) (nc_chunk nc_within) batch ch h w",
            nr_chunk=nr_chunk,
            nr_within=nr_within,
            nc_chunk=nc_chunk,
            nc_within=nc_within,
        )
        x0 = rearrange(out, "g0 g1 v c h w->(g0 g1) v c h w", g0=g[0], g1=g[1])

        return x0

    def block_lsq_patch_norm(self, x0_chunk, measurement_chunk, g, masked=True):
        ch = 3 if self.mode == "rgb" else 1
        py, px = self.patch_size[-2:]

        if self.kernel is not None:
            ys, xs = self.kernel.shape[-2] // 2, self.kernel.shape[-1] // 2
        else:
            ys, xs = 0, 0

        fh = measurement_chunk.shape[-2] + 2 * ys
        fw = measurement_chunk.shape[-1] + 2 * xs
        A = torch.zeros(
            size=(*x0_chunk.shape[0:2], ch, fh, fw), dtype=torch.float32, device="cuda"
        )

        for pi in range(x0_chunk.shape[0]):
            if self.kernel is not None:
                out = general_convolve(
                    x0_chunk[pi], self.kernel, rfft=True, mode="full"
                )
            else:
                out = x0_chunk[pi]

            if self.mode == "gs":
                out = torch.sum(out, axis=-3, keepdim=True)
            elif self.mode == "rgb":
                out = hsi_to_rgb(out, self.lam, tensor_ordering=True, normalize=False)

            ri, ci = divmod(pi, g[-1])
            A[
                pi,
                :,
                :,
                ri * py : (ri * py + py + 2 * ys),
                ci * px : (ci * px + px + 2 * xs),
            ] = out
        A = A[:, :, :, ys:-ys, xs:-xs] if self.kernel is not None else A

        # Apply mask to measurement chunk and A
        if masked:
            H, W = measurement_chunk.shape[-2:]
            subcrop_mask = torch.zeros((1, H, W), dtype=torch.float32, device="cuda")
            subcrop_mask[
                :,
                ys : H - ys,
                xs : W - xs,
            ] = 1.0
            A = A * subcrop_mask[None]
            measurement_chunk = measurement_chunk * subcrop_mask

            cps = []
            for vi in range(A.shape[1]):
                Ai = A[:, vi].reshape(x0_chunk.shape[0], -1).T
                cp = torch.inverse(Ai.T @ Ai) @ Ai.T @ measurement_chunk.flatten()
                cps.append(cp)
            cps = torch.stack(cps).T
        return cps[:, :, None, None, None].to(torch.float32)


class DiffSSI(DDIM_Sampler):
    def __init__(
        self,
        diffusion_model,
        n_steps,
        ddim_scheme="uniform",
        ddim_eta=1.0,
        gtstart=999,
        gtend=0,
        gloop=10,
        gscale=1,
        kernel=None,
        kernel_path=None,
        kernel_key=None,
        mode="gs",
        patch_normalize=True,
        patch_scale_guidance=True,
        sub_crop=False,
        lam=None,
        guidance_loss="L2",
        tv_scale=0.00,
        sigma_loss=1.0,
    ):
        super().__init__(diffusion_model, n_steps, ddim_scheme, ddim_eta)
        assert mode in ["rgb", "gs"], "mode must be 'rgb' or 'gs'."
        if kernel_path is not None:
            assert kernel is None, "If kernel_path is provided, do not pass in a kernel"
            assert (
                kernel_key is not None
            ), "dictionary key name corresponding to kernel must be provided."
            with open(kernel_path, "rb") as f:
                data = pickle.load(f)
                kernel = data[kernel_key]
                kernel = kernel[None] / np.max(kernel)
        if kernel is not None:
            assert len(kernel.shape) == 4, "Expected kernel of shape [1, C, H, W]"
            self.kernel = torch.tensor(kernel, dtype=torch.float32, device="cuda")
        else:
            self.kernel = None
        if mode == "rgb":
            assert lam is not None, "Lambda must be provided for rgb mode."

        self.lam = lam
        self.sub_crop = sub_crop
        self.mode = mode
        self.gloop = gloop
        self.gscale = gscale
        self.gtstart = gtstart
        self.gtend = gtend
        self.variance_n = None
        self.batch_size = None
        self.patch_normalize = patch_normalize
        self.patch_scale_guidance = patch_scale_guidance
        self.orig_clamp = True
        self.guidance_loss = guidance_loss
        self.tv_loss = TotalVariationLoss()
        self.tv_scale = tv_scale
        self.sigma_loss = sigma_loss

    def sample(
        self,
        measurement,
        variance_n=1,
        split_n=None,
        return_intermediate=False,
        x_start=None,
        patch_size=[64, 64],
        stride_size=None,
        use_guidance=True,
        xcond=None,
        rescale_return=True,
        boundary_crop=0,
        diffusion_batching=None,
    ):
        assert len(measurement.shape) == 3, "measurement should be rank 3"
        if not torch.is_tensor(measurement):
            measurement = torch.tensor(measurement)
        measurement = measurement.to(dtype=torch.float32, device="cuda")
        self.measurement = measurement

        # Get a boundary crop mask for large image divisions
        H, W = measurement.shape[-2:]
        self.subcrop_mask = torch.zeros((1, H, W), dtype=torch.float32, device="cuda")
        self.subcrop_mask[
            :,
            boundary_crop : H - boundary_crop,
            boundary_crop : W - boundary_crop,
        ] = 1.0

        split_n = variance_n if split_n is None else int(split_n)
        assert len(patch_size) == 2, "patchsize should be [h, w]"
        h, w = measurement.shape[-2:]
        ch = self.diffusion_model._seed_channels
        sh, sw = patch_size
        assert (
            h % sh == 0 and w % sw == 0
        ), "Height and width must be perfectly divisible by sh and sw"

        ccond, ccond_pn, g = self._forward_img2patch(
            measurement, patch_size, stride_size
        )
        batch_size = ccond.shape[0]  # number patches

        if x_start is None:
            x_start = torch.randn(
                (
                    variance_n * batch_size,
                    ch,
                    *patch_size,
                ),
                dtype=torch.float32,
                device="cuda",
            )
        else:
            x_start = x_start.to(dtype=torch.float32, device="cuda")

        # ccond = self.diffusion_model.ccond_stage_model(ccond)
        ccond = self.diffusion_model.reshape_batched_variance(ccond, variance_n)
        ccond_pn = self.diffusion_model.reshape_batched_variance(ccond_pn, variance_n)

        if xcond is not None:
            xcond = xcond.to(dtype=torch.float32, device="cuda")[None]
            xcond = (xcond * 2) - 1
            xcond = self.diffusion_model.xcond_stage_model(xcond).detach()
            xcond = torch.tile(xcond, [ccond.shape[0], 1])

        self.batch_size = batch_size
        self.g = g
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.variance_n = split_n

        # Allow splitting of variance sampled draws into subgroups
        rep_calc = variance_n // split_n
        total_steps = self.n_steps if return_intermediate else 1
        imgs = torch.zeros((total_steps, variance_n, ch, h, w), device="cpu")
        x0s = torch.zeros((total_steps, variance_n, ch, h, w), device="cpu")
        self.hold_losses = []
        for repi in range(rep_calc):
            ilow = repi * split_n * batch_size
            ihigh = (repi + 1) * split_n * batch_size
            if diffusion_batching is None:
                self.diffusion_batching = int(ihigh - ilow)
            else:
                self.diffusion_batching = int(diffusion_batching)

            im = x_start[ilow:ihigh]
            use_ccond = ccond[ilow:ihigh]
            use_ccond_pn = ccond_pn[ilow:ihigh]
            use_xcond = xcond[ilow:ihigh] if xcond is not None else None
            progress_bar = tqdm(
                zip(reversed(range(self.n_steps)), reversed(self.ddim_times)),
                total=self.n_steps,
                desc="",
            )
            step_idx = 0

            for idx, i in progress_bar:
                im, x0 = self.p_sample(
                    im,
                    torch.full((im.shape[0],), i, dtype=torch.long, device="cuda"),
                    torch.full((im.shape[0],), idx, dtype=torch.long, device="cuda"),
                    use_ccond,
                    use_ccond_pn,
                    use_xcond,
                    use_guidance,
                )

                if (i == 0) or return_intermediate:
                    im_ = im.clone().detach()
                    x0_ = x0.clone().detach()

                    ## Apply LSQ patch scaling
                    if rescale_return:
                        _, x0_, cps = self._guidance_step(x0_, return_scales=True)
                        if i == 0:
                            _, im_, cps = self._guidance_step(im_, return_scales=True)
                        else:
                            im_ = rearrange(
                                im_,
                                "(v b) c h w -> b v c h w",
                                v=self.variance_n,
                                b=self.batch_size,
                            )
                            im_ = torch.clamp(im_, -1, 1)
                            im_ = (im_ + 1) / 2

                    im_ = self._reverse_patch2img(im_, split_n, batch_size)
                    x0_ = self._reverse_patch2img(x0_, split_n, batch_size)

                    imgs[step_idx, split_n * repi : split_n * (repi + 1)] = im_.cpu()
                    x0s[step_idx, split_n * repi : split_n * (repi + 1)] = x0_.cpu()
                    step_idx += 1

                progress_bar.set_description(f"Index {idx}, Time: {i}")

        np_hold_losses = np.stack(self.hold_losses)
        np_hold_losses = rearrange(
            np_hold_losses, "(g t) s -> t (g s)", g=variance_n // split_n, s=split_n
        )
        return imgs.numpy(), x0s.numpy(), np_hold_losses

    def p_sample(self, x_t, t, ti, ccond, ccond_pn, xcond, use_guidance):
        ### Collect coefficients
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, ti, x_t.shape
        )
        sqrt_recip_alphas_cumprod_t = extract(
            self.sqrt_recip_alphas_cumprod, ti, x_t.shape
        )
        sqrt_alpha_cumprod_prev_t = torch.sqrt(
            extract(self.alphas_cumprod_prev, ti, x_t.shape)
        )
        ddim_coeff_t = extract(self.ddim_coeff, ti, x_t.shape)
        ddim_sigma_t = extract(self.ddim_sigma, ti, x_t.shape)

        ### Get noise/score
        with torch.no_grad():
            ccond_ = ccond_pn if self.patch_normalize else ccond
            model_output = self.batched_model_inference(x_t, ccond_, t, xcond)

        if self.prediction_type == "epsilon":
            epsilon = model_output
        elif self.prediction_type == "start_x":
            sqrt_alphas_cumprod_t = torch.sqrt(
                extract(self.alphas_cumprod, ti, x_t.shape)
            )
            epsilon = (
                x_t - model_output * sqrt_alphas_cumprod_t
            ) / sqrt_one_minus_alphas_cumprod_t
        else:
            raise ValueError("Unsupported model prediction type.")

        xtp = x_t.clone()
        time = t.flatten()[0]
        if use_guidance and time <= self.gtstart and time >= self.gtend:
            use_gloop = self.gloop
            xtp.requires_grad_(True)

            for _ in range(use_gloop):
                if xtp.grad is not None:
                    xtp.grad.zero_()

                ccond_ = ccond_pn if self.patch_normalize else ccond
                model_output = self.batched_model_inference(xtp, ccond_, t, xcond)

                if self.prediction_type == "epsilon":
                    x0 = (
                        xtp - sqrt_one_minus_alphas_cumprod_t * model_output
                    ) * sqrt_recip_alphas_cumprod_t
                elif self.prediction_type == "start_x":
                    x0 = model_output

                x0 = torch.clamp(x0, -1, 1)
                losses, _ = self._guidance_step(x0)
                loss = torch.sum(losses)
                loss.backward()

                with torch.no_grad():
                    xtp -= self.gscale * xtp.grad / torch.norm(xtp.grad)
                    xtp = xtp.detach().requires_grad_(True)

            xtp = xtp.detach()

        ### Get x0hat and Compute DDIM Step
        pred_xstart = (
            xtp - sqrt_one_minus_alphas_cumprod_t * epsilon
        ) * sqrt_recip_alphas_cumprod_t
        pred_xstart = torch.clamp(pred_xstart, -1, 1)

        nonzero_mask = (1 - (t == 0).float()).reshape(
            x_t.shape[0], *((1,) * (len(x_t.shape) - 1))
        )
        noise = torch.randn_like(x_t)
        xtm1 = (
            sqrt_alpha_cumprod_prev_t * pred_xstart
            + ddim_coeff_t * epsilon
            + nonzero_mask * ddim_sigma_t * noise
        )

        # Save the projection loss for reference
        losses, _ = self._guidance_step(pred_xstart)
        self.hold_losses.append(losses[:, 0].detach().cpu().numpy())

        return xtm1, pred_xstart

    def batched_model_inference(self, x_t, ccond, t, xcond):
        b = self.diffusion_batching
        n = x_t.shape[0]
        outputs = []

        for i in range(0, n, b):
            end = min(i + b, n)
            batch_x_t = x_t[i:end]
            batch_ccond = ccond[i:end]
            batch_t = t[i:end]
            batch_xcond = xcond[i:end] if xcond is not None else None

            batch_input = torch.cat((batch_x_t, batch_ccond), dim=1)
            batch_output = self.diffusion_model.model(
                batch_input, batch_t, context=batch_xcond
            )

            outputs.append(batch_output)

        return torch.cat(outputs, dim=0)

    def _forward_img2patch(self, ccond, patch_size, stride_size):
        ccond = ccond / torch.amax(ccond)
        ccond, g = split_patches(ccond, patch_size, stride_size)

        ccond_pn = ccond / torch.amax(ccond, axis=(-1, -2, -3), keepdim=True)
        ccond_pn = (ccond_pn * 2) - 1
        ccond = (ccond * 2) - 1

        return ccond, ccond_pn, g

    def _reverse_patch2img(self, hsi, v, b):
        # hsi = rearrange(hsi, "(v b) c h w -> b v c h w", v=v, b=b)
        hsi = combine_patches(hsi, self.g, self.patch_size, self.stride_size)
        # if self.orig_clamp:
        #     hsi = torch.clamp(hsi, -1, 1)
        # else:
        #     hsi = torch.clamp(hsi, -1, 1e3)
        # hsi = (hsi + 1) / 2
        # hsi = hsi / torch.amax(hsi, axis=(-1, -2, -3), keepdim=True)
        return hsi

    def _guidance_step(self, x0, return_scales=False):
        measurement = self.measurement * self.subcrop_mask
        x0 = rearrange(
            x0, "(v b) c h w -> b v c h w", v=self.variance_n, b=self.batch_size
        )
        x0 = torch.clamp(x0, -1, 1)
        x0 = (x0 + 1) / 2

        if self.patch_normalize:
            x0_resh = combine_patches(x0, self.g, self.patch_size, self.stride_size)
            patch_masks = generate_patch_masks(
                x0_resh.shape, self.patch_size, self.stride_size
            )

            A = torch.zeros(
                (self.batch_size, self.variance_n, *measurement.shape[-3:]),
                dtype=torch.float32,
                device="cuda",
            )
            for i in range(patch_masks.shape[0]):
                mask = patch_masks[i].to(dtype=torch.float32, device="cuda")
                if self.kernel is not None:
                    Ai = torch.clamp(
                        general_convolve(x0_resh * mask[None], self.kernel, rfft=True),
                        0.0,
                        1e6,
                    )
                else:
                    Ai = x0_resh * mask[None]

                if self.mode == "gs":
                    Ai = torch.sum(Ai, axis=-3, keepdim=True)
                elif self.mode == "rgb":
                    Ai = hsi_to_rgb(Ai, self.lam, tensor_ordering=True, normalize=False)
                A[i] = Ai
            A = A * self.subcrop_mask[None, None]

            cps = []
            for vi in range(A.shape[1]):
                Ai = A[:, vi].view(self.batch_size, -1).T
                cp = torch.inverse(Ai.T @ Ai) @ Ai.T @ measurement.flatten()
                cps.append(cp)

            cps = torch.stack(cps).T
            x0 = x0 * cps[:, :, None, None, None]

        x0_resc = combine_patches(x0, self.g, self.patch_size, self.stride_size)

        if self.kernel is not None:
            mhsi = general_convolve(x0_resc, self.kernel, rfft=True)
        else:
            mhsi = x0_resc

        if self.mode == "rgb":
            meas = hsi_to_rgb(
                mhsi,
                self.lam,
                tensor_ordering=True,
                normalize=False,
            )
        elif self.mode == "gs":
            meas = torch.sum(mhsi, dim=-3, keepdim=True)

        meas = meas * self.subcrop_mask

        if not self.patch_normalize:
            meas = meas / torch.amax(meas, dim=(-1, -2, -3), keepdim=True)
            measurement = measurement / torch.max(measurement)

        if self.guidance_loss == "L2":
            losses = (
                torch.sum((measurement - meas) ** 2, axis=(-1, -2)) / self.sigma_loss**2
            )
        elif self.guidance_loss == "L1":
            losses = torch.sum(torch.abs(measurement - meas), axis=(-1, -2))
        else:
            raise ValueError("unknown guidance loss ")

        tv_loss_val = self.tv_loss(meas)
        losses = losses + self.tv_scale * tv_loss_val

        if return_scales:
            return losses, x0, cps

        return losses, x0
