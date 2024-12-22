import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image import SpectralDistortionIndex

from diffvis.diffusion.sampler_guided_ddim import generate_uncertainty_mask


def sPSNR(s1, s2, mask=None):
    """_summary_

    Args:
        s1 (_type_): Input Tensor of shape [B, C, H, W]
        s2 (_type_): Input Tensor of shape [B, C, H, W]
        mask (_type_, optional): Spatial Mask of shape [H, W]. Defaults to None.

    Returns:
        _type_: _description_
    """
    if not torch.is_tensor(s1):
        s1 = torch.tensor(s1)
    if not torch.is_tensor(s2):
        s2 = torch.tensor(s2)
    if mask is None:
        mask = torch.ones_like(s1)
    else:
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, dtype=s1.dtype, device=s1.device)[None, None]

    error = (s1 - s2) * mask
    msec = torch.sum(error**2, axis=(-1, -2)) / torch.sum(mask, axis=(-1, -2))
    # maxval = torch.max(s1.max(), s2.max())
    maxval = torch.max((s1 * mask).max(), (s2 * mask).max())
    PSNRc = 10 * torch.log10(maxval**2 / msec)
    PSNR = torch.mean(PSNRc, axis=-1)
    return PSNR


def pPSNR(s1, s2, mask=None):
    ### INput Batch C H W
    if not torch.is_tensor(s1):
        s1 = torch.tensor(s1)
    if not torch.is_tensor(s2):
        s2 = torch.tensor(s2)
    if mask is None:
        mask = torch.ones_like(s1)
    else:
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, dtype=s1.dtype, device=s1.device)[None, None]

    error = s1 - s2
    pmse = torch.mean(error**2, axis=(-3))
    maxval = torch.max((s1 * mask).max(), (s2 * mask).max())
    PSNR = 10 * torch.log10(maxval**2 / pmse) * mask
    PSNR = torch.mean(PSNR, axis=(-1, -2))
    return PSNR


def SSIM(
    preds,
    target,
    weights=None,
    gaussian_kernel=True,
    sigma=1.5,
    kernel_size=11,
    reduction="elementwise_mean",
    data_range=None,
    k1=0.01,
    k2=0.03,
    return_full_image=False,
    return_contrast_sensitivity=False,
):
    """
    Compute the weighted SSIM for hyperspectral image cubes.
    Aligned with the official SSIM package parameters.

    Args:
        preds (torch.Tensor): Predictions from model (C, H, W)
        target (torch.Tensor): Ground truth values (C, H, W)
        weights (Optional[torch.Tensor]): Weight map (H, W), default is None
        gaussian_kernel (bool): If True, use Gaussian kernel, else uniform kernel
        sigma (float or tuple): Standard deviation for Gaussian kernel
        kernel_size (int or tuple): Size of the kernel
        reduction (str): Method to reduce metric score over individual batch scores
        data_range (float or tuple): Range of the data. If None, determined from data
        k1 (float): Parameter of SSIM
        k2 (float): Parameter of SSIM
        return_full_image (bool): If True, return full SSIM image
        return_contrast_sensitivity (bool): If True, return contrast sensitivity

    Returns:
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: SSIM score(s) and optional full image or contrast sensitivity
    """
    if preds.shape != target.shape:
        raise ValueError("Predicted and target images must have the same shape.")
    if weights is not None and preds.shape[1:] != weights.shape:
        raise ValueError(
            "Weight map must have the same spatial dimensions as the input images."
        )

    # Determine data range
    if data_range is None:
        data_range = torch.max(torch.max(preds), torch.max(target)) - torch.min(
            torch.min(preds), torch.min(target)
        )
    elif isinstance(data_range, tuple):
        preds = torch.clamp(preds, data_range[0], data_range[1])
        target = torch.clamp(target, data_range[0], data_range[1])
        data_range = data_range[1] - data_range[0]

    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    # Create kernel
    if gaussian_kernel:
        kernel = _gaussian_kernel(kernel_size, sigma).to(preds.device)
    else:
        kernel = _uniform_kernel(kernel_size).to(preds.device)

    # Compute SSIM
    num_channels = preds.shape[0]
    ssim_per_channel = torch.zeros(num_channels, device=preds.device)
    contrast_sensitivity = (
        torch.zeros(num_channels, device=preds.device)
        if return_contrast_sensitivity
        else None
    )

    for c in range(num_channels):
        if weights is not None:
            pred_c = preds[c] * weights
            target_c = target[c] * weights
        else:
            pred_c = preds[c]
            target_c = target[c]

        mu1 = F.conv2d(pred_c.unsqueeze(0).unsqueeze(0), kernel, padding="same")
        mu2 = F.conv2d(target_c.unsqueeze(0).unsqueeze(0), kernel, padding="same")
        mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(pred_c.unsqueeze(0).unsqueeze(0).pow(2), kernel, padding="same")
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(target_c.unsqueeze(0).unsqueeze(0).pow(2), kernel, padding="same")
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(
                pred_c.unsqueeze(0).unsqueeze(0) * target_c.unsqueeze(0).unsqueeze(0),
                kernel,
                padding="same",
            )
            - mu1_mu2
        )

        cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs

        if return_contrast_sensitivity:
            contrast_sensitivity[c] = torch.mean(cs)

        ssim_per_channel[c] = torch.mean(ssim_map)

    # Apply reduction
    if reduction == "elementwise_mean":
        ssim_score = torch.mean(ssim_per_channel)
    elif reduction == "sum":
        ssim_score = torch.sum(ssim_per_channel)
    else:  # "none" or None
        ssim_score = ssim_per_channel

    if return_full_image:
        return ssim_score, ssim_map.squeeze()
    elif return_contrast_sensitivity:
        return ssim_score, contrast_sensitivity
    else:
        return ssim_score


def _gaussian_kernel(kernel_size, sigma):
    """Create a Gaussian kernel."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(sigma, float):
        sigma = (sigma, sigma)

    channel = len(kernel_size)
    kernel = torch.zeros(kernel_size)
    mesh_grids = torch.meshgrid(
        [torch.arange(size, dtype=torch.float32) for size in kernel_size]
    )

    for i, std in enumerate(sigma):
        kernel += torch.exp(
            -((mesh_grids[i] - kernel_size[i] // 2) ** 2) / (2 * std**2)
        )

    kernel = kernel / torch.sum(kernel)
    return kernel.unsqueeze(0).unsqueeze(0)


def _uniform_kernel(kernel_size):
    """Create a uniform kernel."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    kernel = torch.ones(kernel_size)
    return kernel / kernel.sum()


def SDI(s1, s2, mask=None):
    # N, C, H, W
    assert s1.shape == s2.shape, "s1 and s2 must have the same shape."
    if mask is None:
        mask = torch.ones_like(s1)
    s1, s2 = s1 * mask, s2 * mask

    sdi = SpectralDistortionIndex().to(s1.device)
    batchsize = s1.shape[0]
    res = []
    for bi in range(batchsize):
        res.append(sdi(s1[bi : bi + 1], s2[bi : bi + 1]))
    return torch.stack(res)


def SAM(s1, s2, mask=None):
    # N, C, H, W
    assert s1.shape == s2.shape, "s1 and s2 must have the same shape."
    if mask is None:
        mask = torch.ones(*s1.shape[-2:], dtype=s1.dtype, device=s1.device)
    dot_product = (s1 * s2).sum(dim=-3)
    preds_norm = s1.norm(dim=1)
    target_norm = s2.norm(dim=1)
    sam_score = torch.clamp(dot_product / (preds_norm * target_norm), -1, 1).acos()

    return torch.sum(sam_score * mask[None, None], axis=(-1, -2)) / torch.sum(mask)
    # return torch.mean(sam_score, axis=(-1, -2))


def compute_eval(est, gt_hsi, drop_perc):
    avg_est = np.mean(est, axis=0, keepdims=True)
    unc = np.sum(np.var(est, axis=0), axis=0)

    s1 = torch.tensor(avg_est, dtype=torch.float32, device="cuda")
    s2 = torch.tensor(gt_hsi[None], dtype=torch.float32, device="cuda")

    spsnr = []
    ppsnr = []
    ssim = []
    sdi = []
    sam = []
    masks = []
    for perc in drop_perc:
        mask = generate_uncertainty_mask(unc, perc)
        mask = torch.tensor(mask, dtype=torch.float32, device="cuda")

        spsnr.append(sPSNR(s1, s2, mask).squeeze().cpu().numpy().item())
        ppsnr.append(pPSNR(s1, s2, mask)[0].cpu().numpy().item())
        ssim.append(
            [SSIM(s1[i], s2[i], mask) for i in range(s1.shape[0])][0]
            .cpu()
            .numpy()
            .item()
        )
        sdi.append(SDI(s1, s2)[0].cpu().numpy().item())
        sam.append(SAM(s1, s2, mask)[0].cpu().numpy().item())
        masks.append(mask.cpu().numpy())

    out = {
        "spsnr": spsnr,
        "ppsnr": ppsnr,
        "ssim": ssim,
        "sdi": sdi,
        "sam": sam,
        "masks": masks,
        "unc": unc,
    }

    return out
