import numpy as np
import torch
from inspect import isfunction
from enum import Enum


def repeat_dimension(tensor, new_size, dim=1, rescale=True):
    """
    Repeat the tensor along the specified dimension until it reaches the new size.

    Parameters:
    - tensor: The input tensor.
    - new_size: The desired size of the repeated dimension.
    - dim: The dimension along which to repeat the tensor (default is 1).

    Returns:
    - A new tensor with the specified dimension repeated until it reaches the new size.
    """
    current_size = tensor.size(dim)
    repeat_count = (new_size + current_size - 1) // current_size
    scale = current_size / new_size if rescale else 1.0
    repeated_tensor = tensor.repeat_interleave(repeat_count, dim=dim)
    return repeated_tensor.narrow(dim, 0, new_size) * scale


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (
        1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
    )


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))
        ),
    )
    assert log_probs.shape == x.shape
    return log_probs


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def identity(x):
    return x


class StrEnum(str, Enum):
    """
    Enum subclass that converts its value to a string.

    .. code-block:: python

        from monai.utils import StrEnum

        class Example(StrEnum):
            MODE_A = "A"
            MODE_B = "B"

        assert (list(Example) == ["A", "B"])
        assert Example.MODE_A == "A"
        assert str(Example.MODE_A) == "A"
        assert monai.utils.look_up_option("A", Example) == "A"
    """

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class ModelMeanType(StrEnum):
    """Selection of model's output predictions"""

    EPSILON = "epsilon"  # The model predicts epsilon
    START_X = "start_x"  # The model predicts x_0
    PREVIOUS_X = "previous_x"  # The model predicts the previous x_{t-1}


class ModelVarType(StrEnum):
    """Selection of model's output variance"""

    FIXED_SMALL = "fixed_small"
    FIXED_LARGE = "fixed_large"
    LEARNED = "learned"
    LEARNED_RANGE = "learned_range"


def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "cosine":
        # cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = torch.clip(betas, min=0, max=0.999)

    elif schedule == "linear":
        betas = torch.linspace(
            linear_start, linear_end, n_timestep, dtype=torch.float64
        )

    elif schedule == "quadratic":
        betas = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
            )
            ** 2
        )

    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timestep)
        betas = torch.sigmoid(betas) * (linear_end - linear_start) + linear_start

    return betas
