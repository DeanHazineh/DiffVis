import math
import torch
import torch.nn as nn
from abc import abstractmethod


def positional_encoding(position, dim, fq=10000.0):
    """Create sinusoidal embedding matrix with shape (num_t, dim)."""
    args = position[:, None] * torch.exp(
        torch.arange(0, dim, 2, dtype=position.dtype, device=position.device)
        * -(math.log(fq) / dim)
    )
    embeddings = torch.cat([torch.cos(args), torch.sin(args)], axis=-1)
    return embeddings


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, time_embd):
        return


class CrossAttnBlock(nn.Module):
    @abstractmethod
    def forward(self, x, cond):
        return


class AbstractSequential(nn.Sequential):
    """This is used to modify sequential in order to handle the passing in of a time embedding or context embedding."""

    def forward(self, x, time_embd, cond=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, time_embd)
            elif isinstance(layer, CrossAttnBlock):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x
