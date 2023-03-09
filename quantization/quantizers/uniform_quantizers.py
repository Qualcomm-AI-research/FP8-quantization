#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import inspect
import torch

from quantization.quantizers.rounding_utils import scale_grad_func, round_ste_func
from .utils import QuantizerNotInitializedError
from .base_quantizers import QuantizerBase


class AsymmetricUniformQuantizer(QuantizerBase):
    """
    PyTorch Module that implements Asymmetric Uniform Quantization using STE.
    Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.

    Parameters
    ----------
    n_bits: int
        Number of bits for quantization.
    scale_domain: str ('log', 'linear) with default='linear'
        Domain of scale factor
    per_channel: bool
        If True: allows for per-channel quantization
    """

    def __init__(
        self,
        n_bits,
        scale_domain="linear",
        discretizer=round_ste_func,
        discretizer_args=tuple(),
        grad_scaling=False,
        eps=1e-8,
        **kwargs
    ):
        super().__init__(n_bits=n_bits, **kwargs)

        assert scale_domain in ("linear", "log")
        self.register_buffer("_delta", None)
        self.register_buffer("_zero_float", None)

        if inspect.isclass(discretizer):
            self.discretizer = discretizer(*discretizer_args)
        else:
            self.discretizer = discretizer

        self.scale_domain = scale_domain
        self.grad_scaling = grad_scaling
        self.eps = eps

    # A few useful properties
    @property
    def delta(self):
        if self._delta is not None:
            return self._delta
        else:
            raise QuantizerNotInitializedError()

    @property
    def zero_float(self):
        if self._zero_float is not None:
            return self._zero_float
        else:
            raise QuantizerNotInitializedError()

    @property
    def is_initialized(self):
        return self._delta is not None

    @property
    def symmetric(self):
        return False

    @property
    def int_min(self):
        # integer grid minimum
        return 0.0

    @property
    def int_max(self):
        # integer grid maximum
        return 2.0**self.n_bits - 1

    @property
    def scale(self):
        if self.scale_domain == "linear":
            return torch.clamp(self.delta, min=self.eps)
        elif self.scale_domain == "log":
            return torch.exp(self.delta)

    @property
    def zero_point(self):
        zero_point = self.discretizer(self.zero_float)
        zero_point = torch.clamp(zero_point, self.int_min, self.int_max)
        return zero_point

    @property
    def x_max(self):
        return self.scale * (self.int_max - self.zero_point)

    @property
    def x_min(self):
        return self.scale * (self.int_min - self.zero_point)

    def to_integer_forward(self, x_float, *args, **kwargs):
        """
        Qunatized input to its integer representation
        Parameters
        ----------
        x_float: PyTorch Float Tensor
                Full-precision Tensor

        Returns
        -------
        x_int: PyTorch Float Tensor of integers
        """
        if self.grad_scaling:
            grad_scale = self.calculate_grad_scale(x_float)
            scale = scale_grad_func(self.scale, grad_scale)
            zero_point = (
                self.zero_point if self.symmetric else scale_grad_func(self.zero_point, grad_scale)
            )
        else:
            scale = self.scale
            zero_point = self.zero_point

        x_int = self.discretizer(x_float / scale) + zero_point
        x_int = torch.clamp(x_int, self.int_min, self.int_max)

        return x_int

    def forward(self, x_float, *args, **kwargs):
        """
        Quantizes (quantized to integer and the scales back to original domain)
        Parameters
        ----------
        x_float: PyTorch Float Tensor
                Full-precision Tensor

        Returns
        -------
        x_quant: PyTorch Float Tensor
                Quantized-Dequantized Tensor
        """
        if self.per_channel:
            self._adjust_params_per_channel(x_float)

        if self.grad_scaling:
            grad_scale = self.calculate_grad_scale(x_float)
            scale = scale_grad_func(self.scale, grad_scale)
            zero_point = (
                self.zero_point if self.symmetric else scale_grad_func(self.zero_point, grad_scale)
            )
        else:
            scale = self.scale
            zero_point = self.zero_point

        x_int = self.to_integer_forward(x_float, *args, **kwargs)
        x_quant = scale * (x_int - zero_point)

        return x_quant

    def calculate_grad_scale(self, quant_tensor):
        num_pos_levels = self.int_max  # Qp in LSQ paper
        num_elements = quant_tensor.numel()  # nfeatures or nweights in LSQ paper
        if self.per_channel:
            # In the per tensor case we do not sum the gradients over the output channel dimension
            num_elements /= quant_tensor.shape[0]

        return (num_pos_levels * num_elements) ** -0.5  # 1 / sqrt (Qn * nfeatures)

    def _adjust_params_per_channel(self, x):
        """
        Adjusts the quantization parameter tensors (delta, zero_float)
        to the input tensor shape if they don't match
        Parameters
        ----------
        x: input tensor
        """
        if x.ndim != self.delta.ndim:
            new_shape = [-1] + [1] * (len(x.shape) - 1)
            self._delta = self.delta.view(new_shape)
            if self._zero_float is not None:
                self._zero_float = self._zero_float.view(new_shape)

    def _tensorize_min_max(self, x_min, x_max):
        """
        Converts provided min max range into tensors
        Parameters
        ----------
        x_min: float or PyTorch 1D tensor
        x_max: float or PyTorch 1D tensor

        Returns
        -------
        x_min: PyTorch Tensor 0 or 1-D
        x_max: PyTorch Tensor 0 or 1-D
        """
        # Ensure a torch tensor
        if not torch.is_tensor(x_min):
            x_min = torch.tensor(x_min).float()
            x_max = torch.tensor(x_max).float()

        if x_min.dim() > 0 and len(x_min) > 1 and not self.per_channel:
            print(x_min)
            print(self.per_channel)
            raise ValueError(
                "x_min and x_max must be a float or 1-D Tensor"
                " for per-tensor quantization (per_channel=False)"
            )
        # Ensure we always use zero and avoid division by zero
        x_min = torch.min(x_min, torch.zeros_like(x_min))
        x_max = torch.max(x_max, torch.ones_like(x_max) * self.eps)

        return x_min, x_max

    def set_quant_range(self, x_min, x_max):
        """
        Instantiates the quantization parameters based on the provided
        min and max range

        Parameters
        ----------
        x_min: tensor or float
                Quantization range minimum limit
        x_max: tensor of float
                Quantization range minimum limit
        """
        self.x_min_fp32, self.x_max_fp32 = x_min, x_max
        x_min, x_max = self._tensorize_min_max(x_min, x_max)
        self._delta = (x_max - x_min) / self.int_max
        self._zero_float = (-x_min / self.delta).detach()

        if self.scale_domain == "log":
            self._delta = torch.log(self.delta)

        self._delta = self._delta.detach()

    def make_range_trainable(self):
        # Converts trainable parameters to nn.Parameters
        if self.delta not in self.parameters():
            self._delta = torch.nn.Parameter(self._delta)
            self._zero_float = torch.nn.Parameter(self._zero_float)

    def fix_ranges(self):
        # Removes trainable quantization params from nn.Parameters
        if self.delta in self.parameters():
            _delta = self._delta.data
            _zero_float = self._zero_float.data
            del self._delta  # delete the parameter
            del self._zero_float
            self.register_buffer("_delta", _delta)
            self.register_buffer("_zero_float", _zero_float)


class SymmetricUniformQuantizer(AsymmetricUniformQuantizer):
    """
    PyTorch Module that implements Symmetric Uniform Quantization using STE.
    Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.

    Parameters
    ----------
    n_bits: int
        Number of bits for quantization.
    scale_domain: str ('log', 'linear) with default='linear'
        Domain of scale factor
    per_channel: bool
        If True: allows for per-channel quantization
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("_signed", None)

    @property
    def signed(self):
        if self._signed is not None:
            return self._signed.item()
        else:
            raise QuantizerNotInitializedError()

    @property
    def symmetric(self):
        return True

    @property
    def int_min(self):
        return -(2.0 ** (self.n_bits - 1)) if self.signed else 0

    @property
    def int_max(self):
        pos_n_bits = self.n_bits - self.signed
        return 2.0**pos_n_bits - 1

    @property
    def zero_point(self):
        return 0.0

    def set_quant_range(self, x_min, x_max):
        self.x_min_fp32, self.x_max_fp32 = x_min, x_max
        x_min, x_max = self._tensorize_min_max(x_min, x_max)
        self._signed = x_min.min() < 0

        x_absmax = torch.max(x_min.abs(), x_max)
        self._delta = x_absmax / self.int_max

        if self.scale_domain == "log":
            self._delta = torch.log(self._delta)

        self._delta = self._delta.detach()

    def make_range_trainable(self):
        # Converts trainable parameters to nn.Parameters
        if self.delta not in self.parameters():
            self._delta = torch.nn.Parameter(self._delta)

    def fix_ranges(self):
        # Removes trainable quantization params from nn.Parameters
        if self.delta in self.parameters():
            _delta = self._delta.data
            del self._delta  # delete the parameter
            self.register_buffer("_delta", _delta)

    def generate_grid(self):
        x_int_rng = torch.arange(self.int_min, self.int_max + 1)
        grid = self.scale * (x_int_rng - self.zero_point)
        return grid
