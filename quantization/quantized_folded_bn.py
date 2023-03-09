#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.conv import _ConvNd

from quantization.hijacker import QuantizationHijacker


class BNFusedHijacker(QuantizationHijacker):
    """Extension to the QuantizationHijacker that fuses batch normalization (BN) after a weight
    layer into a joined module. The parameters and the statistics of the BN layer remain in
    full-precision.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("bias", None)  # Bias will be learned by BN params
        super().__init__(*args, **kwargs, bias=False)
        bn_dim = self.get_bn_dim()
        self.register_buffer("running_mean", torch.zeros(bn_dim))
        self.register_buffer("running_var", torch.ones(bn_dim))
        self.momentum = kwargs.pop("momentum", 0.1)
        self.gamma = nn.Parameter(torch.ones(bn_dim))
        self.beta = nn.Parameter(torch.zeros(bn_dim))
        self.epsilon = kwargs.get("eps", 1e-5)
        self.bias = None

    def forward(self, x):
        # Quantize input
        if self.quantize_input and self._quant_a:
            x = self.activation_quantizer(x)

        # Get quantized weight
        weight, bias = self.get_params()
        res = self.run_forward(x, weight, bias)

        res = F.batch_norm(
            res,
            self.running_mean,
            self.running_var,
            self.gamma,
            self.beta,
            self.training,
            self.momentum,
            self.epsilon,
        )
        # Apply fused activation function
        if self.activation_function is not None:
            res = self.activation_function(res)

        # Quantize output
        if not self.quantize_input and self._quant_a:
            res = self.activation_quantizer(res)
        return res

    def get_bn_dim(self):
        if isinstance(self, nn.Linear):
            return self.out_features
        elif isinstance(self, _ConvNd):
            return self.out_channels
        else:
            msg = (
                f"Unsupported type used: {self}. Must be a linear or (transpose)-convolutional "
                f"nn.Module"
            )
            raise NotImplementedError(msg)
