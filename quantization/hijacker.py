#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import copy

from timm.models.layers.activations import Swish, HardSwish, HardSigmoid
from timm.models.layers.activations_me import SwishMe, HardSwishMe, HardSigmoidMe
from torch import nn

from quantization.base_quantized_classes import QuantizedModule
from quantization.quantization_manager import QuantizationManager
from quantization.range_estimators import RangeEstimators

activations_set = [
    nn.ReLU,
    nn.ReLU6,
    nn.Hardtanh,
    nn.Sigmoid,
    nn.Tanh,
    nn.GELU,
    nn.PReLU,
    Swish,
    SwishMe,
    HardSwish,
    HardSwishMe,
    HardSigmoid,
    HardSigmoidMe,
]


class QuantizationHijacker(QuantizedModule):
    """Mixin class that 'hijacks' the forward pass in a module to perform quantization and
    dequantization on the weights and output distributions.

    Usage:
    To make a quantized nn.Linear layer:
    class HijackedLinear(QuantizationHijacker, nn.Linear):
        pass
    """

    def __init__(self, *args, activation: nn.Module = None, **kwargs):

        super().__init__(*args, **kwargs)
        if activation:
            assert isinstance(activation, tuple(activations_set)), str(activation)

        self.activation_function = copy.deepcopy(activation) if activation else None

        self.activation_quantizer = QuantizationManager(
            qmethod=self.act_method,
            init=self.act_range_method,
            qparams=self.act_qparams,
            range_estim_params=self.act_range_options,
        )

        if self.weight_range_method == RangeEstimators.current_minmax:
            weight_init_params = dict(percentile=self.percentile)
        else:
            weight_init_params = self.weight_range_options

        self.weight_quantizer = QuantizationManager(
            qmethod=self.method,
            init=self.weight_range_method,
            per_channel=self.per_channel_weights,
            qparams=self.weight_qparams,
            range_estim_params=weight_init_params,
        )

    def forward(self, x, offsets=None):
        # Quantize input
        if self.quantize_input and self._quant_a:
            x = self.activation_quantizer(x)

        # Get quantized weight
        weight, bias = self.get_params()
        res = self.run_forward(x, weight, bias, offsets=offsets)

        # Apply fused activation function
        if self.activation_function is not None:
            res = self.activation_function(res)

        # Quantize output
        if not self.quantize_input and self._quant_a:
            res = self.activation_quantizer(res)
        return res

    def get_params(self):

        weight, bias = self.get_weight_bias()

        if self._quant_w:
            weight = self.quantize_weights(weight)

        return weight, bias

    def quantize_weights(self, weights):
        return self.weight_quantizer(weights)

    def get_weight_bias(self):
        bias = None
        if hasattr(self, "bias"):
            bias = self.bias
        return self.weight, bias

    def run_forward(self, x, weight, bias, offsets=None):
        # Performs the actual linear operation of the layer
        raise NotImplementedError()

    def extra_repr(self):
        activation = "input" if self.quantize_input else "output"
        return f"{super().extra_repr()}-{activation}"
