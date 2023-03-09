#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import copy
import warnings

from torch import nn
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.pooling import _AdaptiveAvgPoolNd, _AvgPoolNd


from quantization.base_quantized_classes import QuantizedActivation, QuantizedModule
from quantization.hijacker import QuantizationHijacker, activations_set
from quantization.quantization_manager import QuantizationManager
from quantization.quantized_folded_bn import BNFusedHijacker


class QuantConv1d(QuantizationHijacker, nn.Conv1d):
    def run_forward(self, x, weight, bias, offsets=None):
        return F.conv1d(
            x.contiguous(),
            weight.contiguous(),
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class QuantConv(QuantizationHijacker, nn.Conv2d):
    def run_forward(self, x, weight, bias, offsets=None):
        return F.conv2d(
            x.contiguous(),
            weight.contiguous(),
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class QuantConvTransposeBase(QuantizationHijacker):
    def quantize_weights(self, weights):
        if self.per_channel_weights:
            # NOTE: ND tranpose conv weights are stored as (in_channels, out_channels, *)
            # instead of (out_channels, in_channels, *) for convs
            # and per-channel quantization should be applied to out channels
            # transposing before passing to quantizer is trick to avoid
            # changing logic in range estimators and quantizers
            weights = weights.transpose(1, 0).contiguous()
        weights = self.weight_quantizer(weights)
        if self.per_channel_weights:
            weights = weights.transpose(1, 0).contiguous()
        return weights


class QuantConvTranspose1d(QuantConvTransposeBase, nn.ConvTranspose1d):
    def run_forward(self, x, weight, bias, offsets=None):
        return F.conv_transpose1d(
            x.contiguous(),
            weight.contiguous(),
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class QuantConvTranspose(QuantConvTransposeBase, nn.ConvTranspose2d):
    def run_forward(self, x, weight, bias, offsets=None):
        return F.conv_transpose2d(
            x.contiguous(),
            weight.contiguous(),
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class QuantLinear(QuantizationHijacker, nn.Linear):
    def run_forward(self, x, weight, bias, offsets=None):
        return F.linear(x.contiguous(), weight.contiguous(), bias=bias)


class BNQConv1d(BNFusedHijacker, nn.Conv1d):
    def run_forward(self, x, weight, bias, offsets=None):
        return F.conv1d(
            x.contiguous(),
            weight.contiguous(),
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class BNQConv(BNFusedHijacker, nn.Conv2d):
    def run_forward(self, x, weight, bias, offsets=None):
        return F.conv2d(
            x.contiguous(),
            weight.contiguous(),
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class BNQLinear(BNFusedHijacker, nn.Linear):
    def run_forward(self, x, weight, bias, offsets=None):
        return F.linear(x.contiguous(), weight.contiguous(), bias=bias)


class QuantizedActivationWrapper(QuantizedActivation):
    """
    Wraps over a layer and quantized the activation.
    It also allow for tying the input and output quantizer which is helpful
    for layers such Average Pooling
    """

    def __init__(
        self,
        layer,
        tie_activation_quantizers=False,
        input_quantizer: QuantizationManager = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tie_activation_quantizers = tie_activation_quantizers
        if input_quantizer:
            assert isinstance(input_quantizer, QuantizationManager)
            self.activation_quantizer = input_quantizer
        self.layer = layer

    def quantize_activations_no_range_update(self, x):
        if self._quant_a:
            return self.activation_quantizer.quantizer(x)
        else:
            return x

    def forward(self, x):
        x = self.layer(x)
        if self.tie_activation_quantizers:
            # The input activation quantizer is used to quantize the activation
            # but without updating the quantization range
            return self.quantize_activations_no_range_update(x)
        else:
            return self.quantize_activations(x)

    def extra_repr(self):
        return f"tie_activation_quantizers={self.tie_activation_quantizers}"


class QuantLayerNorm(QuantizationHijacker, nn.LayerNorm):
    def run_forward(self, x, weight, bias, offsets=None):
        return F.layer_norm(
            input=x.contiguous(),
            normalized_shape=self.normalized_shape,
            weight=weight.contiguous(),
            bias=bias.contiguous(),
            eps=self.eps,
        )


class Flattener(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


# Non BN Quant Modules Map
non_bn_module_map = {
    nn.Conv1d: QuantConv1d,
    nn.Conv2d: QuantConv,
    nn.ConvTranspose1d: QuantConvTranspose1d,
    nn.ConvTranspose2d: QuantConvTranspose,
    nn.Linear: QuantLinear,
    nn.LayerNorm: QuantLayerNorm,
}

non_param_modules = (_AdaptiveAvgPoolNd, _AvgPoolNd)
# BN Quant Modules Map
bn_module_map = {nn.Conv1d: BNQConv1d, nn.Conv2d: BNQConv, nn.Linear: BNQLinear}

quant_conv_modules = (QuantConv1d, QuantConv, BNQConv1d, BNQConv)


def next_bn(module, i):
    return len(module) > i + 1 and isinstance(module[i + 1], (nn.BatchNorm2d, nn.BatchNorm1d))


def get_act(module, i):
    # Case 1: conv + act
    if len(module) - i > 1 and isinstance(module[i + 1], tuple(activations_set)):
        return module[i + 1], i + 1

    # Case 2: conv + bn + act
    if (
        len(module) - i > 2
        and next_bn(module, i)
        and isinstance(module[i + 2], tuple(activations_set))
    ):
        return module[i + 2], i + 2

    # Case 3: conv + bn + X -> return false
    # Case 4: conv + X -> return false
    return None, None


def get_conv_args(module):
    args = dict(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
    )
    if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d)):
        args["output_padding"] = module.output_padding
    return args


def get_linear_args(module):
    args = dict(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=module.bias is not None,
    )
    return args


def get_layernorm_args(module):
    args = dict(normalized_shape=module.normalized_shape, eps=module.eps)
    return args


def get_module_args(mod, act):
    if isinstance(mod, _ConvNd):
        kwargs = get_conv_args(mod)
    elif isinstance(mod, nn.Linear):
        kwargs = get_linear_args(mod)
    elif isinstance(mod, nn.LayerNorm):
        kwargs = get_layernorm_args(mod)
    else:
        raise ValueError

    kwargs["activation"] = act

    return kwargs


def fold_bn(module, i, **quant_params):
    bn = next_bn(module, i)
    act, act_idx = get_act(module, i)
    modmap = bn_module_map if bn else non_bn_module_map
    modtype = modmap[type(module[i])]

    kwargs = get_module_args(module[i], act)
    new_module = modtype(**kwargs, **quant_params)
    new_module.weight.data = module[i].weight.data.clone()

    if bn:
        new_module.gamma.data = module[i + 1].weight.data.clone()
        new_module.beta.data = module[i + 1].bias.data.clone()
        new_module.running_mean.data = module[i + 1].running_mean.data.clone()
        new_module.running_var.data = module[i + 1].running_var.data.clone()
        if module[i].bias is not None:
            new_module.running_mean.data -= module[i].bias.data
            print("Warning: bias in conv/linear before batch normalization.")
        new_module.epsilon = module[i + 1].eps

    elif module[i].bias is not None:
        new_module.bias.data = module[i].bias.data.clone()

    return new_module, i + int(bool(act)) + int(bn) + 1


def quantize_sequential(model, specials=None, tie_activation_quantizers=False, **quant_params):
    specials = specials or dict()

    i = 0
    quant_modules = []
    while i < len(model):
        if isinstance(model[i], QuantizedModule):
            quant_modules.append(model[i])
        elif type(model[i]) in non_bn_module_map:
            new_module, new_i = fold_bn(model, i, **quant_params)
            quant_modules.append(new_module)
            i = new_i
            continue

        elif type(model[i]) in specials:
            quant_modules.append(specials[type(model[i])](model[i], **quant_params))

        elif isinstance(model[i], non_param_modules):
            # Check for last quantizer
            input_quantizer = None
            if quant_modules and isinstance(quant_modules[-1], QuantizedModule):
                last_layer = quant_modules[-1]
                input_quantizer = quant_modules[-1].activation_quantizer
            elif (
                quant_modules
                and isinstance(quant_modules[-1], nn.Sequential)
                and isinstance(quant_modules[-1][-1], QuantizedModule)
            ):
                last_layer = quant_modules[-1][-1]
                input_quantizer = quant_modules[-1][-1].activation_quantizer

            if input_quantizer and tie_activation_quantizers:
                # If input quantizer is found the tie input/output act quantizers
                print(
                    f"Tying input quantizer {i-1}^th layer of type {type(last_layer)} to the "
                    f"quantized {type(model[i])} following it"
                )
                quant_modules.append(
                    QuantizedActivationWrapper(
                        model[i],
                        tie_activation_quantizers=tie_activation_quantizers,
                        input_quantizer=input_quantizer,
                        **quant_params,
                    )
                )
            else:
                # Input quantizer not found
                quant_modules.append(QuantizedActivationWrapper(model[i], **quant_params))
                if tie_activation_quantizers:
                    warnings.warn("Input quantizer not found, so we do not tie quantizers")
        else:
            quant_modules.append(quantize_model(model[i], specials=specials, **quant_params))
        i += 1
    return nn.Sequential(*quant_modules)


def quantize_model(model, specials=None, tie_activation_quantizers=False, **quant_params):
    specials = specials or dict()

    if isinstance(model, nn.Sequential):
        quant_model = quantize_sequential(
            model, specials, tie_activation_quantizers, **quant_params
        )

    elif type(model) in specials:
        quant_model = specials[type(model)](model, **quant_params)

    elif isinstance(model, non_param_modules):
        quant_model = QuantizedActivationWrapper(model, **quant_params)

    elif type(model) in non_bn_module_map:
        # If we do isinstance() then we might run into issues with modules that inherit from
        # one of these classes, for whatever reason
        modtype = non_bn_module_map[type(model)]
        kwargs = get_module_args(model, None)
        quant_model = modtype(**kwargs, **quant_params)

        quant_model.weight.data = model.weight.data
        if getattr(model, "bias", None) is not None:
            quant_model.bias.data = model.bias.data

    else:
        # Unknown type, try to quantize all child modules
        quant_model = copy.deepcopy(model)
        for name, module in quant_model._modules.items():
            new_model = quantize_model(module, specials=specials, **quant_params)
            if new_model is not None:
                setattr(quant_model, name, new_model)

    return quant_model
