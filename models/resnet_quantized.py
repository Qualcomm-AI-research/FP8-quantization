#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models import resnet18, resnet50

from quantization.autoquant_utils import quantize_model, Flattener, QuantizedActivationWrapper
from quantization.base_quantized_classes import QuantizedActivation, FP32Acts
from quantization.base_quantized_model import QuantizedModel


class QuantizedBlock(QuantizedActivation):
    def __init__(self, block, **quant_params):
        super().__init__(**quant_params)

        if isinstance(block, Bottleneck):
            features = nn.Sequential(
                block.conv1,
                block.bn1,
                block.relu,
                block.conv2,
                block.bn2,
                block.relu,
                block.conv3,
                block.bn3,
            )
        elif isinstance(block, BasicBlock):
            features = nn.Sequential(block.conv1, block.bn1, block.relu, block.conv2, block.bn2)

        self.features = quantize_model(features, **quant_params)
        self.downsample = (
            quantize_model(block.downsample, **quant_params) if block.downsample else None
        )

        self.relu = block.relu

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.features(x)

        out += residual
        out = self.relu(out)

        return self.quantize_activations(out)


class QuantizedResNet(QuantizedModel):
    def __init__(self, resnet, input_size=(1, 3, 224, 224), quant_setup=None, **quant_params):
        super().__init__(input_size)
        specials = {BasicBlock: QuantizedBlock, Bottleneck: QuantizedBlock}

        if hasattr(resnet, "maxpool"):
            # ImageNet ResNet case
            features = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            )
        else:
            # Tiny ImageNet ResNet case
            features = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            )

        self.features = quantize_model(features, specials=specials, **quant_params)

        if quant_setup and quant_setup == "LSQ_paper":
            # Keep avgpool intact as we quantize the input the last layer
            self.avgpool = resnet.avgpool
        else:
            self.avgpool = QuantizedActivationWrapper(
                resnet.avgpool,
                tie_activation_quantizers=True,
                input_quantizer=self.features[-1][-1].activation_quantizer,
                **quant_params,
            )
        self.flattener = Flattener()
        self.fc = quantize_model(resnet.fc, **quant_params)

        # Adapt to specific quantization setup
        if quant_setup == "LSQ":
            print("Set quantization to LSQ (first+last layer in 8 bits)")
            # Weights of the first layer
            self.features[0].weight_quantizer.quantizer.n_bits = 8
            # The quantizer of the residual (input to last layer)
            self.features[-1][-1].activation_quantizer.quantizer.n_bits = 8
            # Output of the last conv (input to last layer)
            self.features[-1][-1].features[-1].activation_quantizer.quantizer.n_bits = 8
            # Weights of the last layer
            self.fc.weight_quantizer.quantizer.n_bits = 8
            # no activation quantization of logits
            self.fc.activation_quantizer = FP32Acts()
        elif quant_setup == "LSQ_paper":
            # Weights of the first layer
            self.features[0].activation_quantizer = FP32Acts()
            self.features[0].weight_quantizer.quantizer.n_bits = 8
            # Weights of the last layer
            self.fc.activation_quantizer.quantizer.n_bits = 8
            self.fc.weight_quantizer.quantizer.n_bits = 8
            # Set all QuantizedActivations to FP32
            for layer in self.features.modules():
                if isinstance(layer, QuantizedActivation):
                    layer.activation_quantizer = FP32Acts()
        elif quant_setup == "FP_logits":
            print("Do not quantize output of FC layer")
            self.fc.activation_quantizer = FP32Acts()  # no activation quantization of logits
        elif quant_setup == "fc4":
            self.features[0].weight_quantizer.quantizer.n_bits = 8
            self.fc.weight_quantizer.quantizer.n_bits = 4
        elif quant_setup is not None and quant_setup != "all":
            raise ValueError("Quantization setup '{}' not supported for Resnet".format(quant_setup))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = self.flattener(x)
        x = self.fc(x)

        return x


def resnet18_quantized(pretrained=True, model_dir=None, load_type="fp32", **qparams):
    if load_type == "fp32":
        # Load model from pretrained FP32 weights
        fp_model = resnet18(pretrained=pretrained)
        quant_model = QuantizedResNet(fp_model, **qparams)
    elif load_type == "quantized":
        # Load pretrained QuantizedModel
        print(f"Loading pretrained quantized model from {model_dir}")
        state_dict = torch.load(model_dir)
        fp_model = resnet18()
        quant_model = QuantizedResNet(fp_model, **qparams)
        quant_model.load_state_dict(state_dict)
    else:
        raise ValueError("wrong load_type specified")
    return quant_model


def resnet50_quantized(pretrained=True, model_dir=None, load_type="fp32", **qparams):
    if load_type == "fp32":
        # Load model from pretrained FP32 weights
        fp_model = resnet50(pretrained=pretrained)
        quant_model = QuantizedResNet(fp_model, **qparams)
    elif load_type == "quantized":
        # Load pretrained QuantizedModel
        print(f"Loading pretrained quantized model from {model_dir}")
        state_dict = torch.load(model_dir)
        fp_model = resnet50()
        quant_model = QuantizedResNet(fp_model, **qparams)
        quant_model.load_state_dict(state_dict)
    else:
        raise ValueError("wrong load_type specified")
    return quant_model
