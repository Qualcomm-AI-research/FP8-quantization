#!/usr/bin/env python
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.


import torch
import torch.serialization

from quantization.quantizers import QuantizerBase
from quantization.quantizers.rounding_utils import ParametrizedGradEstimatorBase
from quantization.range_estimators import RangeEstimators
from utils import StopForwardException, get_layer_by_name


def separate_quantized_model_params(quant_model):
    """
    This method separates the parameters of the quantized model to 4 categories.
    Parameters
    ----------
    quant_model:      (QuantizedModel)

    Returns
    -------
    quant_params:       (list)
        Quantization parameters, e.g. delta and zero_float
    model_params:    (list)
        The model parameters of the base model without any quantization operations
    grad_params:        (list)
        Parameters found in the gradient estimators (ParametrizedGradEstimatorBase)
    -------

    """
    quant_params, grad_params = [], []
    quant_params_names, grad_params_names = [], []
    for mod_name, module in quant_model.named_modules():
        if isinstance(module, QuantizerBase):
            for name, param in module.named_parameters(recurse=False):
                quant_params.append(param)
                quant_params_names.append(".".join((mod_name, name)))
        if isinstance(module, ParametrizedGradEstimatorBase):
            # gradient estimator params
            for name, param in module.named_parameters(recurse=False):
                grad_params.append(param)
                grad_params_names.append(".".join((mod_name, name)))

    def tensor_in_list(tensor, lst):
        return any([e is tensor for e in lst])

    found_params = quant_params + grad_params

    model_params = [p for p in quant_model.parameters() if not tensor_in_list(p, found_params)]
    model_param_names = [
        n for n, p in quant_model.named_parameters() if not tensor_in_list(p, found_params)
    ]

    print("Quantization parameters ({}):".format(len(quant_params_names)))
    print(quant_params_names)

    print("Gradient estimator parameters ({}):".format(len(grad_params_names)))
    print(grad_params_names)

    print("Other model parameters ({}):".format(len(model_param_names)))
    print(model_param_names)

    assert len(model_params + quant_params + grad_params) == len(
        list(quant_model.parameters())
    ), "{}; {}; {} -- {}".format(
        len(model_params), len(quant_params), len(grad_params), len(list(quant_model.parameters()))
    )

    return quant_params, model_params, grad_params


def pass_data_for_range_estimation(
    loader, model, act_quant, weight_quant, max_num_batches=20, cross_entropy_layer=None, inp_idx=0
):
    print("\nEstimate quantization ranges on training data")
    model.set_quant_state(weight_quant, act_quant)
    # Put model in eval such that BN EMA does not get updated
    model.eval()

    if cross_entropy_layer is not None:
        layer_xent = get_layer_by_name(model, cross_entropy_layer)
        if layer_xent:
            print('Set cross entropy estimator for layer "{}"'.format(cross_entropy_layer))
            act_quant_mgr = layer_xent.activation_quantizer
            act_quant_mgr.range_estimator = RangeEstimators.cross_entropy.cls(
                per_channel=act_quant_mgr.per_channel,
                quantizer=act_quant_mgr.quantizer,
                **act_quant_mgr.range_estim_params,
            )
        else:
            raise ValueError("Cross-entropy layer not found")

    batches = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for i, data in enumerate(loader):
            try:
                if isinstance(data, (tuple, list)):
                    x = data[inp_idx].to(device=device)
                    batches.append(x.data.cpu().numpy())
                    model(x)
                    print(f"proccesed step={i}")
                else:
                    x = {k: v.to(device=device) for k, v in data.items()}
                    model(**x)
                    print(f"proccesed step={i}")

                if i >= max_num_batches - 1 or not act_quant:
                    break
            except StopForwardException:
                pass
        return batches


def set_range_estimators(config, model):
    print("Make quantizers learnable")
    model.learn_ranges()

    if config.qat.grad_scaling:
        print("Activate gradient scaling")
        model.grad_scaling(True)

    # Ensure we have the desired quant state
    model.set_quant_state(config.quant.weight_quant, config.quant.act_quant)
