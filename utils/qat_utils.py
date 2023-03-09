#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import copy
import torch

from quantization.quantized_folded_bn import BNFusedHijacker
from utils.imagenet_dataloaders import ImageNetDataLoaders


def get_dataloaders_and_model(config, load_type="fp32", **qparams):
    dataloaders = ImageNetDataLoaders(
        config.base.images_dir,
        224,
        config.base.batch_size,
        config.base.num_workers,
        config.base.interpolation,
    )

    model = config.base.architecture(
        pretrained=config.base.pretrained,
        load_type=load_type,
        model_dir=config.base.model_dir,
        **qparams,
    )
    if config.base.cuda:
        model = model.cuda()

    return dataloaders, model


class ReestimateBNStats:
    def __init__(self, model, data_loader, num_batches=50):
        super().__init__()
        self.model = model
        self.data_loader = data_loader
        self.num_batches = num_batches

    def __call__(self, engine):
        print("-- Reestimate current BN statistics --")
        reestimate_BN_stats(self.model, self.data_loader, self.num_batches)


def reestimate_BN_stats(model, data_loader, num_batches=50, store_ema_stats=False):
    # We set BN momentum to 1 an use train mode
    # -> the running mean/var have the current batch statistics
    model.eval()
    org_momentum = {}
    for name, module in model.named_modules():
        if isinstance(module, BNFusedHijacker):
            org_momentum[name] = module.momentum
            module.momentum = 1.0
            module.running_mean_sum = torch.zeros_like(module.running_mean)
            module.running_var_sum = torch.zeros_like(module.running_var)
            # Set all BNFusedHijacker modules to train mode for but not its children
            module.training = True

            if store_ema_stats:
                # Save the original EMA, make sure they are in buffers so they end in the state dict
                if not hasattr(module, "running_mean_ema"):
                    module.register_buffer("running_mean_ema", copy.deepcopy(module.running_mean))
                    module.register_buffer("running_var_ema", copy.deepcopy(module.running_var))
                else:
                    module.running_mean_ema = copy.deepcopy(module.running_mean)
                    module.running_var_ema = copy.deepcopy(module.running_var)

    # Run data for estimation
    device = next(model.parameters()).device
    batch_count = 0
    with torch.no_grad():
        for x, y in data_loader:
            model(x.to(device))
            # We save the running mean/var to a buffer
            for name, module in model.named_modules():
                if isinstance(module, BNFusedHijacker):
                    module.running_mean_sum += module.running_mean
                    module.running_var_sum += module.running_var

            batch_count += 1
            if batch_count == num_batches:
                break
    # At the end we normalize the buffer and write it into the running mean/var
    for name, module in model.named_modules():
        if isinstance(module, BNFusedHijacker):
            module.running_mean = module.running_mean_sum / batch_count
            module.running_var = module.running_var_sum / batch_count
            # We reset the momentum in case it would be used anywhere else
            module.momentum = org_momentum[name]
    model.eval()
