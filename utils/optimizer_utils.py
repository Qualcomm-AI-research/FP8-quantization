#!/usr/bin/env python
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch


def get_lr_scheduler(optimizer, lr_schedule, epochs):
    scheduler = None
    if lr_schedule:
        if lr_schedule.startswith("multistep"):
            epochs = [int(s) for s in lr_schedule.split(":")[1:]]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, epochs)
        elif lr_schedule.startswith("cosine"):
            eta_min = float(lr_schedule.split(":")[1])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, epochs, eta_min=eta_min
            )
    return scheduler


def optimizer_lr_factory(config_optim, params, epochs):
    if config_optim.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config_optim.learning_rate,
            momentum=config_optim.momentum,
            weight_decay=config_optim.weight_decay,
        )
    elif config_optim.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            params, lr=config_optim.learning_rate, weight_decay=config_optim.weight_decay
        )
    else:
        raise ValueError()

    lr_scheduler = get_lr_scheduler(optimizer, config_optim.learning_rate_schedule, epochs)

    return optimizer, lr_scheduler
