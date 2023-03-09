# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
import logging
import os

import click
from ignite.contrib.handlers import ProgressBar
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, TopKCategoricalAccuracy, Loss
from torch.nn import CrossEntropyLoss

from quantization.utils import pass_data_for_range_estimation
from utils import DotDict
from utils.click_options import (
    qat_options,
    quantization_options,
    fp8_options,
    quant_params_dict,
    base_options,
)
from utils.qat_utils import get_dataloaders_and_model, ReestimateBNStats


class Config(DotDict):
    pass


@click.group()
def fp8_cmd_group():
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


pass_config = click.make_pass_decorator(Config, ensure=True)


@fp8_cmd_group.command()
@pass_config
@base_options
@fp8_options
@quantization_options
@qat_options
@click.option(
    "--load-type",
    type=click.Choice(["fp32", "quantized"]),
    default="quantized",
    help='Either "fp32", or "quantized". Specify weather to load a quantized or a FP ' "model.",
)
def validate_quantized(config, load_type):
    """
    function for running validation on pre-trained quantized models
    """
    print("Setting up network and data loaders")
    qparams = quant_params_dict(config)

    dataloaders, model = get_dataloaders_and_model(config=config, load_type=load_type, **qparams)

    if load_type == "fp32":
        # Estimate ranges using training data
        pass_data_for_range_estimation(
            loader=dataloaders.train_loader,
            model=model,
            act_quant=config.quant.act_quant,
            weight_quant=config.quant.weight_quant,
            max_num_batches=config.quant.num_est_batches,
        )
        # Ensure we have the desired quant state
        model.set_quant_state(config.quant.weight_quant, config.quant.act_quant)

    # Fix ranges
    model.fix_ranges()

    # Create evaluator
    loss_func = CrossEntropyLoss()
    metrics = {
        "top_1_accuracy": Accuracy(),
        "top_5_accuracy": TopKCategoricalAccuracy(),
        "loss": Loss(loss_func),
    }

    pbar = ProgressBar()
    evaluator = create_supervised_evaluator(
        model=model, metrics=metrics, device="cuda" if config.base.cuda else "cpu"
    )
    pbar.attach(evaluator)
    print("Model with the ranges estimated:\n{}".format(model))

    # BN Re-estimation
    if config.qat.reestimate_bn_stats:
        ReestimateBNStats(
            model, dataloaders.train_loader, num_batches=int(0.02 * len(dataloaders.train_loader))
        )(None)

    print("Start quantized validation")
    evaluator.run(dataloaders.val_loader)
    final_metrics = evaluator.state.metrics
    print(final_metrics)


if __name__ == "__main__":
    fp8_cmd_group()
