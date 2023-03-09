#!/usr/bin/env python
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Checkpoint, global_step_from_engine
from torch.optim import Optimizer


def create_trainer_engine(
    model,
    optimizer,
    criterion,
    metrics,
    data_loaders,
    lr_scheduler=None,
    save_checkpoint_dir=None,
    device="cuda",
):
    # Create trainer
    trainer = create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=criterion,
        device=device,
        output_transform=custom_output_transform,
    )

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    # Add lr_scheduler
    if lr_scheduler:
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: lr_scheduler.step())

    # Create evaluator
    evaluator = create_supervised_evaluator(model=model, metrics=metrics, device=device)

    # Save model checkpoint
    if save_checkpoint_dir:
        to_save = {"model": model, "optimizer": optimizer}
        if lr_scheduler:
            to_save["lr_scheduler"] = lr_scheduler
        checkpoint = Checkpoint(
            to_save,
            save_checkpoint_dir,
            n_saved=1,
            global_step_transform=global_step_from_engine(trainer),
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)

    # Add hooks for logging metrics
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results, optimizer)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, run_evaluation_for_training, evaluator, data_loaders.val_loader
    )

    return trainer, evaluator


def custom_output_transform(x, y, y_pred, loss):
    return y_pred, y


def log_training_results(trainer, optimizer):
    learning_rate = optimizer.param_groups[0]["lr"]
    log_metrics(trainer.state.metrics, "Training", trainer.state.epoch, learning_rate)


def run_evaluation_for_training(trainer, evaluator, val_loader):
    evaluator.run(val_loader)
    log_metrics(evaluator.state.metrics, "Evaluation", trainer.state.epoch)


def log_metrics(metrics, stage: str = "", training_epoch=None, learning_rate=None):
    log_text = "  {}".format(metrics) if metrics else ""
    if training_epoch is not None:
        log_text = "Epoch: {}".format(training_epoch) + log_text
    if learning_rate and learning_rate > 0.0:
        log_text += "  Learning rate: {:.2E}".format(learning_rate)
    log_text = "Results - " + log_text
    if stage:
        log_text = "{} ".format(stage) + log_text
    print(log_text, flush=True)


def setup_tensorboard_logger(trainer, evaluator, output_path, optimizers=None):
    logger = TensorboardLogger(logdir=output_path)

    # Attach the logger to log loss and accuracy for both training and validation
    for tag, cur_evaluator in [("train", trainer), ("validation", evaluator)]:
        logger.attach_output_handler(
            cur_evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=global_step_from_engine(trainer),
        )

    # Log optimizer parameters
    if isinstance(optimizers, Optimizer):
        optimizers = {None: optimizers}

    for k, optimizer in optimizers.items():
        logger.attach_opt_params_handler(
            trainer, Events.EPOCH_COMPLETED, optimizer, param_name="lr", tag=k
        )

    return logger
