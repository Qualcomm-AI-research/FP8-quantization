#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

from enum import auto

from torch import nn
from quantization.quantizers import QuantizerBase
from quantization.quantizers.utils import QuantizerNotInitializedError
from quantization.range_estimators import RangeEstimators, RangeEstimatorBase
from utils import BaseEnumOptions

from quantization.quantizers.uniform_quantizers import (
    SymmetricUniformQuantizer,
    AsymmetricUniformQuantizer,
)
from quantization.quantizers.fp8_quantizer import FPQuantizer

from utils import ClassEnumOptions, MethodMap


class QMethods(ClassEnumOptions):
    symmetric_uniform = MethodMap(SymmetricUniformQuantizer)
    asymmetric_uniform = MethodMap(AsymmetricUniformQuantizer)
    fp_quantizer = MethodMap(FPQuantizer)


class QuantizationManager(nn.Module):
    """Implementation of Quantization and Quantization Range Estimation

    Parameters
    ----------
    n_bits: int
        Number of bits for the quantization.
    qmethod: QMethods member (Enum)
        The quantization scheme to use, e.g. symmetric_uniform, asymmetric_uniform,
        qmn_uniform etc.
    init: RangeEstimators member (Enum)
        Initialization method for the grid from
    per_channel: bool
        If true, will use a separate quantization grid for each kernel/channel.
    x_min: float or PyTorch Tensor
        The minimum value which needs to be represented.
    x_max: float or PyTorch Tensor
        The maximum value which needs to be represented.
    qparams: kwargs
        dictionary of quantization parameters to passed to the quantizer instantiation
    range_estim_params: kwargs
         dictionary of parameters to passed to the range estimator instantiation
    """

    def __init__(
        self,
        qmethod: QuantizerBase = QMethods.symmetric_uniform.cls,
        init: RangeEstimatorBase = RangeEstimators.current_minmax.cls,
        per_channel=False,
        x_min=None,
        x_max=None,
        qparams=None,
        range_estim_params=None,
    ):
        super().__init__()
        self.state = Qstates.estimate_ranges
        self.qmethod = qmethod
        self.init = init
        self.per_channel = per_channel
        self.qparams = qparams if qparams else {}
        self.range_estim_params = range_estim_params if range_estim_params else {}
        self.range_estimator = None

        # define quantizer
        self.quantizer = self.qmethod(per_channel=self.per_channel, **qparams)
        self.quantizer.state = self.state

        # define range estimation method for quantizer initialisation
        if x_min is not None and x_max is not None:
            self.set_quant_range(x_min, x_max)
            self.fix_ranges()
        else:
            # set up the collector function to set the ranges
            self.range_estimator = self.init(
                per_channel=self.per_channel, quantizer=self.quantizer, **self.range_estim_params
            )

    @property
    def n_bits(self):
        return self.quantizer.n_bits

    def estimate_ranges(self):
        self.state = Qstates.estimate_ranges
        self.quantizer.state = self.state

    def fix_ranges(self):
        if self.quantizer.is_initialized:
            self.state = Qstates.fix_ranges
            self.quantizer.state = self.state
        else:
            raise QuantizerNotInitializedError()

    def learn_ranges(self):
        self.quantizer.make_range_trainable()
        self.state = Qstates.learn_ranges
        self.quantizer.state = self.state

    def estimate_ranges_train(self):
        self.state = Qstates.estimate_ranges_train
        self.quantizer.state = self.state

    def reset_ranges(self):
        self.range_estimator.reset()
        self.quantizer.reset()
        self.estimate_ranges()

    def forward(self, x):
        if self.state == Qstates.estimate_ranges or (
            self.state == Qstates.estimate_ranges_train and self.training
        ):
            # Note this can be per tensor or per channel
            cur_xmin, cur_xmax = self.range_estimator(x)
            self.set_quant_range(cur_xmin, cur_xmax)

        return self.quantizer(x)

    def set_quant_range(self, x_min, x_max):
        self.quantizer.set_quant_range(x_min, x_max)

    def extra_repr(self):
        return "state={}".format(self.state.name)


class Qstates(BaseEnumOptions):
    estimate_ranges = auto()  # ranges are updated in eval and train mode
    fix_ranges = auto()  # quantization ranges are fixed for train and eval
    learn_ranges = auto()  # quantization params are nn.Parameters
    estimate_ranges_train = auto()  # quantization ranges are updated during train and fixed for
    # eval
