# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import numpy as np
import torch

from utils.distributions import ClippedGaussDistr, UniformDistr, ClippedStudentTDistr
from quantization.quant_error_estimator import (
    compute_expected_quant_mse,
    compute_expected_dot_prod_mse,
)
from quantization.quantizers.fp8_quantizer import FPQuantizer
from quantization.range_estimators import estimate_range_line_search
from quantization.quantizers.uniform_quantizers import SymmetricUniformQuantizer
from utils import seed_all


def compute_quant_error(distr, n_bits=8, n_samples=5000000, seed=10):
    seed_all(seed)
    distr_sample = torch.tensor(distr.sample((n_samples,)))
    for exp_bits in [5, 4, 3, 2, 0]:
        mantissa_bits = n_bits - 1 - exp_bits
        if exp_bits > 0:
            quant = FPQuantizer(n_bits=8, mantissa_bits=mantissa_bits, set_maxval=True)
        elif exp_bits == 0:
            quant = SymmetricUniformQuantizer(n_bits=n_bits)

        (quant_range_min, quant_range_max) = estimate_range_line_search(distr_sample, quant)
        quant_expected_mse = compute_expected_quant_mse(
            distr, quant, quant_range_min, quant_range_max, n_samples
        )
        quant_sqnr = -10.0 * np.log10(quant_expected_mse)

        dot_prod_expected_mse = compute_expected_dot_prod_mse(
            distr,
            distr,
            quant,
            quant,
            quant_range_min,
            quant_range_max,
            quant_range_min,
            quant_range_max,
        )

        dot_prod_sqnr = -10.0 * np.log10(dot_prod_expected_mse)

        print(
            "FP8 {} E {} M Quantization: expected MSE {:.2e}".format(
                exp_bits, mantissa_bits, quant_expected_mse
            ),
            " SQNR ",
            "{:.2e}\n".format(quant_sqnr),
            "Dot product:".rjust(23),
            " expected MSE {:.2e}".format(dot_prod_expected_mse),
            " SQNR ",
            "{:.2e}".format(dot_prod_sqnr),
        )


if __name__ == "__main__":
    distr_list = [
        UniformDistr(range_min=-1.0, range_max=1.0, params_dict={}),
        ClippedGaussDistr(params_dict={"mu": 0.0, "sigma": 1.0}, range_min=-10.0, range_max=10.0),
        ClippedStudentTDistr(params_dict={"nu": 8.0}, range_min=-100.0, range_max=100.0),
    ]

    for distr in distr_list:
        print("*" * 80)
        distr.print()
        compute_quant_error(distr)
