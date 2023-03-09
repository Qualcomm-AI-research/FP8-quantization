#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch
import numpy as np
from utils.grid import integrate_pdf_grid_func_analyt
from quantization.quantizers.fp8_quantizer import FPQuantizer, generate_all_float_values_scaled


def generate_integr_grid_piecewise(integr_discontin, num_intervals_smallest_bin):
    bin_widths = np.diff(integr_discontin)
    min_bin_width = np.min(bin_widths[bin_widths > 0.0])
    integr_min_step = min_bin_width / num_intervals_smallest_bin
    grid_list = []
    for i in range(len(integr_discontin) - 1):
        curr_interv_min = integr_discontin[i]
        curr_interv_max = integr_discontin[i + 1]
        curr_interv_width = curr_interv_max - curr_interv_min

        if curr_interv_width == 0.0:
            continue
        assert curr_interv_min < curr_interv_max
        curr_interv_n_subintervals = np.ceil(curr_interv_width / integr_min_step).astype("int")
        curr_interv_n_pts = curr_interv_n_subintervals + 1
        lspace = torch.linspace(curr_interv_min, curr_interv_max, curr_interv_n_pts)
        grid_list.append(lspace)

    grid_all = torch.cat(grid_list)
    grid_all_no_dup = torch.unique(grid_all)

    return grid_all_no_dup


def estimate_rounding_error_analyt(distr, grid):
    err = integrate_pdf_grid_func_analyt(distr, grid, "integr_interv_p_sqr_r")
    return err


def estimate_dot_prod_error_analyt(distr_x, grid_x, distr_y, grid_y):
    rounding_err_x = integrate_pdf_grid_func_analyt(distr_x, grid_x, "integr_interv_p_sqr_r")
    rounding_err_y = integrate_pdf_grid_func_analyt(distr_y, grid_y, "integr_interv_p_sqr_r")
    second_moment_x = distr_x.eval_non_central_second_moment()
    second_moment_y = distr_y.eval_non_central_second_moment()
    y_p_y_R_y_signed = integrate_pdf_grid_func_analyt(distr_y, grid_y, "integr_interv_x_p_signed_r")
    x_p_x_R_x_signed = integrate_pdf_grid_func_analyt(distr_x, grid_x, "integr_interv_x_p_signed_r")

    term_rounding_err_x_moment_y = rounding_err_x * second_moment_y
    term_rounding_err_y_moment_x = rounding_err_y * second_moment_x
    term_mixed_rounding_err = rounding_err_x * rounding_err_y
    term_mixed_R_signed = 2.0 * y_p_y_R_y_signed * x_p_x_R_x_signed
    term_rounding_err_x_R_y_signed = 2.0 * rounding_err_x * y_p_y_R_y_signed
    term_rounding_err_y_R_x_signed = 2.0 * rounding_err_y * x_p_x_R_x_signed

    total_sum = (
        term_rounding_err_x_moment_y
        + term_rounding_err_y_moment_x
        + term_mixed_R_signed
        + term_mixed_rounding_err
        + term_rounding_err_x_R_y_signed
        + term_rounding_err_y_R_x_signed
    )

    return total_sum


def estimate_rounding_error_empirical(W, quantizer, range_min, range_max):
    quantizer.set_quant_range(range_min, range_max)
    W_int_quant = quantizer.forward(W)

    round_err_sqr_int_quant_emp = (W_int_quant - W) ** 2
    res = torch.mean(round_err_sqr_int_quant_emp.flatten()).item()
    return res


def estimate_dot_prod_error_empirical(
    x, y, quantizer_x, quantizer_y, x_range_min, x_range_max, y_range_min, y_range_max
):
    quantizer_x.set_quant_range(x_range_min, x_range_max)
    quantizer_y.set_quant_range(y_range_min, y_range_max)
    x_quant = quantizer_x.forward(x)
    y_quant = quantizer_y.forward(y)

    scalar_prod_err_emp = (torch.mul(x, y) - torch.mul(x_quant, y_quant)) ** 2
    res = torch.mean(scalar_prod_err_emp.flatten()).item()
    return res


def compute_expected_dot_prod_mse(
    distr_x,
    distr_y,
    quant_x,
    quant_y,
    quant_x_range_min,
    quant_x_range_max,
    quant_y_range_min,
    quant_y_range_max,
    num_samples=2000000,
):

    quant_x.set_quant_range(quant_x_range_min, quant_x_range_max)
    quant_y.set_quant_range(quant_y_range_min, quant_y_range_max)
    if isinstance(quant_x, FPQuantizer):
        grid_x = generate_all_float_values_scaled(
            quant_x.n_bits, quant_x.ebits, quant_x.default_bias, quant_x_range_max
        )
    else:
        grid_x = quant_x.generate_grid().numpy()

    if isinstance(quant_y, FPQuantizer):
        grid_y = generate_all_float_values_scaled(
            quant_y.n_bits, quant_y.ebits, quant_y.default_bias, quant_y_range_max
        )
    else:
        grid_y = quant_x.generate_grid().numpy()

    err_analyt = estimate_dot_prod_error_analyt(distr_x, grid_x, distr_y, grid_y)
    distr_x_sample = torch.tensor(distr_x.sample((num_samples,)))
    distr_y_sample = torch.tensor(distr_x.sample((num_samples,)))
    err_emp = estimate_dot_prod_error_empirical(
        distr_x_sample,
        distr_y_sample,
        quant_x,
        quant_y,
        quant_x_range_min,
        quant_x_range_max,
        quant_y_range_min,
        quant_y_range_max,
    )

    rel_err = np.abs((err_emp - err_analyt) / err_analyt)
    return err_analyt


def compute_expected_quant_mse(distr, quant, quant_range_min, quant_range_max, num_samples):
    quant.set_quant_range(quant_range_min, quant_range_max)
    if isinstance(quant, FPQuantizer):
        grid = generate_all_float_values_scaled(
            quant.n_bits, quant.ebits, quant.default_bias, quant_range_max
        )
    else:
        grid = quant.generate_grid().numpy()

    err_analyt = estimate_rounding_error_analyt(distr, grid)
    distr_sample = torch.tensor(
        distr.sample(
            num_samples,
        )
    )
    err_emp = estimate_rounding_error_empirical(
        distr_sample, quant, quant_range_min, quant_range_max
    )

    rel_err = np.abs((err_emp - err_analyt) / err_analyt)
    if rel_err > 0.1:
        print(
            "Warning: the relative difference between the analytical and empirical error estimate is too high,\n"
            "please consider increasing the number of samples for the quantization range estimator."
        )

    return err_analyt
