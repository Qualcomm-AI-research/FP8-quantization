#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch
import numpy as np
from utils.distributions import ClippedGaussDistr, ClippedStudentTDistr


def rounding_error_abs_nearest(x_float, grid):
    n_grid = grid.size
    grid_row = np.array(grid).reshape(1, n_grid)
    m_vals = x_float.numel()
    x_float = x_float.cpu().detach().numpy().reshape(m_vals, 1)

    dist = np.abs(x_float - grid_row)
    min_dist = np.min(dist, axis=1)

    return min_dist


def quant_scalar_nearest(x_float, grid):
    dist = np.abs(x_float - grid)
    idx = np.argmin(dist)
    q_x = grid[idx]
    return q_x


def clip_grid_exclude_bounds(grid, min_val, max_val):
    idx_subset = torch.logical_and(grid > min_val, grid < max_val)
    return grid[idx_subset]


def clip_grid_include_bounds(grid, min_val, max_val):
    idx_subset = torch.logical_and(grid >= min_val, grid <= max_val)
    return grid[idx_subset]


def clip_grid_add_bounds(grid, min_val, max_val):
    grid_clipped = clip_grid_exclude_bounds(grid, min_val, max_val)
    bounds_np = np.array([min_val, max_val])
    clipped_with_bounds = np.sort(np.concatenate((grid_clipped, bounds_np)))
    return clipped_with_bounds


def integrate_pdf_grid_func_analyt(distr, grid, distr_attr_func_name):
    grid = np.sort(grid)
    interval_integr_func = getattr(distr, distr_attr_func_name)
    res = 0.0

    if distr.range_min < grid[0]:
        res += interval_integr_func(distr.range_min, grid[0], grid[0])

    for i_interval in range(0, len(grid) - 1):
        grid_mid = 0.5 * (grid[i_interval] + grid[i_interval + 1])

        first_half_a = max(grid[i_interval], distr.range_min)
        first_half_b = min(grid_mid, distr.range_max)

        second_half_a = max(grid_mid, distr.range_min)
        second_half_b = min(grid[i_interval + 1], distr.range_max)

        if first_half_a < first_half_b:
            res += interval_integr_func(first_half_a, first_half_b, grid[i_interval])
        if second_half_a < second_half_b:
            res += interval_integr_func(second_half_a, second_half_b, grid[i_interval + 1])

    if distr.range_max > grid[-1]:
        res += interval_integr_func(grid[-1], distr.range_max, grid[-1])

    if (
        isinstance(distr, ClippedGaussDistr) or isinstance(distr, ClippedStudentTDistr)
    ) and distr_attr_func_name == "integr_interv_x_p_signed_r":
        q_range_min = quant_scalar_nearest(torch.Tensor([distr.range_min]), grid)
        q_range_max = quant_scalar_nearest(torch.Tensor([distr.range_max]), grid)

        term_point_mass = (
            distr.range_min * (q_range_min - distr.range_min) * distr.point_mass_range_min
            + distr.range_max * (q_range_max - distr.range_max) * distr.point_mass_range_max
        )
        res += term_point_mass
    elif (
        isinstance(distr, ClippedGaussDistr) or isinstance(distr, ClippedStudentTDistr)
    ) and distr_attr_func_name == "integr_interv_p_sqr_r":
        q_range_min = quant_scalar_nearest(torch.Tensor([distr.range_min]), grid)
        q_range_max = quant_scalar_nearest(torch.Tensor([distr.range_max]), grid)

        term_point_mass = (q_range_min - distr.range_min) ** 2 * distr.point_mass_range_min + (
            q_range_max - distr.range_max
        ) ** 2 * distr.point_mass_range_max
        res += term_point_mass

    return res
