#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import contextlib
import glob
import io
import logging
import math
import os
import sys

import numpy as np
import torch

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def read_comparison_file(path):
    """Read a complete, finite float32 comparison artifact."""
    dtype = np.dtype(np.float32)
    with open(path, "rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if size % dtype.itemsize:
            raise ValueError(f"{path}: {size} bytes do not form complete float32 values")
        values = np.fromfile(stream, dtype=dtype)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: nonfinite comparison values")
    return values


def comparison_details(output_np, golden_np, atol, rtol):
    output = torch.from_numpy(output_np).to(torch.float64)
    golden = torch.from_numpy(golden_np).to(torch.float64)
    diff = torch.abs(output - golden)
    max_abs_error = torch.max(diff).item()
    golden_norm = torch.norm(golden)
    if golden_norm > 0:
        relative_l2 = (torch.norm(output - golden) / golden_norm).item()
    else:
        relative_l2 = torch.norm(output - golden).item()
    mismatch_count = torch.sum(diff > atol).item()
    passed = True
    if mismatch_count > 0:
        first_idx = torch.where(diff > atol)[0][0].item()
        detail = (f"{int(mismatch_count)}/{len(golden)} exceed atol={atol}, "
                  f"max_abs_err={max_abs_error:.6e}, rel_l2={relative_l2:.6e}, "
                  f"first at idx={first_idx}: "
                  f"output={output[first_idx].item():.6f}, golden={golden[first_idx].item():.6f}")
        passed = False
    elif relative_l2 > rtol:
        detail = f"rel_l2={relative_l2:.6e} > rtol={rtol}, max_abs_err={max_abs_error:.6e}"
        passed = False
    else:
        detail = f"max_abs_err={max_abs_error:.6e}, rel_l2={relative_l2:.6e} (atol={atol}, rtol={rtol})"
    return passed, detail


def compare_one(curr_dir, name, atol, rtol, detail_only=False):
    golden_file = os.path.join(curr_dir, f"{name}_golden.bin")
    output_file = os.path.join(curr_dir, f"{name}_output.bin")
    if not os.path.exists(golden_file):
        # golden 缺失按失败计（issue #212）：原先批量模式按 SKIP 且返回 True，
        # gen_data.py 从未成功产出数据时所有用例走此分支，批量比对可零校验
        # 报告全部通过；detail_only（单用例显式指定）分支已有独立 FAIL 语义
        if not detail_only:
            logger.info(f"[FAIL] {name}: golden file not found (data generation failed?)")
        return False
    if not os.path.exists(output_file):
        if not detail_only:
            logger.info(f"[FAIL] {name}: output file not found")
        return False
    try:
        golden_np = read_comparison_file(golden_file)
        output_np = read_comparison_file(output_file)
    except (ValueError, OSError) as error:
        if detail_only:
            logger.info(str(error))
        else:
            logger.info(f"[FAIL] {name}: {error}")
        return False
    if len(golden_np) != len(output_np):
        detail = f"size mismatch (golden={len(golden_np)}, output={len(output_np)})"
        if detail_only:
            logger.info(detail)
        else:
            logger.info(f"[FAIL] {name}: {detail}")
        return False
    passed, detail = comparison_details(output_np, golden_np, atol, rtol)
    if detail_only:
        logger.info(detail)
    else:
        tag = "[PASS]" if passed else "[FAIL]"
        logger.info(f"{tag} {name}: {detail}")
    return passed


def cleanup_case(curr_dir, name):
    for f in glob.glob(os.path.join(curr_dir, f"{name}_*.bin")):
        os.remove(f)


def ordered_tolerance(value):
    tolerance = float(value)
    if math.isnan(tolerance):
        raise argparse.ArgumentTypeError("comparison tolerance must not be NaN")
    return tolerance


def compare_main(curr_dir, configs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=ordered_tolerance, default=None)
    parser.add_argument("--rtol", type=ordered_tolerance, default=None)
    parser.add_argument("--case", type=str, default=None)
    args = parser.parse_args()
    if args.case:
        cfg = configs.get(args.case)
        if cfg is None:
            logger.info(f"Unknown case: {args.case}")
            return 1
        atol = args.atol if args.atol is not None else cfg["atol"]
        rtol = args.rtol if args.rtol is not None else cfg["rtol"]
        if not os.path.isfile(os.path.join(curr_dir, f"{args.case}_golden.bin")):
            logger.info(f"[FAIL] {args.case}: explicitly selected golden file not found")
            return 1
        ok = compare_one(curr_dir, args.case, atol, rtol, detail_only=True)
        cleanup_case(curr_dir, args.case)
        return 0 if ok else 1
    total = passed = failed = 0
    for name, cfg in configs.items():
        atol = args.atol if args.atol is not None else cfg["atol"]
        rtol = args.rtol if args.rtol is not None else cfg["rtol"]
        if compare_one(curr_dir, name, atol, rtol):
            passed += 1
        else:
            failed += 1
        total += 1
    logger.info(f"\n总测试数={total}, 通过={passed}, 失败={failed}")
    return 0 if failed == 0 else 1


def gen_main(configs, gen_one_fn):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--case", type=str, default=None)
    args = parser.parse_args()
    if args.case:
        cfg = next((c for c in configs if c["name"] == args.case), None)
        if cfg is None:
            logger.info(f"Unknown case: {args.case}")
            return 1
        with contextlib.redirect_stdout(io.StringIO()):
            gen_one_fn(cfg, args.seed)
    else:
        for cfg in configs:
            cleanup_case(os.getcwd(), cfg["name"])
        logger.info(f"Generating golden data with seed={args.seed}...")
        for cfg in configs:
            gen_one_fn(cfg, args.seed)
        logger.info(f"Done. {len(configs)} test cases generated.")
    return 0
