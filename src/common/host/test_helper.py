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
    # 0 字节比对文件（写盘被截断、磁盘满等）在此按明确失败返回（issue #361）：
    # 空张量直接进入 torch.max 会抛 RuntimeError 中断批量比对
    if len(golden_np) == 0 or len(output_np) == 0:
        detail = (f"empty comparison data (golden={len(golden_np)}, output={len(output_np)}, "
                  f"atol={atol}, rtol={rtol})")
        return False, detail
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
    # Registered case names can be prefixes of other cases (for example
    # dd_32x32_forward and dd_32x32_forward_b8). Delete exact owned assets.
    for suffix in ("input", "golden", "golden_forward", "output"):
        path = os.path.join(curr_dir, f"{name}_{suffix}.bin")
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def ordered_tolerance(value):
    tolerance = float(value)
    if math.isnan(tolerance):
        raise argparse.ArgumentTypeError("comparison tolerance must not be NaN")
    return tolerance


def parse_case_filter(raw):
    """Parse a comma-separated --cases value into a name list (None when absent).

    issue #345: C++ 侧传入本架构实际运行的用例集，批量 gen/compare 仅覆盖该集合，
    避免把对端架构专属用例计入失败并空跑其 golden 生成。
    """
    if not raw:
        return None
    names = [item.strip() for item in raw.split(",") if item.strip()]
    return names or None


def compare_main(curr_dir, configs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=ordered_tolerance, default=None)
    parser.add_argument("--rtol", type=ordered_tolerance, default=None)
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--cases", type=str, default=None,
                        help="comma-separated case names to restrict the batch loop (issue #345)")
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
    case_filter = parse_case_filter(args.cases)
    missing = []
    if case_filter is not None:
        missing = [name for name in case_filter if name not in configs]
        for name in missing:
            # C++ 用例集与 compare_data.py 配置不同步属注册缺陷，必须保持失败信号
            logger.info(f"[FAIL] {name}: no config in compare_data.py")
        names = [name for name in case_filter if name in configs]
    else:
        names = list(configs.keys())
    total = passed = failed = 0
    # 缺名用例同时计入总数与失败，保证 恒有 通过+失败=总测试数（issue #369），
    # 避免 通过23/失败2 之类大于总数的失败计数出现在汇总行
    total += len(missing)
    failed += len(missing)
    for name in names:
        cfg = configs[name]
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
    parser.add_argument("--cases", type=str, default=None,
                        help="comma-separated case names to restrict the batch loop (issue #345)")
    args = parser.parse_args()
    if args.case:
        cfg = next((c for c in configs if c["name"] == args.case), None)
        if cfg is None:
            logger.info(f"Unknown case: {args.case}")
            return 1
        cleanup_case(os.getcwd(), cfg["name"])
        with contextlib.redirect_stdout(io.StringIO()):
            gen_one_fn(cfg, args.seed)
    else:
        case_filter = parse_case_filter(args.cases)
        selected = configs
        if case_filter is not None:
            known = {cfg["name"] for cfg in configs}
            missing = [name for name in case_filter if name not in known]
            for name in missing:
                # C++ 用例集与 gen_data.py 配置不同步属注册缺陷，必须保持失败信号
                logger.info(f"Unknown case in --cases: {name}")
            if missing:
                return 1
            selected_set = set(case_filter)
            selected = [cfg for cfg in configs if cfg["name"] in selected_set]
        for cfg in selected:
            cleanup_case(os.getcwd(), cfg["name"])
        logger.info(f"Generating golden data with seed={args.seed}...")
        for cfg in selected:
            gen_one_fn(cfg, args.seed)
        logger.info(f"Done. {len(selected)} test cases generated.")
    return 0
