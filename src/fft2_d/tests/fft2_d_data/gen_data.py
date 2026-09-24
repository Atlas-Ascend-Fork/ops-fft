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

import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'common', 'host'))
import test_helper

logger = logging.getLogger(__name__)

configs = [
    {"name": "dd_32x32_forward", "fft_x": 32, "fft_y": 32, "batch": 1, "forward": True},
    {"name": "dd_32x64_forward", "fft_x": 32, "fft_y": 64, "batch": 1, "forward": True},
    {"name": "dd_64x64_forward", "fft_x": 64, "fft_y": 64, "batch": 1, "forward": True},
    {"name": "dd_64x128_forward", "fft_x": 64, "fft_y": 128, "batch": 1, "forward": True},
    {"name": "dd_128x128_forward", "fft_x": 128, "fft_y": 128, "batch": 1, "forward": True},
    {"name": "dd_32x32_backward", "fft_x": 32, "fft_y": 32, "batch": 1, "forward": False},
    {"name": "dd_64x64_backward", "fft_x": 64, "fft_y": 64, "batch": 1, "forward": False},
    {"name": "dd_128x128_backward", "fft_x": 128, "fft_y": 128, "batch": 1, "forward": False},
    # 多 batch 用例（issue #125）：覆盖 CalcBatchNumsPerLoop 全部 6 种
    # batchNumsPerLoop>1 的形状组合，守护 SeparateRITWOMultiBatch /
    # MatmulWithSpecialTransfer 等仅 batch>1 才执行的内核路径
    {"name": "dd_128x32_forward_b2", "fft_x": 128, "fft_y": 32, "batch": 2, "forward": True},
    {"name": "dd_64x64_forward_b2", "fft_x": 64, "fft_y": 64, "batch": 2, "forward": True},
    {"name": "dd_64x32_forward_b4", "fft_x": 64, "fft_y": 32, "batch": 4, "forward": True},
    {"name": "dd_32x128_forward_b2", "fft_x": 32, "fft_y": 128, "batch": 2, "forward": True},
    {"name": "dd_32x64_forward_b4", "fft_x": 32, "fft_y": 64, "batch": 4, "forward": True},
    {"name": "dd_32x32_forward_b8", "fft_x": 32, "fft_y": 32, "batch": 8, "forward": True},
    {"name": "dd_64x64_backward_b2", "fft_x": 64, "fft_y": 64, "batch": 2, "forward": False},
]


def gen_one(cfg, seed):
    name = cfg["name"]
    fft_x = cfg["fft_x"]
    fft_y = cfg["fft_y"]
    batch = cfg["batch"]
    total = fft_x * fft_y * batch
    torch.manual_seed(seed)
    input_complex = torch.randn(batch, fft_x, fft_y, dtype=torch.complex64)
    input_flat = np.empty(total * 2, dtype=np.float32)
    input_flat[0::2] = input_complex.reshape(-1).real.numpy()
    input_flat[1::2] = input_complex.reshape(-1).imag.numpy()
    if cfg["forward"]:
        golden = torch.fft.fft2(input_complex, dim=(1, 2))
    else:
        golden = torch.fft.ifft2(input_complex, dim=(1, 2)) * (fft_x * fft_y)
    golden_flat = np.empty(total * 2, dtype=np.float32)
    golden_flat[0::2] = golden.reshape(-1).real.numpy()
    golden_flat[1::2] = golden.reshape(-1).imag.numpy()
    input_flat.tofile(f"{name}_input.bin")
    golden_flat.tofile(f"{name}_golden.bin")
    logger.info(f"  Generated: {name}_input.bin ({len(input_flat)} floats), "
                f"{name}_golden.bin ({len(golden_flat)} floats)")

if __name__ == "__main__":
    sys.exit(test_helper.gen_main(configs, gen_one))
