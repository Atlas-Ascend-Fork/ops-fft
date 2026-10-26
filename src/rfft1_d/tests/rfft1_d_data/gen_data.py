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
    {"name": "r2c_basic", "n": 8, "batch": 1},
    {"name": "r2c_n16", "n": 16, "batch": 1},
    {"name": "r2c_n1024", "n": 1024, "batch": 1},
    {"name": "r2c_n2048", "n": 2048, "batch": 1},
    {"name": "r2c_n4096", "n": 4096, "batch": 1},
    {"name": "r2c_n2100", "n": 2100, "batch": 1},
    {"name": "r2c_n5040", "n": 5040, "batch": 1},
    {"name": "r2c_n25200", "n": 25200, "batch": 1},
    {"name": "r2c_fft_n1536", "n": 1536, "batch": 4},
    {"name": "r2c_fft_n8192", "n": 8192, "batch": 4},
    {"name": "r2c_fft_n2048_b100", "n": 2048, "batch": 100},

    # added
    {"name": "r2c_fft1_1", "n": 2, "batch": 3},
    {"name": "r2c_fft1_2", "n": 4, "batch": 16384},
    {"name": "r2c_fft1_3", "n": 8, "batch": 1},
    {"name": "r2c_fft1_4", "n": 128, "batch": 16384},

    {"name": "r2c_fft2_1", "n": 256, "batch": 3},
    {"name": "r2c_fft2_2", "n": 512, "batch": 16384},
    {"name": "r2c_fft2_3", "n": 4096, "batch": 1},
    {"name": "r2c_fft2_4", "n": 16384, "batch": 16384},

    {"name": "r2c_fft3_1_1", "n": 32768, "batch": 3},
    {"name": "r2c_fft3_1_2", "n": 32768, "batch": 4096},
    {"name": "r2c_fft3_1_3", "n": 65536, "batch": 1},
    {"name": "r2c_fft3_1_4", "n": 65536, "batch": 4096},

    {"name": "r2c_fft3_2_1", "n": 131072, "batch": 3},
    {"name": "r2c_fft3_2_2", "n": 262144, "batch": 2048},
    {"name": "r2c_fft3_2_3", "n": 1048576, "batch": 1},
    {"name": "r2c_fft3_2_4", "n": 2097152, "batch": 64},

    {"name": "r2c_fft4_1", "n": 4194304, "batch": 3},
    {"name": "r2c_fft4_2", "n": 8388608, "batch": 16},
    {"name": "r2c_fft4_3", "n": 16777216, "batch": 1},
    {"name": "r2c_fft4_4", "n": 33554432, "batch": 8},

    {"name": "r2c_fft5_1", "n": 67108864, "batch": 4},
    {"name": "r2c_fft5_2", "n": 134217728, "batch": 2},
    {"name": "r2c_fft5_3", "n": 268435456, "batch": 1},
    {"name": "r2c_fft5_4", "n": 536870912, "batch": 1},
]


def gen_one(cfg, seed):
    name = cfg["name"]
    n = cfg["n"]
    batch = cfg["batch"]
    total_input = n * batch
    torch.manual_seed(seed)
    input_real = torch.randn(total_input).float().reshape(batch, n)
    golden = torch.fft.rfft(input_real, dim=1)
    output_size = n // 2 + 1
    total_output = output_size * batch
    golden_flat = np.empty(total_output * 2, dtype=np.float32)
    golden_flat[0::2] = golden.reshape(-1).real.numpy()
    golden_flat[1::2] = golden.reshape(-1).imag.numpy()
    input_real.numpy().tofile(f"{name}_input.bin")
    golden_flat.tofile(f"{name}_golden.bin")
    logger.info(f"  Generated: {name}_input.bin ({total_input} floats), "
                f"{name}_golden.bin ({total_output * 2} floats)")

if __name__ == "__main__":
    sys.exit(test_helper.gen_main(configs, gen_one))
