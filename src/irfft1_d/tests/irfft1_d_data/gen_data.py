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
    {"name": "c2r_basic", "n": 8, "batch": 1},
    {"name": "c2r_n16", "n": 16, "batch": 1},
    # issue #227: odd-n coverage on the DFT path
    {"name": "c2r_dft_n7", "n": 7, "batch": 1},
    {"name": "c2r_dft_n15", "n": 15, "batch": 1},
    {"name": "c2r_dft_n255", "n": 255, "batch": 1},
    {"name": "c2r_n2100", "n": 2100, "batch": 1},
    {"name": "c2r_n5040", "n": 5040, "batch": 1},
    {"name": "c2r_fft_n1536", "n": 1536, "batch": 4},
    {"name": "c2r_fft_n8192", "n": 8192, "batch": 4},
    {"name": "c2r_fft_n8192_nyquist", "n": 8192, "batch": 1, "nyquist_only": True},
    {"name": "c2r_fft_n2048_b100", "n": 2048, "batch": 100},
    {"name": "c2r_fft_odd_n1025", "n": 1025, "batch": 4},
]


def gen_one(cfg, seed):
    name = cfg["name"]
    n = cfg["n"]
    batch = cfg["batch"]
    freq_size = n // 2 + 1
    total_input = freq_size * batch
    torch.manual_seed(seed)
    if cfg.get("nyquist_only", False):
        input_complex = torch.zeros(batch, freq_size, dtype=torch.complex64)
        input_complex[:, -1] = 1.0
    else:
        input_complex = torch.randn(batch, freq_size, dtype=torch.complex64)
    input_flat = np.empty(total_input * 2, dtype=np.float32)
    input_flat[0::2] = input_complex.reshape(-1).real.numpy()
    input_flat[1::2] = input_complex.reshape(-1).imag.numpy()
    golden = torch.fft.irfft(input_complex, n=n, dim=1) * n
    golden_flat = np.empty(n * batch, dtype=np.float32)
    golden_flat[:] = golden.reshape(-1).numpy()
    input_flat.tofile(f"{name}_input.bin")
    golden_flat.tofile(f"{name}_golden.bin")
    logger.info(f"  Generated: {name}_input.bin ({len(input_flat)} floats), "
                f"{name}_golden.bin ({len(golden_flat)} floats)")

if __name__ == "__main__":
    sys.exit(test_helper.gen_main(configs, gen_one))
