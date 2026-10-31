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

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'common', 'host'))
import test_helper

configs = {
    "c2r_basic": {"atol": 1e-3, "rtol": 1e-3},
    "c2r_n16": {"atol": 1e-3, "rtol": 1e-3},
    # issue #227: odd-n coverage on the DFT path
    "c2r_dft_n7": {"atol": 1e-3, "rtol": 1e-3},
    "c2r_dft_n15": {"atol": 1e-3, "rtol": 1e-3},
    "c2r_dft_n255": {"atol": 1e-2, "rtol": 1e-3},
    "c2r_n2100": {"atol": 1.0, "rtol": 1e-2},
    "c2r_n5040": {"atol": 5.0, "rtol": 1e-2},
    "c2r_fft_n1536": {"atol": 5e-2, "rtol": 1e-2},
    "c2r_fft_n8192": {"atol": 5e-2, "rtol": 1e-2},
    "c2r_fft_n8192_nyquist": {"atol": 5e-2, "rtol": 1e-2},
    "c2r_fft_n2048_b100": {"atol": 5e-2, "rtol": 1e-2},
    "c2r_fft_odd_n1025": {"atol": 5e-2, "rtol": 1e-2},
}

curr_dir = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    sys.exit(test_helper.compare_main(curr_dir, configs))
