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
    "c2c_forward_n512": {"atol": 1e-2, "rtol": 1e-3},
    "c2c_forward_n420": {"atol": 1e-2, "rtol": 1e-3},
    "c2c_roundtrip_n512": {"atol": 0.512, "rtol": 1e-2},
    "dft_forward": {"atol": 1e-3, "rtol": 1e-3},
    "dft_backward": {"atol": 1e-3, "rtol": 1e-3},
    "dft_config": {"atol": 1e-2, "rtol": 1e-3},
    "stride_forward": {"atol": 1e-2, "rtol": 1e-3},
    "stride_forward_n4096": {"atol": 1e-2, "rtol": 1e-3},
    "stride_forward_bigstride": {"atol": 1e-2, "rtol": 1e-3},  # issue #444 UINT16 fallback branch
    "fft_n_forward": {"atol": 1e-2, "rtol": 1e-3},
    "fft_n_roundtrip": {"atol": 3.2768, "rtol": 1e-2},
    "fft_n_config": {"atol": 1e-2, "rtol": 1e-3},
    "fft_b_n512": {"atol": 1e-2, "rtol": 1e-3},
    "fft_b_n2048": {"atol": 1e-2, "rtol": 1e-3},
    "fft_b_n16384": {"atol": 1e-2, "rtol": 1e-3},
    "mix_fwd_n6": {"atol": 1e-2, "rtol": 1e-3},
    "mix_fwd_n105": {"atol": 1e-2, "rtol": 1e-3},
    "mix_fwd_n210": {"atol": 1e-2, "rtol": 1e-3},
    "mix_bwd_n15": {"atol": 1e-2, "rtol": 1e-3},
    "mix_prime_n11": {"atol": 1e-2, "rtol": 1e-3},
    "mix_primepow_n121": {"atol": 1e-2, "rtol": 1e-3},
    "mix_mixed_n143": {"atol": 1e-2, "rtol": 1e-3},
    "mix_all4_n2310": {"atol": 1e-2, "rtol": 1e-3},
    "mix_vlarge_n46200": {"atol": 1e-2, "rtol": 1e-3},
    "mix_roundtrip_n210": {"atol": 2.1, "rtol": 1e-2},
    "mix_bwd_n210_b4": {"atol": 1e-2, "rtol": 1e-3},
    "mix_fwd_n1536_b4": {"atol": 1e-2, "rtol": 1e-3},
}

curr_dir = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    sys.exit(test_helper.compare_main(curr_dir, configs))
