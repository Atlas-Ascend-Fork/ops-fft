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
    "dd_32x32_forward": {"atol": 1e-2, "rtol": 1e-3},
    "dd_32x64_forward": {"atol": 1e-2, "rtol": 1e-3},
    "dd_64x64_forward": {"atol": 1e-2, "rtol": 1e-3},
    "dd_64x128_forward": {"atol": 1e-2, "rtol": 1e-3},
    "dd_128x128_forward": {"atol": 1e-2, "rtol": 1e-3},
    "dd_32x32_backward": {"atol": 1.0, "rtol": 1e-2},
    "dd_64x64_backward": {"atol": 1.0, "rtol": 1e-2},
    "dd_128x128_backward": {"atol": 1.0, "rtol": 1e-2},
    "dd_128x32_forward_b2": {"atol": 1e-2, "rtol": 1e-3},
    "dd_64x64_forward_b2": {"atol": 1e-2, "rtol": 1e-3},
    "dd_64x32_forward_b4": {"atol": 1e-2, "rtol": 1e-3},
    "dd_32x128_forward_b2": {"atol": 1e-2, "rtol": 1e-3},
    "dd_32x64_forward_b4": {"atol": 1e-2, "rtol": 1e-3},
    "dd_32x32_forward_b8": {"atol": 1e-2, "rtol": 1e-3},
    "dd_64x64_backward_b2": {"atol": 1.0, "rtol": 1e-2},
}

curr_dir = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    sys.exit(test_helper.compare_main(curr_dir, configs))
