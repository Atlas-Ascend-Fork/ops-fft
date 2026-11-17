# Copyright (c) 2026 Xztxzt
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run the real C++ test orchestration with disposable files and no device code."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SUFFIXES = ("input", "golden", "golden_forward", "output")
FIXTURE = r'''
namespace OpsFftTest {
TestStats g_global_stats;
std::vector<std::string> g_failed_cases;
// The fixture supplies only runtime lifecycle and callback outcomes, not FFT execution.
int ACLManager::init(aclrtStream& stream) { stream = nullptr; return 0; }
void ACLManager::finalize(aclrtStream&) {}
}
struct Case { const char* name; };
void run_selected(aclrtStream stream, OpsFftTest::TestStats& stats) {
    OpsFftTest::RunBatchTests(stream, stats, "selected", std::vector<Case>{{"fft_case"}},
        [](aclrtStream, const Case&) { return std::getenv("FIXTURE_RUN_FAIL") == nullptr; });
}
void run_skipped(aclrtStream, OpsFftTest::TestStats&) { std::exit(99); }
struct RegisterFixtures {
    RegisterFixtures() {
        OpsFftTest::TestRegistry::register_test("selected", run_selected);
        OpsFftTest::TestRegistry::register_test("skipped", run_skipped);
    }
} register_fixtures;
'''


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required to execute the C++ orchestration")
    directory = tmp_path_factory.mktemp("cpp_cleanup")
    acl = directory / "acl/acl.h"
    acl.parent.mkdir()
    acl.write_text("#pragma once\nusing aclrtStream = void*;\n", encoding="utf-8")
    source = directory / "runner.cpp"
    source.write_text((ROOT / "tests/all_tests.cpp.in").read_text(encoding="utf-8").replace(
        "@TEST_INCLUDES@", FIXTURE), encoding="utf-8")
    executable = directory / "runner"
    subprocess.run([compiler, "-std=c++17", f"-I{directory}", f"-I{ROOT / 'tests'}",
                    str(source), "-o", str(executable)], check=True, capture_output=True, text=True)
    return executable


@pytest.mark.parametrize("failure", ["none", "callback", "comparison"])
def test_runner_cleans_only_executed_case_files(runner, tmp_path, failure):
    root = tmp_path / "test root's space"
    data = root / "selected/tests/selected_data"
    skipped = root / "skipped/tests/skipped_data"
    data.mkdir(parents=True)
    skipped.mkdir(parents=True)
    # Stand-in scripts avoid FFT generation; the C++ runner and cleanup are unchanged.
    (data / "gen_data.py").write_text("pass\n", encoding="utf-8")
    compare = "raise SystemExit(3)\n" if failure == "comparison" else "print('[PASS] fft_case: fixture')\n"
    (data / "compare_data.py").write_text(compare, encoding="utf-8")
    owned = [data / f"fft_case_{suffix}.bin" for suffix in SUFFIXES]
    for path in owned:
        path.write_bytes(b"disposable owned file")
    preserved = [data / "measurements.bin", data / "fft_case_measurements.bin",
                 data / "fft_case_b8_output.bin", skipped / "skipped_output.bin", root / "measurements.bin"]
    for path in preserved:
        path.write_bytes(b"preserve this fixture")
    env = dict(os.environ, OPS_FFT_TEST_DATA_ROOT=str(root), OPS_FFT_TEST_FILTER="selected")
    env.pop("FIXTURE_RUN_FAIL", None)
    if failure == "callback":
        env["FIXTURE_RUN_FAIL"] = "1"
    result = subprocess.run([str(runner)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == (0 if failure == "none" else 1), result.stdout + result.stderr
    assert "[SKIP] skipped" in result.stdout
    assert all(not path.exists() for path in owned)
    assert all(path.exists() and path.read_bytes() == b"preserve this fixture" for path in preserved)
