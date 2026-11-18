# Copyright (c) 2026 Xztxzt
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Exercise the actual CMake dependency gate with package metadata, without CANN."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("runtime", "opbase", "metadef", "ge-compiler", "bisheng-compiler", "asc-devkit")


def write_sdk(path, version):
    for package in PACKAGES:
        info = path / "share" / "info" / package / "version.info"
        info.parent.mkdir(parents=True)
        info.write_text(f"Version={version}\n", encoding="utf-8")


@pytest.mark.parametrize("cmake_source,selected_version,environment,success", [
    ("selected", "9.0.0", "unset", True),
    ("selected", "9.0.0", "empty", True),
    ("selected", "9.0.0", "matching", True),
    ("selected", "9.0.0", "other", True),
    ("selected", "8.4.0", "other", False),
    ("unset", "9.0.0", "matching", True),
    ("empty", "9.0.0", "matching", True),
    ("unset", "8.4.0", "matching", False),
    ("empty", "8.4.0", "matching", False),
    ("unset", "9.0.0", "unset", False),
    ("empty", "9.0.0", "empty", False),
    ("missing", "8.4.0", "other", False),
])
def test_dependency_check_uses_selected_sdk(tmp_path, cmake_source, selected_version, environment, success):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for the dependency gate regression")
    selected = tmp_path / "selected sdk"
    other = tmp_path / "other sdk"
    write_sdk(selected, selected_version)
    write_sdk(other, "9.0.0" if selected_version == "8.4.0" else "8.4.0")
    script = tmp_path / "check.cmake"
    script.write_text(
        f'include("{ROOT / "cmake/func.cmake"}")\n'
        f'include("{ROOT / "version.cmake"}")\n'
        "check_pkg_build_deps(ops_fft)\n", encoding="utf-8")
    env = os.environ.copy()
    env.pop("ASCEND_HOME_PATH", None)
    if environment != "unset":
        env["ASCEND_HOME_PATH"] = {"empty": "", "matching": str(selected), "other": str(other)}[environment]
    command = [cmake]
    if cmake_source != "unset":
        cmake_paths = {"selected": str(selected), "empty": "", "missing": str(tmp_path / "missing sdk")}
        command.append(f"-DASCEND_HOME_PATH={cmake_paths.get(cmake_source)}")
    result = subprocess.run(
        command + ["-P", str(script)],
        cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    if not success and selected_version == "8.4.0" and cmake_source != "missing":
        assert "runtime version is 8.4.0" in result.stderr
    if cmake_source == "missing":
        assert "missing sdk" in result.stderr
