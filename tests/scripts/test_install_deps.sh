#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# Source the library functions without running the dependency installer.
source "$repo_root/install_deps.sh"

tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

cat > "$tmp_dir/environment.sh" <<'EOF'
export INSTALL_DEPS_SOURCE_TEST=preserved
EOF
run_command source "$tmp_dir/environment.sh"
[[ "${INSTALL_DEPS_SOURCE_TEST:-}" == "preserved" ]]

run_command printf 'stdout-is-preserved\n' | cat > "$tmp_dir/stdout.txt"
[[ "$(cat "$tmp_dir/stdout.txt")" == "stdout-is-preserved" ]]

set +e
run_command bash -c 'exit 23' 2>"$tmp_dir/error.txt"
status=$?
set -e
[[ "$status" -eq 23 ]]
grep -q 'exit code: 23' "$tmp_dir/error.txt"

echo "install_deps.sh run_command tests passed"
