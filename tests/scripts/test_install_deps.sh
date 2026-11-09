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

# issue #312: parse_gcc_version must tolerate Ubuntu / RHEL / no-suffix vendor
# string formats (the former awk '$4' field-position parse took the date on RHEL
# and an empty string on short outputs). Fake compilers cover the three shapes;
# they are passed by absolute path so the helper's own head/grep pipeline keeps
# the normal PATH (a PATH-prefix function call would leak into the pipeline).
fake_bin="$tmp_dir/fakebin"
mkdir -p "$fake_bin"
printf '#!/usr/bin/env bash\necho "gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"\n' > "$fake_bin/gcc-ubuntu"
printf '#!/usr/bin/env bash\necho "gcc (GCC) 8.3.1 20190507 (Red Hat 8.3.1-4.5)"\n' > "$fake_bin/gcc-rhel"
printf '#!/usr/bin/env bash\nif [ "${1:-}" = "-dumpfullversion" ]; then printf "12.2.0\\n"; else echo "gcc (GCC) 12"; fi\n' > "$fake_bin/gcc-short"
chmod +x "$fake_bin/gcc-ubuntu" "$fake_bin/gcc-rhel" "$fake_bin/gcc-short"

[[ "$(parse_gcc_version "$fake_bin/gcc-ubuntu")" == "11.4.0" ]]
[[ "$(parse_gcc_version "$fake_bin/gcc-rhel")" == "8.3.1" ]]
[[ "$(parse_gcc_version "$fake_bin/gcc-short")" == "12.2.0" ]]
[[ "$(parse_gcc_version "$tmp_dir/definitely-missing-compiler")" == "0.0.0" ]]

echo "install_deps.sh run_command and parse_gcc_version tests passed"
