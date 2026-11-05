#!/usr/bin/env bash
# 跑上游官方 CPU 测试（与官方 CI 同一套，不需要权重 / GPU / NPU）。
#
# 官方 CI（alphafold3 @ c0f97ed）做的事：
#   uv run build_data
#   uv run python run_alphafold_data_test.py
#   uv run python src/alphafold3/common/folding_input_test.py
#
# 用法：
#   ./test/run_official_cpu_tests.sh <workspace>
#   AF3_SRC=/data/af3/alphafold3 ./test/run_official_cpu_tests.sh
#
# 前置：hmmer（jackhmmer/nhmmer/hmmsearch/hmmbuild/hmmalign）、
#      alphafold3 已 `pip install -e`、LIBCIFPP_DATA_DIR 指向 deps/libcifpp/rsrc。
set -uo pipefail

WS="${1:-${WS:-}}"
AF3_SRC="${AF3_SRC:-${WS:+$WS/alphafold3}}"
if [[ -z "$AF3_SRC" || ! -d "$AF3_SRC" ]]; then
  echo "用法: $0 <workspace>  或  AF3_SRC=<alphafold3 源码根> $0" >&2
  exit 2
fi
[[ -d "$AF3_SRC/src/alphafold3" ]] || { echo "不是 alphafold3 源码根: $AF3_SRC" >&2; exit 2; }

FAILED=0
step() { printf '\n=== %s ===\n' "$*"; }

step "环境检查"
for bin in jackhmmer nhmmer hmmsearch hmmbuild hmmalign; do
  if command -v "$bin" >/dev/null; then echo "  OK   $bin -> $(command -v "$bin")"
  else echo "  FAIL $bin 不存在（apt-get install -y hmmer）"; FAILED=1; fi
done
if [[ -z "${LIBCIFPP_DATA_DIR:-}" ]]; then
  echo "  WARN LIBCIFPP_DATA_DIR 未设置，cpp.so 可能找不到 components.cif"
fi

step "build_data（生成 CCD pickle 与 cpp 扩展入口）"
if command -v build_data >/dev/null; then
  ( cd "$AF3_SRC" && build_data ) || FAILED=1
else
  ( cd "$AF3_SRC" && python -m alphafold3.build_data ) || FAILED=1
fi

step "run_alphafold_data_test.py（数据管线：MSA / 模板 / 特征化）"
( cd "$AF3_SRC" && python run_alphafold_data_test.py ) || FAILED=1

step "src/alphafold3/common/folding_input_test.py（输入校验）"
( cd "$AF3_SRC" && python src/alphafold3/common/folding_input_test.py ) || FAILED=1

# 上游 v3.0.5 起测试文件扩到 56 个、CI 改用 pytest；基线 c0f97ed 只有上面两个。
N_TESTS=$(find "$AF3_SRC" -name '*_test.py' | wc -l)
echo "上游测试文件数: $N_TESTS"
if [[ $N_TESTS -gt 10 ]]; then
  step "pytest（上游测试文件 > 10 个时启用，v3.0.5+）"
  ( cd "$AF3_SRC" && python -m pytest -q --ignore=run_alphafold_test.py ) || FAILED=1
else
  echo
  echo "（基线 c0f97ed 只有 2 个测试文件，跳过 pytest 全量）"
fi

printf '\n'
if [[ $FAILED -eq 0 ]]; then echo "官方 CPU 测试全部通过"; else echo "有测试失败（见上）"; exit 1; fi
