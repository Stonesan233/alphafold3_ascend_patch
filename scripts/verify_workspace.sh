#!/usr/bin/env bash
# 校验工作区：补丁是否已应用、依赖是否齐、Python 文件是否可编译。
#
# 用法：
#   ./scripts/verify_workspace.sh <workspace>
#   ./scripts/verify_workspace.sh <workspace> --reference <vendored-tree>
#
#   --reference 指向「改造前的全量源码树」（含 alphafold3/ xfold/ deps/），
#   用于逐字节回归：bootstrap 出来的工作区应与它完全一致。
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WS=""
REF=""
FAILED=0

usage() { sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reference) REF="$2"; shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    -*)          echo "未知参数: $1" >&2; exit 2 ;;
    *)           WS="$1"; shift ;;
  esac
done
[[ -n "$WS" ]] || { usage; exit 2; }

ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAILED=1; }
warn() { printf '  \033[33mWARN\033[0m %s\n' "$*"; }

echo "[1/5] 目录结构"
for d in alphafold3 xfold; do
  [[ -d "$WS/$d/.git" ]] && ok "$WS/$d" || bad "$WS/$d 不存在或不是 git 仓"
done
[[ -d "$WS/deps/libcifpp" ]] && ok "$WS/deps" || warn "$WS/deps 不存在（未执行 --deps）"

echo "[2/5] 补丁应用状态（反向 apply 可成功 = 已应用）"
declare -A PATCHES=(
  ["alphafold3"]="patches/alphafold3-c0f97ed-ascend.patch"
  ["xfold"]="patches/xfold-22bdeed-ascend.patch"
)
for repo in "${!PATCHES[@]}"; do
  p="$REPO_ROOT/${PATCHES[$repo]}"
  if [[ -d "$WS/$repo/.git" ]]; then
    if git -C "$WS/$repo" apply -R --check "$p" 2>/dev/null; then
      ok "$repo: $(basename "$p")"
    else
      bad "$repo: 补丁未应用或不匹配"
    fi
  fi
done

echo "[3/5] 关键文件"
CHECKS=(
  "$WS/alphafold3/CMakeLists.txt:ALPHAFOLD3_DEPS_DIR"
  "$WS/xfold/run_alphafold.py:AF3_DEVICE"
  "$WS/xfold/xfold/params.py:zstandard"
  "$WS/deps/libcifpp/rsrc/mmcif_pdbx.dic:"
  "$WS/deps/eigen/Eigen/Core:"
  "$WS/deps/libmcfp/include/mcfp/mcfp.hpp:"
)
for item in "${CHECKS[@]}"; do
  f="${item%%:*}"; needle="${item#*:}"
  if [[ ! -f "$f" ]]; then warn "缺少 $f"; continue; fi
  if [[ -z "$needle" || $(grep -c -- "$needle" "$f" 2>/dev/null || true) -gt 0 ]]; then
    ok "$f"
  else
    bad "$f 中未找到 $needle"
  fi
done

echo "[4/5] Python 语法"
if command -v python3 >/dev/null; then
  # 上游源码使用 `match` 语法（需 Python 3.10+），低版本 py_compile 必然 SyntaxError，
  # 不具备判定价值，直接跳过。
  if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    mapfile -t pyfiles < <(cd "$WS" && find xfold -name '*.py' -o -path 'alphafold3/*' -name '*.py' | head -400)
    if ( cd "$WS" && python3 -m py_compile "${pyfiles[@]}" 2>/dev/null ); then
      ok "py_compile ${#pyfiles[@]} 个文件"
    else
      bad "py_compile 失败"
    fi
  else
    warn "python3 < 3.10，跳过语法检查（低版本不认识 match 语法）"
  fi
else
  warn "未找到 python3，跳过语法检查"
fi

echo "[5/5] 运行时数据文件（不入库，缺失只 WARN）"
CONVERTERS="$WS/alphafold3/src/alphafold3/constants/converters"
for f in "$WS/deps/libcifpp/rsrc/components.cif" \
         "$CONVERTERS/chemical_components.pickle" \
         "$CONVERTERS/chemical_component_sets.pickle"; do
  if [[ -s "$f" ]]; then
    ok "$f"
  else
    warn "缺少 $f（按 README「编译 C++ 扩展」「生成 CCD pickle」两节生成）"
  fi
done

if [[ -n "$REF" ]]; then
  echo "[额外] 与参考源码树逐字节比对"
  for d in alphafold3 xfold deps; do
    [[ -d "$REF/$d" ]] || continue
    if diff -r -q --exclude=.git --exclude=.cache --exclude=src \
         --exclude=README.md --exclude=download_deps.sh --exclude=__pycache__ \
         "$REF/$d" "$WS/$d" >/tmp/verify_diff_$d.txt 2>&1; then
      ok "$d 一致"
    else
      bad "$d 存在差异，见 /tmp/verify_diff_$d.txt"
    fi
  done
fi

echo
if [[ $FAILED -eq 0 ]]; then
  echo "校验通过"
else
  echo "校验发现问题（见上）"; exit 1
fi
