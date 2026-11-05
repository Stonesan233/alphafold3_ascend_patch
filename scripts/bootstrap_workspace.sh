#!/usr/bin/env bash
# 从上游源码重建可编译的 AlphaFold3-Ascend 工作区。
#
# 用法：
#   ./scripts/bootstrap_workspace.sh <workspace> [--deps] [--proxy URL]
#        [--af3-git URL] [--xfold-git URL] [--full-clone]
#
#   <workspace>/alphafold3   google-deepmind/alphafold3 @ c0f97ed + 补丁
#   <workspace>/xfold        shenggan/xfold @ 22bdeed + 补丁
#   <workspace>/deps         依赖源码（仅 --deps 时下载）
#
# 说明：
#   * 默认用 `git fetch --depth 1 <sha>` 只取基线提交，速度快；失败自动回落到完整 clone。
#   * 内网镜像可传 --af3-git / --xfold-git。
#   * 需要走代理时 --proxy http://127.0.0.1:7890（等价于已经 export https_proxy）。
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
AF3_GIT="https://github.com/google-deepmind/alphafold3.git"
XFOLD_GIT="https://github.com/shenggan/xfold.git"
AF3_SHA="c0f97eda2f1f482fd94d3a38bece18c7069b4a5c"   # 见 patches/README.md
XFOLD_SHA="22bdeedfa309ef4ff6f9199910d8403915de69d6"
WITH_DEPS=0
FULL_CLONE=0

usage() { sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --deps)        WITH_DEPS=1; shift ;;
    --full-clone)  FULL_CLONE=1; shift ;;
    --proxy)       export http_proxy="$2" https_proxy="$2" HTTP_PROXY="$2" HTTPS_PROXY="$2"; shift 2 ;;
    --af3-git)     AF3_GIT="$2"; shift 2 ;;
    --xfold-git)   XFOLD_GIT="$2"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    -*)            echo "未知参数: $1" >&2; exit 2 ;;
    *)             POSITIONAL+=("$1"); shift ;;
  esac
done

WS="${POSITIONAL[0]:-}"
[[ -n "$WS" ]] || { usage; exit 2; }
mkdir -p "$WS"

checkout_repo() {  # checkout_repo <url> <sha> <dir>
  local url="$1" sha="$2" dir="$3"
  if [[ -d "$dir/.git" ]]; then
    echo "[skip] $dir 已存在，直接校验基线"
  elif [[ $FULL_CLONE -eq 0 ]]; then
    echo "[fetch] $url @ $sha -> $dir"
    git init -q "$dir"
    git -C "$dir" remote add origin "$url"
    if ! git -C "$dir" fetch -q --depth 1 origin "$sha"; then
      echo "  浅取失败，回落完整 clone"
      rm -rf "$dir"
    fi
  fi
  if [[ ! -d "$dir/.git" ]]; then
    echo "[clone] $url -> $dir"
    git clone -q "$url" "$dir"
  fi
  # 确保落到目标提交（完整 clone 或浅取后都走这里）
  if ! git -C "$dir" rev-parse --verify -q "$sha^{commit}" >/dev/null; then
    git -C "$dir" fetch -q origin "$sha" || true
  fi
  git -C "$dir" checkout -q "$sha" 2>/dev/null || git -C "$dir" checkout -q FETCH_HEAD
  echo "  HEAD = $(git -C "$dir" rev-parse --short HEAD)"
}

apply_patch() {  # apply_patch <repo-dir> <patch-file>
  local dir="$1" patch="$2"
  if git -C "$dir" apply -R --check "$patch" 2>/dev/null; then
    echo "  [skip] 补丁已应用: $(basename "$patch")"
    return
  fi
  git -C "$dir" apply --check "$patch"
  git -C "$dir" apply "$patch"
  echo "  [ok] $(basename "$patch")"
}

checkout_repo "$AF3_GIT"   "$AF3_SHA"   "$WS/alphafold3"
checkout_repo "$XFOLD_GIT" "$XFOLD_SHA" "$WS/xfold"

echo "[patch] 应用补丁"
apply_patch "$WS/alphafold3" "$REPO_ROOT/patches/alphafold3-c0f97ed-ascend.patch"
apply_patch "$WS/xfold"      "$REPO_ROOT/patches/xfold-22bdeed-ascend.patch"

if [[ $WITH_DEPS -eq 1 ]]; then
  echo "[deps] 下载构建期依赖"
  "$REPO_ROOT/deps/download_deps.sh" --dest "$WS/deps"
else
  echo "[deps] 跳过（加 --deps 或稍后执行 deps/download_deps.sh --dest $WS/deps）"
fi

cat <<EOF

工作区就绪：$WS
  alphafold3  $(git -C "$WS/alphafold3" rev-parse --short HEAD) + 补丁
  xfold       $(git -C "$WS/xfold" rev-parse --short HEAD) + 补丁

后续步骤（详见 README「快速开始」）：
  export ALPHAFOLD3_DEPS_DIR=$WS/deps
  export LIBCIFPP_DATA_DIR=$WS/deps/libcifpp/rsrc
  cd $WS/alphafold3 && pip install -e . --no-build-isolation
  cd $WS/xfold      && pip install -e . --no-build-isolation
EOF
