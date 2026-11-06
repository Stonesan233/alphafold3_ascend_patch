#!/usr/bin/env bash
# 下载并准备 AlphaFold3 C++ 扩展的构建期依赖（7 个源码包）。
#
# 产物：<dest>/<name>/  ——可直接交给 alphafold3 顶层的 SOURCE_DIR 逻辑
#      （打补丁后 export ALPHAFOLD3_DEPS_DIR=<dest> 即可离线编译）。
#
# 用法：
#   ./download_deps.sh                     # 下载到 ../deps/src
#   ./download_deps.sh --dest /data/deps   # 指定产物目录
#   ./download_deps.sh --list              # 只打印下载链接与 SHA256
#   ./download_deps.sh --no-patch          # 只解压不裁剪、不打补丁（排查用）
#
# 环境变量：
#   http_proxy / https_proxy   走代理下载（curl 原生支持）
#   CURL_OPTS                  追加给 curl 的参数，如 "-m 120 --retry 5"
#   DEPS_CACHE                 下载缓存目录，默认 ../deps/.cache
#
# 依赖：bash 4+、curl、tar、sha256sum、patch
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PATCH_DIR="${REPO_ROOT}/patches"
DEST="${REPO_ROOT}/deps/src"
CACHE="${DEPS_CACHE:-${REPO_ROOT}/deps/.cache}"
APPLY_PATCH=1
LIST_ONLY=0
CURL_OPTS="${CURL_OPTS:--L --fail --retry 3 --retry-delay 2 --connect-timeout 20}"

# name|url|sha256|prune（空格分隔的子目录）
SOURCES=(
  "abseil-cpp|https://github.com/abseil/abseil-cpp/archive/d7aaad83b488fd62bd51c81ecf16cd938532cc0a.tar.gz|3a1f642ea21f45b220a01708d06b46042a3b750f31772b52371d6268f02d5bbc|"
  "pybind11|https://github.com/pybind/pybind11/archive/2e0815278cb899b20870a67ca8205996ef47e70f.tar.gz|a50dfa85eda59b4c95180dd06baec89437ee1bc28ba92c5d9d1f853738ddd490|"
  "pybind11_abseil|https://github.com/pybind/pybind11_abseil/archive/bddf30141f9fec8e577f515313caec45f559d319.tar.gz|a92e5faba0397eb4a6ba79317ab57de7891dc69ca8848cf2326abe28d5179c8f|"
  "libcifpp|https://github.com/pdb-redo/libcifpp/archive/ac98531a2fc8daf21131faa0c3d73766efa46180.tar.gz|5fa208efa5aa7890b2a31ac1743ffb4dfc8bdc2605f590cc4639cc8922b18890|examples test"
  "dssp|https://github.com/PDB-REDO/dssp/archive/57560472b4260dc41f457706bc45fc6ef0bc0f10.tar.gz|6cf3068527117e9a613728633a62f489af8890d5c30e7db919183b470e972258|test"
  "eigen|https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz|8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72|bench blas ci debug demos doc failtest lapack scripts test .gitlab"
  "libmcfp|https://github.com/mhekkel/libmcfp/archive/refs/tags/v1.3.4.tar.gz|79783923527711bc27219592f25b9e92d6e402e15be146eeb6cb2050796ec3b6|"
)

# 依赖源码的本地补丁（相对仓库根的 patches/）
declare -A DEP_PATCH=(
  ["libcifpp"]="deps-libcifpp-boost-eigen.patch"
  ["dssp"]="deps-dssp-mkdssp-offline.patch"
)

usage() { sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)      DEST="$2"; shift 2 ;;
    --cache)     CACHE="$2"; shift 2 ;;
    --no-patch)  APPLY_PATCH=0; shift ;;
    --list)      LIST_ONLY=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

if [[ $LIST_ONLY -eq 1 ]]; then
  printf '%-16s %-92s %s\n' "name" "url" "sha256"
  for entry in "${SOURCES[@]}"; do
    IFS='|' read -r name url sha _ <<< "$entry"
    printf '%-16s %-92s %s\n' "$name" "$url" "$sha"
  done
  exit 0
fi

for bin in curl tar sha256sum patch; do
  if ! command -v "$bin" >/dev/null; then
    case "$bin" in
      patch) echo "缺少命令: patch（apt-get install -y patch / yum install -y patch）" >&2 ;;
      *)     echo "缺少命令: $bin" >&2 ;;
    esac
    exit 1
  fi
done

mkdir -p "$CACHE" "$DEST"

fetch() {  # fetch <url> <out>
  local url="$1" out="$2"
  if [[ -f "$out" ]]; then
    echo "  使用缓存: $out"
    return 0
  fi
  echo "  下载: $url"
  # shellcheck disable=SC2086
  curl $CURL_OPTS -o "${out}.part" "$url"
  mv "${out}.part" "$out"
}

for entry in "${SOURCES[@]}"; do
  IFS='|' read -r name url sha prune <<< "$entry"
  echo "== $name"
  archive="${CACHE}/${name}.tar.gz"
  fetch "$url" "$archive"

  actual=$(sha256sum "$archive" | awk '{print $1}')
  if [[ "$actual" != "$sha" ]]; then
    echo "  SHA256 不匹配：期望 $sha，实际 $actual" >&2
    echo "  删除缓存后重试：rm -f $archive" >&2
    exit 1
  fi
  echo "  SHA256 OK"

  rm -rf "${DEST:?}/${name}"
  mkdir -p "${DEST}/${name}"
  tar -xzf "$archive" -C "${DEST}/${name}" --strip-components=1

  # 上游自带的 .gitignore 会被 git 用来过滤源码（eigen 的 core 规则曾把
  # Eigen/Core 整批滤掉），统一删顶层 .gitignore。
  rm -f "${DEST}/${name}/.gitignore"

  if [[ $APPLY_PATCH -eq 1 && -n "$prune" ]]; then
    for d in $prune; do rm -rf "${DEST}/${name:?}/${d}"; done
    echo "  裁剪: $prune"
  fi

  if [[ $APPLY_PATCH -eq 1 && -n "${DEP_PATCH[$name]:-}" ]]; then
    echo "  打补丁: ${DEP_PATCH[$name]}"
    patch -d "${DEST}/${name}" -p1 --silent < "${PATCH_DIR}/${DEP_PATCH[$name]}"
  fi

  echo "  文件数: $(find "${DEST}/${name}" -type f | wc -l)"
done

echo
echo "完成。依赖目录: $DEST"
echo "后续：export ALPHAFOLD3_DEPS_DIR=$DEST"
