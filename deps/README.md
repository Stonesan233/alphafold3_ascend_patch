# 构建期依赖（C++ 扩展）

`alphafold3.cpp` 编译时需要 7 个第三方源码树。**本仓不分发这些源码**，只提供
下载脚本、版本锁定清单和两个必要的本地补丁。

## 一键准备

```bash
./deps/download_deps.sh                 # 默认落到 deps/src/
./deps/download_deps.sh --dest /data/af3-deps

export ALPHAFOLD3_DEPS_DIR=$PWD/deps/src
cd alphafold3 && pip install -e . --no-build-isolation
```

脚本会：下载 → 校验 SHA256 → 解压成短目录名 → 删顶层 `.gitignore` → 裁剪测试/示例
→ 给 libcifpp / dssp 打补丁。重复执行会命中 `deps/.cache` 的缓存。

`deps/src/`、`deps/.cache/` 均已在 `.gitignore` 中，不入库。

## 版本清单

| 名称 | 版本 | 下载链接 | SHA256 |
|---|---|---|---|
| `abseil-cpp` | 20240116.2 (`d7aaad83`) | https://github.com/abseil/abseil-cpp/archive/d7aaad83b488fd62bd51c81ecf16cd938532cc0a.tar.gz | `3a1f642e…d5bbc` |
| `pybind11` | v2.12.0 (`2e08152`) | https://github.com/pybind/pybind11/archive/2e0815278cb899b20870a67ca8205996ef47e70f.tar.gz | `a50dfa85…dd490` |
| `pybind11_abseil` | `bddf301` | https://github.com/pybind/pybind11_abseil/archive/bddf30141f9fec8e577f515313caec45f559d319.tar.gz | `a92e5fab…79c8f` |
| `libcifpp` | v7.0.3 (`ac98531a`) | https://github.com/pdb-redo/libcifpp/archive/ac98531a2fc8daf21131faa0c3d73766efa46180.tar.gz | `5fa208ef…b18890` |
| `dssp` | v4.4.7 (`5756047`) | https://github.com/PDB-REDO/dssp/archive/57560472b4260dc41f457706bc45fc6ef0bc0f10.tar.gz | `6cf30685…72258` |
| `eigen` | 3.4.0 | https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz | `8586084f…2c1c72` |
| `libmcfp` | v1.3.4 | https://github.com/mhekkel/libmcfp/archive/refs/tags/v1.3.4.tar.gz | `79783923…6ec3b6` |

完整 SHA256 用 `./deps/download_deps.sh --list` 打印。

> 前 6 个的 SHA256 与上游 `google-deepmind/alphafold3@v3.0.5` 的 `CMakeLists.txt`
> 中声明的 `URL_HASH` **完全一致**，可以互相校验。
>
> `libmcfp` 用 v1.3.4 而不是上游 v3.0.5 锁的 v1.3.1：二者都是 C++17 兼容版本，
> v1.3.4 是本仓现场验证用过的版本；**不要用 v2.x**，它需要 C++23（`std::unreachable`），
> 现场 g++ 11 编不过。

## 两个本地补丁

| 补丁 | 对象 | 内容 |
|---|---|---|
| `patches/deps-libcifpp-boost-eigen.patch` | libcifpp v7.0.3 | `find_package(Boost 1.80)` → `1.74`；eigen 的 FetchContent（gitlab，离线不可达）改为 `DEPS_DIR` 存在时用本地 `SOURCE_DIR`；`EIGEN_INCLUDE_DIR` 同时含根目录与 `Eigen/`（模块头写的是 `#include "Core"`） |
| `patches/deps-dssp-mkdssp-offline.patch` | dssp v4.4.7 | 新增 `DSSP_BUILD_MKDSSP` 选项（默认 ON，alphafold3 顶层传 OFF）；mkdssp target 与 install 规则包进 if；libmcfp / libcifpp 的 FetchContent 在 `DEPS_DIR` 存在时改本地 `SOURCE_DIR` |

补丁格式为 `patch -p1`（`download_deps.sh` 会自动应用）。手工从官方源码准备时：

```bash
cd libcifpp && patch -p1 < ../patches/deps-libcifpp-boost-eigen.patch
cd dssp     && patch -p1 < ../patches/deps-dssp-mkdssp-offline.patch
```

> `git apply` 也能用，但官方 tar/zip 解压出的目录名带 commit hash，
> 需先改名成 `libcifpp/`、`dssp/`，与上层 `EXISTS "${DEPS_DIR}/libcifpp"` 判断对齐。

## 裁剪清单（与脚本一致）

| 依赖 | 删除 |
|---|---|
| eigen | `bench blas ci debug demos doc failtest lapack scripts test .gitlab` |
| libcifpp | `examples test` |
| dssp | `test` |
| 全部 | 顶层 `.gitignore`（eigen 的 `core` 规则会把 `Eigen/Core`、`Eigen/src/Core/**` 滤出 git） |

裁剪只为减小体积，不影响编译；保留完整源码树同样可以。

## 与依赖相关的系统包

- `libboost-regex-dev`（>= 1.74）：libcifpp 用 `boost::regex`，**不要打进 deps**。
  缺它会得到 `import alphafold3.cpp` 报 `undefined symbol: _ZN5boost16re_detail…`。
- `cmake >= 3.28`、`ninja`、`g++`（C++20）、`python3-dev`。
