# 依赖打包与离线策略

AlphaFold3 的 `cpp.so` 在编译期要拉 7 个第三方源码。这些源码**不入库**，本仓只提供
下载脚本、版本锁定与两个必要补丁。本文记录为什么这么做，以及断网环境怎么落地。

---

## 1. 为什么不分发源码

历史版本曾把 7 个源码树（2747 个文件、约 43 MB）直接提交进仓库，问题有两个：

1. **体积与归属**：第三方源码进仓后既占体积，又让人误以为可以修改；升级上游时
   还要同步第三方目录，diff 噪声极大。
2. **.gitignore 静默丢文件**：eigen 自带的 `.gitignore` 里有一条大小写不敏感的
   `core` 规则，会把 `Eigen/Core`、`Eigen/src/Core/**`（166 个文件）整批过滤掉。
   现场第一次编译就是 `fatal error: Core: No such file`。

现在的做法：脚本下载 → SHA256 校验 → 解压 → 删顶层 `.gitignore` → 裁剪 →
打依赖补丁。**产物可复现**：仓库改造时实测重建结果与原 vendored 树逐字节一致。

---

## 2. 版本锁定

| 名称 | 版本 | 备注 |
| --- | --- | --- |
| abseil-cpp | 20240116.2 `d7aaad83` | 与上游 alphafold3 锁定一致 |
| pybind11 | v2.12.0 `2e08152` | 同上 |
| pybind11_abseil | `bddf301` | 同上 |
| libcifpp | v7.0.3 `ac98531a` | 同上 |
| dssp | v4.4.7 `5756047` | 同上 |
| eigen | 3.4.0 | 同上 |
| libmcfp | **v1.3.4** | 上游 v3.0.5 锁的是 v1.3.1；两者都是 C++17 兼容，本仓用现场验证过的 1.3.4 |

前 6 个的 SHA256 与上游 `alphafold3@v3.0.5` 的 `CMakeLists.txt` 中 `URL_HASH`
完全一致，可以互相校验；libmcfp 的哈希由本仓独立记录。

**不要用 libmcfp 2.x**：它需要 C++23（`std::unreachable`），现场 g++ 11 编不过。

---

## 3. 离线落地

### 3.1 有外网的准备机

```bash
./deps/download_deps.sh --dest /tmp/deps
# 缓存：deps/.cache/{abseil-cpp,pybind11,pybind11_abseil,libcifpp,dssp,eigen,libmcfp}.tar.gz
```

### 3.2 现场机器

```bash
mkdir -p /data/af3/deps/.cache
cp /media/af3-deps/*.tar.gz /data/af3/deps/.cache/
./deps/download_deps.sh --dest /data/af3/deps/src
```

脚本对每个包先查缓存、再校验 SHA256，命中即跳过下载，因此断网也能跑。

### 3.3 内网 HTTP 镜像（可选）

若希望走内网 HTTP 而不是拷贝文件，把 7 个 tar 包放到内网静态目录，然后在
`deps/download_deps.sh` 的 `SOURCES` 里替换 URL 前缀即可（SHA256 不变，仍会被校验）。

---

## 4. 依赖补丁的作用

| 补丁 | 解决什么 |
| --- | --- |
| libcifpp `Boost 1.74` + eigen 本地路径 | 现场只有 Boost 1.74（libcifpp 默认要求 1.80）；eigen 的 FetchContent 走 gitlab，内网不可达 |
| libcifpp `EIGEN_INCLUDE_DIR` 补 `Eigen/` | eigen 模块头写的是 `#include "Core"`，相对 `Eigen/` 解析，只把根目录加进 include path 会找不到 |
| dssp `DSSP_BUILD_MKDSSP` 开关 | dssp 默认编 `mkdssp` 可执行文件并链接 libmcfp，alphafold3 只需要 `dssp::dssp` 库；关掉可省一次 C++23 兼容性风险 |

补丁以 `patch -p1` 格式提供，`download_deps.sh` 自动应用；若要手工从官方 zip 准备，
先按 commit 把目录改名成 `libcifpp/`、`dssp/`（去 hash 后缀），再打补丁。

---

## 5. 裁剪清单

裁剪只为减小体积，不影响编译；保留完整源码树同样可用。

| 依赖 | 删除 |
| --- | --- |
| eigen | `bench blas ci debug demos doc failtest lapack scripts test .gitlab` |
| libcifpp | `examples test` |
| dssp | `test` |
| 全部 | 顶层 `.gitignore` |

注意只删**顶层** `.gitignore`：`docs/_static/.gitignore` 之类的保留，否则会与官方
源码树产生无意义差异。
