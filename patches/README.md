# 补丁清单

本仓的适配全部以补丁形式维护，**不含上游源码**。两个主补丁合计改动 9 个文件
（alphafold3 +103/−34，xfold +251/−34），另有 2 个依赖补丁。

## 基线

| 仓库 | 地址 | 基线 SHA | 说明 |
| --- | --- | --- | --- |
| alphafold3 | https://github.com/google-deepmind/alphafold3 | `c0f97eda2f1f482fd94d3a38bece18c7069b4a5c` | 2026-08-19，`v3.0.4` 之后 16 个提交（"Add validation that templates is a list"） |
| xfold | https://github.com/shenggan/xfold | `22bdeedfa309ef4ff6f9199910d8403915de69d6` | 2026-07-03，PyTorch 重实现 |

补丁对该 SHA 生成，`git apply --check` 零 fuzz。上游已前进（alphafold3 `v3.0.5`、
xfold `550d852`），升级流程见根 README「基线升级」。

## 应用

```bash
git -C alphafold3 checkout c0f97ed
git -C xfold      checkout 22bdeed

git -C alphafold3 apply patches/alphafold3-c0f97ed-ascend.patch
git -C xfold      apply patches/xfold-22bdeed-ascend.patch
```

依赖补丁由 `deps/download_deps.sh` 自动应用，**不需要手工执行**；它们作用于第三方源码，
不属于上面两个仓库。

## 主补丁

### alphafold3-c0f97ed-ascend.patch（2 文件，+103/−34）

| 文件 | 改动 | 原因 |
| --- | --- | --- |
| `CMakeLists.txt` | 5 个 `FetchContent_Declare` 增加 `ALPHAFOLD3_DEPS_DIR` + `SOURCE_DIR` 分支，缺省回落原 GitHub 路径；新增 `CIFPP_DOWNLOAD_CCD` / `CIFPP_INSTALL_UPDATE_SCRIPT` / `BUILD_TESTING` / `DSSP_BUILD_MKDSSP` 四个 `FORCE` 开关；`find_package(Boost 1.74 REQUIRED COMPONENTS regex)` 并链接 `Boost::regex` | 现场断网：libcifpp 测试会 FetchContent Catch2、CCD 会访问 wwpdb、dssp 会编 mkdssp；libcifpp 用 `boost::regex`，不显式链接则 `import alphafold3.cpp` 报 undefined symbol |
| `pyproject.toml` | `requires-python >=3.12` → `>=3.11`；`jax[cuda12]` 改为 optional extra `cuda`，基础依赖退化为 `jax` | 现场镜像为 Python 3.11/3.12；NPU 上 `jax[cuda12]` 会拖约 3 GB 无用 CUDA 包 |

### xfold-22bdeed-ascend.patch（7 文件，+251/−34）

| 文件 | 改动 | 原因 |
| --- | --- | --- |
| `run_alphafold.py` | `get_device()`（NPU > CUDA > CPU，`AF3_DEVICE` 覆盖）；`device_sync()`；featurised 输入 fp64→fp32；`--precision {auto,fp32,bf16}`；`--num_recycles` / `--diffusion_steps`；迁移到 alphafold3 3.0.4 的 `model.ModelResult` / `InferenceResult`；`--max_template_date`（alphafold3 `c0f97ed` 的 `DataPipelineConfig` 把它设为必填 kw-only）；`--fastnn` 仅在 CUDA 上用 triton（Ascend triton 能 import 但 kernel 依赖 `grid<65536`）；`--input_dir` 的 generator 先 `list()` 再 `len()` | `aclnnMatmul` 不支持 double；NPU 上 fp32 比 bf16 更快更准；xfold 基线写于 3.0.2 时代，3.0.4 删除了 `components.base_model` 与 `diffusion.model.Diffuser`，不改则 import 即失败 |
| `xfold/params.py` | zstd 可选；自动识别 `af3.bin` / `af3.bin.zst`；fourier npy 缺失时报明确错误 | 现场经常拿不到解压权重；zstandard 不是必修依赖 |
| `xfold/fastnn/attention.py` | triton 可选导入 + torch 回退 | NPU 无 triton |
| `xfold/fastnn/config.py` | 新增 `HAS_TRITON` 探测，无 CUDA/NPU 时默认 torch | 同上 |
| `xfold/fastnn/gated_linear_unit.py` | 同上 | 同上 |
| `xfold/fastnn/layer_norm.py` | 同上 | 同上 |
| `pyproject.toml` | 声明 `jax` / `dm-haiku` / `tokamax` | `run_alphafold.py` 经 alphafold3 间接依赖；CPU 版足够 |

## 依赖补丁

| 补丁 | 对象 | 内容 |
| --- | --- | --- |
| `deps-libcifpp-boost-eigen.patch` | pdb-redo/libcifpp @ `ac98531a` (v7.0.3) | Boost `1.80` → `1.74`；eigen FetchContent（gitlab，离线不可达）改为本地 `SOURCE_DIR`；`EIGEN_INCLUDE_DIR` 补 `Eigen/`（模块头写的是 `#include "Core"`） |
| `deps-dssp-mkdssp-offline.patch` | PDB-REDO/dssp @ `5756047` (v4.4.7) | 新增 `DSSP_BUILD_MKDSSP` 选项（alphafold3 顶层传 OFF，不编 mkdssp）；libmcfp / libcifpp 改本地 `SOURCE_DIR` |

详细说明见 [`../deps/README.md`](../deps/README.md)。

## 审计提示

- 三个补丁都不含文件模式（权限位）改动；上游 11 个 `.pyi` / `.sh` 的可执行位保持原样。
- `git apply --check` 已在干净基线上验证；`scripts/verify_workspace.sh` 用反向 apply
  判断工作区是否已打补丁。
- 补丁只改"胶水层"（设备、精度、离线构建、可选依赖），不改模型结构、权重映射与
  数值逻辑；精度基线见 `docs/Ascend-910B-Perf-Precision-Report.md`。
