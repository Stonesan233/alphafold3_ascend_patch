---
license: apache-2.0
library_name: pytorch
language:
  - zh
hardware: NPU
---
# AlphaFold3-Ascend

AlphaFold 3 在昇腾 NPU 上的推理适配，**以补丁形式叠加在上游代码之上**：本仓不含
Alphafold3 与 xfold 的完整源码，只放补丁、依赖下载脚本、测试与文档，因此可以跟随
上游演进，而不必维护一份分叉副本。

- 上游 1：`google-deepmind/alphafold3` @ `c0f97ed`（v3.0.4 + 16 提交）——数据管线与 C++ 扩展
- 上游 2：`shenggan/xfold` @ `22bdeed`——AlphaFold 3 的 PyTorch 实现，模型前向
- 本仓：`patches/`（2 个主补丁 + 2 个依赖补丁）、`deps/`（下载脚本）、`test/`、`docs/`

# 目录

- [AlphaFold3-Ascend](#alphafold3-ascend)
- [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [代码实现](#代码实现)
- [准备运行环境](#准备运行环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装模型环境](#安装模型环境)
- [获取上游代码并打补丁](#获取上游代码并打补丁)
- [准备模型权重与数据](#准备模型权重与数据)
- [快速开始](#快速开始)
  - [设备检查](#设备检查)
  - [编译 C++ 扩展](#编译-c-扩展)
  - [生成 CCD pickle](#生成-ccd-pickle)
  - [批处理推理](#批处理推理)
  - [精度回归](#精度回归)
- [版本说明](#版本说明)
  - [变更](#变更)
  - [基线升级](#基线升级)
  - [FAQ](#faq)
- [许可](#许可)

# 简介

## 模型介绍

AlphaFold 3 是 DeepMind 的蛋白结构预测模型，输入氨基酸/核酸序列与配体，输出复合物
三维结构与置信度（pLDDT、PAE、PDE）。官方实现基于 JAX；`xfold` 是社区的 PyTorch
重实现，本仓的 NPU 适配即建立在 `xfold` 之上——模型结构、权重与数值语义都不变，
只把设备、精度、算子实现换成昇腾可用的路径。

单次推理（150 tokens，官方配置 10 recycle / 5 sample / 200 diffusion step）在
Ascend 910B4 上约 80 s；冒烟配置（1/1/10）约 1.1 s。

## 代码实现

适配遵循**最小改动**原则，全部改动落在 9 个文件上，不改模型结构与数值逻辑：

| 补丁 | 文件 | 改动 |
| --- | --- | --- |
| `alphafold3-c0f97ed-ascend.patch` | `CMakeLists.txt` | 5 个依赖支持 `ALPHAFOLD3_DEPS_DIR` 离线源码；4 个 `FORCE` 开关关掉构建期联网（Catch2 / CCD 下载）；`Boost::regex` 显式查找（>= 1.74）并链接 |
| 同上 | `pyproject.toml` | `requires-python >= 3.11`（现场 Python 3.11/3.12）；`jax[cuda12]` 从必装改为可选 extra `cuda`，避免拖 3 GB 无用 CUDA 包 |
| `xfold-22bdeed-ascend.patch` | `run_alphafold.py` | 设备抽象（NPU > CUDA > CPU，`AF3_DEVICE` 覆盖）；fp64→fp32（`aclnnMatmul` 不支持 double）；`--precision {auto,fp32,bf16}`；`--num_recycles` / `--diffusion_steps`；迁移到 alphafold3 3.0.4 的 `model.ModelResult` / `InferenceResult` |
| 同上 | `xfold/params.py` | zstd 可选；自动识别 `af3.bin` / `af3.bin.zst`；fourier npy 缺失时给明确报错 |
| 同上 | `xfold/fastnn/{attention,config,gated_linear_unit,layer_norm}.py` | triton 可选导入 + 纯 torch 回退；无 CUDA 时自动选 torch 实现 |
| 同上 | `pyproject.toml` | 声明 `jax` / `dm-haiku` / `tokamax` 运行依赖 |

依赖侧另有两个补丁（详见 [`deps/README.md`](deps/README.md)）：libcifpp 的 Boost 1.74
与 eigen 本地路径、dssp 的 `DSSP_BUILD_MKDSSP` 开关。

补丁的基线 SHA、应用顺序与每处改动的原因见 [`patches/README.md`](patches/README.md)。

# 准备运行环境

## 安装昇腾环境

请参考昇腾社区《[PyTorch 框架环境准备](https://www.hiascend.com/document/detail/zh/Pytorch/latest/installguide/swinstall/docs/zh/installation_guide)》。
本仓已支持表 1 中的软件版本。

**表 1** 昇腾软件版本支持表

|  软件类型  |    首次支持版本     |
| :--------: | :-----------------: |
|  TorchNPU  |    2.10.0.post4     |
|    CANN    |        9.1.0        |

已验证硬件为 Atlas 800I A2（Ascend 910B4，32 GB HBM，单卡），A2 / A3 均可运行。
设备节点编号因机型而异，`npu-smi info` 里可能是 NPU 4、对应 `/dev/davinci4`，以实际为准。

## 安装模型环境

**表 2** 版本支持表

|     三方库      |   支持版本   |
| :-------------: | :----------: |
|     Python      | 3.11 / 3.12  |
|     PyTorch     | 2.10.0+cpu   |
|       jax       |    0.10.2    |
|    dm-haiku     |    0.0.17    |
|     tokamax     |    0.0.12    |
|      rdkit      |   2025.9.4   |

0. 激活 CANN 环境

   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

1. 安装昇腾版 PyTorch 与 TorchNPU

   二者版本必须严格对应，且需 `--no-deps` 安装，避免被 PyPI 的 CPU 版覆盖：

   ```bash
   pip install --no-deps torch-2.10.0+cpu-*.whl torchvision-0.25.0+cpu-*.whl torch_npu-2.10.0.post4-*.whl
   ```

2. 安装业务依赖

   ```bash
   pip install -r requirements.txt
   pip install tokamax==0.0.12     # >=0.1.0 移除了 DotProductAttentionImplementation，会破坏 import
   ```

3. 系统包（编译 C++ 扩展需要）

   ```bash
   apt-get install -y libboost-regex-dev cmake ninja-build python3-dev
   pip install scikit_build_core    # --no-build-isolation 下需手动装 build backend
   ```

# 获取上游代码并打补丁

一条命令重建工作区（推荐）：

```bash
git clone https://gitcode.com/Ascend-SACT/Alphafold3-ascend.git AlphaFold3-ascend
# 镜像：https://github.com/Stonesan233/alphafold3_ascend_patch.git
./AlphaFold3-ascend/scripts/bootstrap_workspace.sh /data/af3 --deps
```

脚本会完成：clone `alphafold3@c0f97ed`、`xfold@22bdeed` → `git apply` 两个补丁 →
`deps/download_deps.sh` 下载 7 个构建期依赖。需要代理时加 `--proxy http://127.0.0.1:7890`，
内网镜像用 `--af3-git` / `--xfold-git` 指定。

手工等价步骤：

```bash
export WS=/data/af3 && mkdir -p $WS && cd $WS
git clone https://github.com/google-deepmind/alphafold3.git
git clone https://github.com/shenggan/xfold.git
git -C alphafold3 checkout c0f97ed
git -C xfold      checkout 22bdeed

git -C alphafold3 apply /path/AlphaFold3-ascend/patches/alphafold3-c0f97ed-ascend.patch
git -C xfold      apply /path/AlphaFold3-ascend/patches/xfold-22bdeed-ascend.patch

AlphaFold3-ascend/deps/download_deps.sh --dest $WS/deps
```

校验：

```bash
./AlphaFold3-ascend/scripts/verify_workspace.sh $WS
```

# 准备模型权重与数据

以下文件不属于代码仓，需单独获取（下以 `/mnt/weight/alphafold3` 为例）：

| 文件 | 来源 | 大小 |
| --- | --- | --- |
| `af3.bin.zst` | https://storage.googleapis.com/alphafold3/af3.bin.zst | 1.02 GB |
| `fourier_weight.npy` / `fourier_bias.npy` | 由 alphafold3-decoded 的 `diff_fourier_weight.pt` / `diff_fourier_bias.pt` 转换 | 1.2 KB ×2 |
| `components.cif` | https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz（解压） | 118 MB → 514 MB |
| 遗传数据库 | https://storage.googleapis.com/alphafold-databases/v3.0/（9 个文件，约 143 GB） | 解压约 630 GB |

权重受 [WEIGHTS_TERMS_OF_USE.md](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md) 约束，
需从 Google 官方渠道获取。数据库仅完整数据管线（序列 → MSA/模板）需要；只跑预特征化
输入的推理可暂不准备。

目录结构：

```text
/mnt/weight/alphafold3/
├── af3.bin.zst
├── fourier_weight.npy
├── fourier_bias.npy
├── components.cif          # 514 MB，非空
└── databases/
    ├── mmcif_files/
    ├── uniref90_2022_05.fa
    ├── bfd-first_non_consensus_sequences.fasta
    ├── mgy_clusters_2022_05.fa
    ├── uniprot_all_2021_04.fa
    ├── nt_rna_*.fasta
    ├── rfam_*.fasta
    ├── rnacentral_*.fasta
    └── pdb_seqres_2022_09_28.fasta
```

# 快速开始

## 设备检查

```bash
python -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
# 预期：True 1
```

## 编译 C++ 扩展

```bash
cd $WS

# CCD 字典必须预放置且非空，空文件会被 libcifpp 删除并尝试重新下载
cp /mnt/weight/alphafold3/components.cif deps/libcifpp/rsrc/components.cif
ls -lh deps/libcifpp/rsrc/components.cif          # 应约 514 MB

export ALPHAFOLD3_DEPS_DIR=$WS/deps
export LIBCIFPP_DATA_DIR=$WS/deps/libcifpp/rsrc

cd alphafold3
pip install -e . --no-build-isolation \
  --config-settings="cmake.args=-DBUILD_TESTING=OFF;-DCIFPP_DOWNLOAD_CCD=OFF"

python -c "import alphafold3.cpp; print('OK', len([x for x in dir(alphafold3.cpp) if not x.startswith('_')]))"
# 预期：OK 13，且无 boost undefined symbol
```

编译约 5~15 分钟（abseil + libcifpp 全量）。`cpp.so` 与容器 Python ABI 绑定，
3.11 与 3.12 的产物不能混用。

```bash
cd $WS/xfold && pip install -e . --no-build-isolation
```

## 生成 CCD pickle

```bash
cd $WS/alphafold3/src
python3 -c "
from alphafold3.constants.converters.ccd_pickle_gen import main
main(['g', '/mnt/weight/alphafold3/components.cif',
      'alphafold3/constants/converters/chemical_components.pickle'])"

ln -sf chemical_components.pickle alphafold3/constants/converters/ccd.pickle
python3 -c "
from alphafold3.constants.converters.chemical_component_sets_gen import main
main(['g', 'alphafold3/constants/converters/chemical_component_sets.pickle'])"
```

## 批处理推理

`run_alphafold.py` 支持单文件（`--json_path`）与整目录（`--input_dir`）两种输入，
输出写到 `--output_dir`，适合离线批量跑：

```bash
cd $WS/xfold
export AF3_DEVICE=npu:0          # 多卡机器上指定卡号
export LIBCIFPP_DATA_DIR=$WS/deps/libcifpp/rsrc

python run_alphafold.py \
  --input_dir /data/af3_inputs \
  --output_dir /data/af3_outputs \
  --model_dir /mnt/weight/alphafold3 \
  --db_dir /mnt/weight/alphafold3/databases \
  --num_recycles 10 --num_diffusion_samples 5 --diffusion_steps 200 \
  --precision fp32
```

冒烟时把参数降到 `--num_recycles 1 --num_diffusion_samples 1 --diffusion_steps 10`。
`--precision fp32` 是昇腾上的推荐值：实测比 bf16 更快（79.8 s vs 97.4 s），
置信度指标也更好（PAE/PDE 改善 20~46%）。

只跑数据管线或只跑推理：

```bash
python run_alphafold.py ... --run_inference=False          # 只做 MSA/模板/特征化
python run_alphafold.py ... --run_data_pipeline=False      # 输入已含 MSA
```

## 精度回归

```bash
cd $WS
python /path/AlphaFold3-ascend/test/test_npu_inference.py \
  --af3 $WS/alphafold3 --xfold $WS/xfold \
  --model-dir /mnt/weight/alphafold3 \
  --precision fp32 --num-recycles 1 --diffusion-steps 10
```

预期：`predicted_lddt_mean ≈ 83`、参数量 368,384,666、1/1/10 配置约 1.1 s。
完整的三方精度对比见 [`docs/Ascend-910B-Perf-Precision-Report.md`](docs/Ascend-910B-Perf-Precision-Report.md)。

# 部署注意事项

以下都是在现场（跳板机 + 8×910B4 容器）实测踩到的问题，按出现频率排序。

1. **传输规范**：优先 `git clone`。若只能打包直传（Windows SFTP / 文件复制），
   落地后务必先规范行尾与执行位，否则 bash 会报
   `set: pipefail: invalid option name`、`$'\r': command not found`：

   ```bash
   find . -type f \( -name '*.sh' -o -name '*.py' -o -name '*.patch' \) \
       -exec sed -i 's/\r//g' {} +
   chmod +x scripts/*.sh deps/*.sh test/*.sh
   ```

   仓库已提供 `.gitattributes` 强制 `*.sh/*.py/*.patch/*.md` 使用 LF。

2. **没有 GitHub 直连**：跳板/沙箱里 `git clone`、`curl github.com` 可能直接 DNS 失败，
   `bootstrap_workspace.sh` 需要带 `--proxy http://<代理>:<端口>`；内网镜像可用
   `--af3-git` / `--xfold-git`。

3. **选卡**：容器若映射了全部 8 张卡，`--device npu:0` 会落到物理 NPU 0，被其他业务
   占满时报 `NPU out of memory`。必须显式指定空闲卡：

   ```bash
   export ASCEND_RT_VISIBLE_DEVICES=5    # 再配合 --device npu:0
   ```

4. **数据管线只需要 HMMER**：AF3 的 MSA 用 `jackhmmer`/`nhmmer`，模板用
   `hmmsearch`/`hmmbuild`，**不需要 hh-suite，也不需要 kalign**。离线环境可源码编译：

   ```bash
   # eddylab.org 下载 hmmer-3.4.tar.gz（约 20MB）
   ./configure --prefix=/opt/hmmer && make -j && make install
   ```

   多层 SSH 场景下 `export PATH=...:$PATH` 会被中间 shell 提前展开，建议给
   `run_alphafold.py` 传 `--jackhmmer_binary_path` 等**绝对路径**。

5. **最小推理管线**（不下载数百 GB 数据库，适合 CI/冒烟）：用上游
   `src/alphafold3/test_data/miniature_databases/`（7 个约 1000 条的微型 FASTA +
   `pdb_mmcif` + `pdb_seqres`）+ HMMER 二进制 + 显式 `--db_dir` 与各 binary flag。
   注意微型库的 MSA 深度没有生产意义。

6. **容器内的构建链**：离线环境需预装（走 wheel 中转）
   `scikit_build_core`、`pybind11`、`cmake>=3.28`、`ninja`、`setuptools`；
   `pip install -e .` 必须加 `--no-build-isolation`，否则 pip 进入隔离环境去连 PyPI。

7. **运行时依赖**：`rdkit` 会把 numpy 拉到 2.x。独立容器无碍；与 vLLM 等混合部署时
   （如 `mistral-common` 要求 `numpy<2.4`）需要钉版本。`zstandard` 是加载
   `af3.bin.zst` 的必需依赖。

8. **两阶段加载约束**：`alphafold3.cpp` 与 `torch_npu` 同进程后再对 pkl 做反序列化会
   segfault，`pkl→npz` 必须在未加载 `torch_npu` 的干净子进程里完成
   （`test/af3_test_utils.pkl_to_npz` 已封装，`test_*.py` 均遵守）。

9. **运行时数据文件不入库**：`components.cif`（514 MB）需自行获取；
   `chemical_components.pickle` 与 `chemical_component_sets.pickle`（约 546 MB）
   由 bootstrap 之后手工生成，`verify_workspace.sh` 会对这三项缺失给出 WARN。

# 版本说明

## 变更

2026.11.6：外网验证反馈修复——`verify_workspace.sh` 在 Python < 3.10 上不再误报
（`match` 语法），并新增 CCD pickle / components.cif 缺失的 WARN；`test_npu_inference.py`
与 `test_precision_3way.py` 修正 `contact_probs` 键名（实际为 `distogram/contact_probs`）；
`download_deps.sh` 显式检查 `patch(1)`；`bootstrap_workspace.sh` 的后续步骤补上生成
CCD pickle；xfold 补丁补 `--max_template_date`（alphafold3 c0f97ed 为必填）、
`--fastnn` 仅在 CUDA 上用 triton、`--input_dir` 的 generator 先物化；新增
`.gitattributes` 与「部署注意事项」。
2026.11.5：仓库改造为补丁仓——移除全量 vendored 源码（`deps/` 43 MB、`alphafold3/`
与 `xfold/` 全树），改为 `patches/` 补丁 + `deps/download_deps.sh` 下载脚本 +
`scripts/bootstrap_workspace.sh` 重建工作区；新增 `test/` 回归套件；README 重构，
长步骤下沉到 `docs/`。
2026.09.07：现场部署指导刷新（A2/A3 均验证）。
2026.08.29：Round-2 现场修复（eigen 补齐、libmcfp 1.3.4、jax cuda extra、offline FORCE）。
2026.08.28：首次发布，基于 alphafold3 `c0f97ed` + xfold `22bdeed` 的昇腾适配。

## 基线升级

补丁绑定上游 `c0f97ed` / `22bdeed`。上游演进后补丁可能无法直接应用，按以下顺序处理：

1. 先确认上游是否已自带等价改动，避免重复引入：

   ```bash
   grep -n "ALPHAFOLD3_DEPS_DIR\|BUILD_TESTING\|CIFPP_CACHE_DIR" alphafold3/CMakeLists.txt
   grep -n "jax\[cuda12\]\|requires-python" alphafold3/pyproject.toml
   grep -n "AF3_DEVICE\|torch_npu" -r xfold --include="*.py"
   ```

   > 截至上游 `v3.0.5`，`BUILD_TESTING OFF`、`CIFPP_CACHE_DIR`、依赖改
   > `URL + SHA256` 源包这三项**已被上游吸收**，升级时应删掉补丁里对应 hunk。

2. 在新基线上三方合并：

   ```bash
   git checkout <新基线>
   git checkout -b npu-adapt
   git apply --3way /path/AlphaFold3-ascend/patches/alphafold3-c0f97ed-ascend.patch
   git apply --3way /path/AlphaFold3-ascend/patches/xfold-22bdeed-ascend.patch
   ```

3. 冲突热点（已实测）：
   - `alphafold3/CMakeLists.txt`：上游 v3.0.5 起依赖声明整段改成 `URL` + `URL_HASH`，需重写离线分支；
   - `alphafold3/pyproject.toml`：上游仍是 `requires-python >=3.12`，且新增 `msgpack`、`rdkit 2026.3.3`；
   - `xfold/run_alphafold.py`：上游到 `550d852` 已 +101/−25，补丁 100% 冲突，需手工 rebase；
   - `xfold/xfold/params.py` 等 6 个文件：仅 license 头漂移，`--3way` 可自动合。

4. 升级后必须重跑 `test/` 全套，并在真机确认精度基线（见测试报告）。

## FAQ

**Q：`import alphafold3.cpp` 报 boost undefined symbol？**
A：容器缺 `libboost-regex-dev`，装上后重编。

**Q：`import alphafold3.cpp` 报 `Could not find the libcifpp components.cif file`？**
A：`LIBCIFPP_DATA_DIR` 没设，指向 `deps/libcifpp/rsrc`。

**Q：CMake configure 卡在 github.com / gitlab.com / wwpdb？**
A：`ALPHAFOLD3_DEPS_DIR` 未设或敲错（常见多敲一个 F：`ALPHAFAFOLD3_DEPS_DIR`），
或 `components.cif` 是 0 字节被删重下。

**Q：编译报 `'unreachable' is not a member of 'std'`？**
A：deps 里的 libmcfp 版本不对，v2.x 需要 C++23；必须用 v1.3.4。

**Q：编译报 `fatal error: Core: No such file`？**
A：eigen 不完整，缺 `Eigen/Core` 入口或 `src/Core` 目录，用 `deps/download_deps.sh` 重下。

**Q：`pip install` 报 `Cannot import 'scikit_build_core.build'`？**
A：`--no-build-isolation` 下 build backend 要手动装：`pip install scikit_build_core`。

**Q：import tokamax 报错？**
A：tokamax 必须钉 `0.0.12`。

**Q：换了容器后 `import alphafold3.cpp` 失败？**
A：`cpp.so` 与 Python 小版本绑定（3.11 ≠ 3.12），换容器要重编。

其他现场问题（容器挂载、pip 源、数据库准备）见
[`docs/Ascend-NPU-Deployment.md`](docs/Ascend-NPU-Deployment.md)。

# 许可

本仓的补丁、脚本与文档遵循 Apache-2.0；alphafold3 与 xfold 源码同样为 Apache-2.0。

**AlphaFold 3 的模型权重与推理输出不在本仓分发**，由 Google 单独授权
（仅限非商业用途，且禁止发布或分享权重；商业使用需自行向 Google 取得授权）。
完整边界见 [NOTICE.md](NOTICE.md)。
