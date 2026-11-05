# 昇腾 NPU 现场部署手册

面向客户现场的完整部署步骤：从裸机/容器准备到离线批处理推理跑通。
仓库的快速上手见根目录 [`README.md`](../README.md)，本文只讲现场会踩的细节。

---

## 1. 版本矩阵

四层必须互相咬合，下表是**已验证通过**的组合：

| 层 | 组件 | 版本 | 说明 |
| --- | --- | --- | --- |
| 硬件 | Ascend 910B4 | 32 GB HBM × 1 | Atlas 800I A2 PCIE 形态；A3 亦可 |
| 驱动 | NPU driver | 25.3.rc1.1 | 与 CANN 9.1.0 实测兼容 |
| 工具链 | CANN toolkit | 9.1.0 | 容器镜像自带时无需另装 |
| 框架 | torch | 2.10.0+cpu | 必须是昇腾适配版 |
| 框架 | torch_npu | 2.10.0.post4 | **必须与 torch 主版本一一对应** |
| 业务 | Python | 3.12 | 镜像 `vllm-ascend:v0.23.0-a3` 自带；3.11 亦可 |
| 业务 | jax / dm-haiku / tokamax | 0.10.2 / 0.0.17 / 0.0.12 | CPU 版即可 |

`torch_npu` 的 wheel 名形如 `torch_npu-2.10.0.post4-cp312-cp312-manylinux_2_28_aarch64.whl`，
其中 `2.10.0` 必须与已装 torch 一致，否则 import 直接失败。版本对应关系以
[昇腾社区 PyTorch 扩展下载页](https://www.hiascend.com/developer/download/community/result?module=pt) 为准。

---

## 2. 容器

```bash
IMAGE=quay.io/ascend/vllm-ascend:v0.23.0-a3
CONTAINER=af3-xfold
WEIGHT_DIR=/mnt/weight/alphafold3

docker run -d --restart unless-stopped --name ${CONTAINER} \
  --network host --shm-size 1g --privileged \
  --device /dev/davinci4 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v ${WEIGHT_DIR}:${WEIGHT_DIR} \
  --entrypoint /bin/bash ${IMAGE} -lc 'sleep infinity'
```

要点：

- `--device /dev/davinci4` 把目标卡透进容器，容器内即 `npu:0`；卡号按 `npu-smi info` 调整。
- 驱动相关挂载（dcmi / npu-smi / driver/lib64）缺一会报 `libascend_hal.so not found`。
- 权重目录整目录挂载，代码、权重、数据库容器内都可见。

验证：

```bash
docker exec -it ${CONTAINER} bash
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
npu-smi info
```

---

## 3. 离线物料清单

| 物料 | 获取方式 | 说明 |
| --- | --- | --- |
| 代码工作区 | `scripts/bootstrap_workspace.sh`（需能访问 GitHub，或用 `--af3-git/--xfold-git` 指内网镜像） | 两个上游仓各 clone 一次即可 |
| 构建期依赖 | `deps/download_deps.sh` | 7 个 tar 包；断网环境可把 tar 包按名字放进 `deps/.cache/` |
| 系统包 | apt 内源 | `libboost-regex-dev cmake ninja-build python3-dev` |
| pip 包 | pip 内源 | 见 `requirements.txt`；另需 `scikit_build_core` |
| 权重 | Google 官方 | `af3.bin.zst` + 两个 fourier npy |
| CCD | wwpdb | `components.cif`（514 MB，**不能是 0 字节**） |
| 数据库 | Google 镜像 | 9 个文件约 143 GB，解压约 630 GB |

deps 离线包准备（在能联网的机器上）：

```bash
./deps/download_deps.sh --dest /tmp/deps   # 产物在 deps/.cache/*.tar.gz
scp deps/.cache/*.tar.gz 现场:/data/af3/deps/.cache/
```

现场脚本会命中缓存、校验 SHA256，不再联网。

---

## 4. 工作区与编译

```bash
export WS=/data/af3
./AlphaFold3-ascend/scripts/bootstrap_workspace.sh $WS --deps --proxy http://127.0.0.1:7890
./AlphaFold3-ascend/scripts/verify_workspace.sh $WS

cp /mnt/weight/alphafold3/components.cif $WS/deps/libcifpp/rsrc/components.cif
export ALPHAFOLD3_DEPS_DIR=$WS/deps
export LIBCIFPP_DATA_DIR=$WS/deps/libcifpp/rsrc

cd $WS/alphafold3
pip install -e . --no-build-isolation \
  --config-settings="cmake.args=-DBUILD_TESTING=OFF;-DCIFPP_DOWNLOAD_CCD=OFF"
python -c "import alphafold3.cpp; print(len([x for x in dir(alphafold3.cpp) if not x.startswith('_')]))"  # 13

cd $WS/xfold && pip install -e . --no-build-isolation
```

`LIBCIFPP_DATA_DIR` 建议写进 `~/.bashrc`——后续生成 CCD pickle 与推理都要用。

---

## 5. 数据库准备

官方脚本（`alphafold3/fetch_databases.sh`）按顺序下载 9 个文件到统一目录，现场经验：

- 总下载量约 143 GB，解压后约 630 GB，先确认磁盘。
- `pdb_2022_09_28_mmcif_files.tar.zst` 解开是 19.6 万个 `.cif`，小文件多，
  建议先解到本地盘再软链。
- 需要 `hmmer`（`jackhmmer` / `nhmmer` / `hmmsearch` / `hmmbuild` / `hmmalign`）
  与 `--db_dir` 指向该目录。

只跑预特征化输入的推理时，可以跳过整个数据库准备。

---

## 6. 批处理运行

```bash
cd $WS/xfold
export AF3_DEVICE=npu:0
python run_alphafold.py \
  --input_dir /data/af3_inputs --output_dir /data/af3_outputs \
  --model_dir /mnt/weight/alphafold3 \
  --db_dir /mnt/weight/alphafold3/databases \
  --num_recycles 10 --num_diffusion_samples 5 --diffusion_steps 200 \
  --precision fp32
```

每个输入产出 `<name>_model.cif`、`<name>_summary_confidences.json`、
`<name>_confidences.json` 等文件。批处理建议：

- 用 `--input_dir` 整目录提交，进程内顺序处理，失败不会中断后续样本；
- 多卡并行就起多个进程、各自 `AF3_DEVICE=npu:N`，注意每张卡只跑一个进程
  （单卡模型单实例，显存约 20 GB）。

---

## 7. 故障排查

| 现象 | 原因与处理 |
| --- | --- |
| `libascend_hal.so not found` | 容器缺驱动挂载，检查 dcmi / npu-smi / driver/lib64 三个 `-v` |
| 装 xfold 报 `fatal: detected dubious ownership` | 容器内 git 属主与宿主机不一致；`git config --global --add safe.directory <仓库路径>`（或直接用 `bootstrap_workspace.sh`，它按 SHA 取，不依赖工作树属主） |
| 装 alphafold3 报 `Cannot import 'scikit_build_core.build'` | `--no-build-isolation` 下 build backend 需手动装 |
| pip 装包巨慢 / 找不到 setuptools | pip 走了外网，检查 `~/.pip/pip.conf` |
| pip 报 `--config-settings` 不存在 | 用了系统旧 pip；改用 `/usr/local/python3*/bin/pip` |
| CMake configure 卡在 github.com / gitlab.com / wwpdb | `ALPHAFOLD3_DEPS_DIR` 未设或敲错，或 `components.cif` 为 0 字节 |
| `'unreachable' is not a member of 'std'` | deps 里 libmcfp 是 2.x（需 C++23），改用 v1.3.4 |
| `fatal error: Core: No such file` | deps/eigen 不完整，用 `download_deps.sh` 重下 |
| `import alphafold3.cpp` 报 boost undefined symbol | 缺 `libboost-regex-dev`，装后重编 |
| `Could not find the libcifpp components.cif file` | `LIBCIFPP_DATA_DIR` 未设 |
| 换容器后 `import alphafold3.cpp` 失败 | `cpp.so` 与 Python 小版本绑定，重编 |
| import 报 tokamax 相关错误 | `pip install tokamax==0.0.12` |
| 运行中 OOM | 降 `--num_diffusion_samples` 或 `--num_recycles`；fp32 权重约 21 GB，32 GB 卡上不要同时跑两个进程 |
