# 测试与回归

对齐上游官方的测试方式：**数据管线 / 构建**用官方测试（CPU，无需权重），
**模型推理**用官方 `run_alphafold_test.py::InferenceTest` 的 NPU 版
（xfold + torch_npu + 官方参考输出）。本目录不放 HTTP 测试——实际用法是
`run_alphafold.py` 的离线批处理。

| 脚本 | 覆盖什么 | 需要设备 | 是否需要权重 |
| --- | --- | --- | --- |
| `run_official_cpu_tests.sh` | 上游官方 CI 内容：`build_data` + 数据管线 + 输入校验 | CPU | 否 |
| `test_fastnn_equivalence.py` | 补丁核心：fastnn 的 triton 可选 / torch 回退，三条路径数值一致 | CPU（可选 CUDA） | 否 |
| `test_npu_inference.py` | 端到端推理冒烟：参数量、pLDDT、耗时 | NPU | 是 |
| `test_precision_3way.py` | 三方精度：NPU vs CPU vs JAX 参考；可选结构级 Kabsch RMSD | NPU + CPU | 是 |
| `run_batch_inference.sh` | 离线批处理：`--input_dir` 整目录推理 + 产物校验 | NPU | 是 |

## 前置

1. 工作区已按 `scripts/bootstrap_workspace.sh` 重建，`alphafold3` 与 `xfold` 均已
   `pip install -e`，`LIBCIFPP_DATA_DIR` 指向 `deps/libcifpp/rsrc`；
2. 运行官方 CPU 测试还需要 `hmmer`（`apt-get install -y hmmer`）；
3. 推理类脚本需要 `af3.bin(.zst)` + 两个 fourier npy。

## 一次性全跑

```bash
export WS=/data/af3

# 1. 不需要权重的部分
./test/run_official_cpu_tests.sh $WS
python test/test_fastnn_equivalence.py --xfold $WS/xfold

# 2. 需要权重的部分
python test/test_npu_inference.py --af3 $WS/alphafold3 --xfold $WS/xfold \
    --deps $WS/deps --model-dir /mnt/weight/alphafold3 --device npu:0 \
    --num-recycles 1 --num-diffusion-steps 10

python test/test_precision_3way.py --af3 $WS/alphafold3 --xfold $WS/xfold \
    --deps $WS/deps --model-dir /mnt/weight/alphafold3 --device npu:0 --cpu \
    --reference $WS/alphafold3/src/alphafold3/test_data/alphafold_run_outputs/run_alphafold_test_output_bucket_default.pkl

./test/run_batch_inference.sh --workspace $WS \
    --input /data/af3_inputs --output /data/af3_outputs \
    --model-dir /mnt/weight/alphafold3 --db-dir /mnt/weight/alphafold3/databases
```

## 回归基线（Ascend 910B4，fp32）

| 指标 | 基线 | 脚本断言 |
| --- | --- | --- |
| 参数量 | 368,384,666 | 严格相等 |
| `predicted_lddt_mean` | 83.6 | ±0.5 |
| contact_probs 均值差（vs JAX 参考） | ~0.0010 | 人工核对 |
| 冒烟耗时（1 recycle / 1 sample / 10 step） | ~1.1 s | 人工核对 |
| `import alphafold3.cpp` | 13 个子模块 | `run_official_cpu_tests.sh` 会先 `build_data` |

详细数据见 [`docs/Ascend-910B-Perf-Precision-Report.md`](../docs/Ascend-910B-Perf-Precision-Report.md)。

## 为什么不用 HTTP 服务测试

AlphaFold 3 是固定输入→固定输出的批处理模型，没有 KV Cache / 连续批处理可调度，
线上用法是 `run_alphafold.py --input_dir` 批量跑。之前仓库里的 FastAPI 服务已移除，
对应的 HTTP 冒烟测试一并删除；服务端那套「pkl 先转 npz 再 import torch_npu」的
两阶段布局仍保留在 `af3_test_utils.pkl_to_npz()` 里——**因为 `torch_npu` 与
`alphafold3.cpp` 同进程做 pickle 反序列化会 segfault**，所有测试脚本都必须遵守。
