# NOTICE — AlphaFold3 昇腾适配的许可边界

本仓（`Alphafold3-ascend`）以补丁形式提供 AlphaFold 3 在昇腾 NPU 上的推理适配
（上游补丁、构建期依赖脚本、验证脚本与文档）。
**本仓不包含 AlphaFold 3 模型权重，也不包含任何由权重派生的产物。**

## 1. 模型权重（Model Parameters）

- 许可：[AlphaFold 3 Model Parameters Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)
- 限制：
  - 仅可由**非商业组织**用于**非商业目的**；不得用于任何商业活动，包括代表商业机构开展的研究。
  - **不得发布或分享权重**（`You must not publish or share AlphaFold 3 model parameters`），组织内部共享除外。
  - 商业用途需由使用方自行向 Google 取得授权；本仓仅提供生态支持，不代为申请，也不提供任何授权承诺。
- 因此本仓、魔乐社区及其他任何渠道均不提供权重下载或镜像，README 只给出官方获取入口。

## 2. 推理输出（Output）

- 许可：[AlphaFold 3 Output Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md)
- 可以发布、分享与改编，但需清楚标注所做的修改，且后续使用持续受该条款约束；**不得**用输出训练同类生物分子结构预测模型。

## 3. 代码

| 代码 | 许可 | 说明 |
| --- | --- | --- |
| 本仓的补丁、脚本与文档 | Apache-2.0 | 与上游保持一致的宽松许可 |
| `google-deepmind/alphafold3` | Apache-2.0 | 数据管线与 C++ 扩展，本仓以补丁形式修改 |
| `shenggan/xfold` | Apache-2.0 | PyTorch 模型前向，本仓以补丁形式修改 |

## 4. 构建期第三方依赖

`deps/download_deps.sh` 会下载以下源码，均未修改许可，仅施加构建相关补丁：

| 依赖 | 版本 | 许可 |
| --- | --- | --- |
| abseil-cpp | 20240116.2 | Apache-2.0 |
| pybind11 | v2.12.0 | BSD-3-Clause |
| pybind11_abseil | bddf301 | BSD-3-Clause |
| libcifpp | v7.0.3 | BSD-2-Clause |
| dssp | v4.4.7 | BSD-2-Clause |
| eigen | 3.4.0 | MPL-2.0（部分模块 LGPL） |
| libmcfp | v1.3.4 | BSD-2-Clause |

## 5. 本仓不包含的内容

模型权重（`af3.bin` / `af3.bin.zst`）、CCD 字典（`components.cif`）、遗传数据库、`cpp.so` 等编译产物、
推理输出文件，全部由使用者按各自条款自行获取或生成。

## 6. 免责声明

本仓代码按“现状”提供，不附带任何明示或暗示的担保。使用 AlphaFold 3 权重与输出前，
请自行确认使用场景符合上述条款；因未取得授权而产生的责任由使用方承担。

## 7. 与 LifeScience 仓的关系

同一份适配同时以 `PyTorch/AlphaFold3` 目录维护在
[AI4Science/LifeScience](https://gitcode.com/AI4Science/LifeScience) 仓中，两处 `NOTICE.md` 内容保持一致。
