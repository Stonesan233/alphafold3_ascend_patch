#!/usr/bin/env python3
"""NPU 推理冒烟：加载 af3.bin、跑一次 1/1/10 推理、对齐精度基线。

这是官方 ``run_alphafold_test.py::InferenceTest`` 的 NPU 版：官方用 JAX runner，
这里用 xfold（PyTorch）+ torch_npu，输入仍是官方 ``test_data/featurised_example.pkl``。

用法：
    python test/test_npu_inference.py \
        --af3 /data/af3/alphafold3 --xfold /data/af3/xfold --deps /data/af3/deps \
        --model-dir /mnt/weight/alphafold3 \
        --device npu:0 --precision fp32 \
        --num-recycles 1 --num-samples 1 --diffusion-steps 10

预期（Ascend 910B4，fp32，1/1/10）：
    参数量 368,384,666
    predicted_lddt_mean ≈ 83.6（±0.5）
    耗时 ≈ 1.1 s
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import af3_test_utils as U  # noqa: E402

BASELINE_LDDT = 83.6
BASELINE_PARAMS = 368_384_666


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--af3", required=True, help="alphafold3 源码根（含 src/）")
    p.add_argument("--xfold", required=True, help="xfold 源码根")
    p.add_argument("--deps", default=None, help="deps 目录（定位 libcifpp rsrc）")
    p.add_argument("--model-dir", required=True, help="af3.bin(.zst) + fourier npy 所在目录")
    p.add_argument("--pkl", default=None, help="featurised pkl，默认用上游 test_data")
    p.add_argument("--device", default=None, help="npu:0 / cpu / cuda:0")
    p.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--num-recycles", type=int, default=1)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--diffusion-steps", type=int, default=10)
    p.add_argument("--lddt-tolerance", type=float, default=0.5)
    p.add_argument("--skip-baseline-check", action="store_true")
    p.add_argument("--save-output", default=None, help="把输出张量存成 npz（供精度比对）")
    return p.parse_args()


def main():
    args = parse_args()
    af3 = Path(args.af3)
    U.set_libcifpp_env(args.deps)

    pkl = Path(args.pkl) if args.pkl else af3 / "src/alphafold3/test_data/featurised_example.pkl"
    if not pkl.exists():
        sys.exit(f"找不到测试输入: {pkl}")

    # 第一阶段：干净子进程里把 pkl 转 npz（此时还没 import torch_npu）
    tmpdir = tempfile.mkdtemp(prefix="af3_smoke_")
    npz = Path(tmpdir) / "batch.npz"
    print(f"[1/4] pkl -> npz（干净子进程）: {pkl}")
    U.pkl_to_npz(pkl, npz, af3, args.deps)

    # 第二阶段：加载 torch_npu 与模型
    U.add_paths(af3, args.xfold)
    import torch  # noqa: E402

    device = U.pick_device(args.device)
    print(f"[2/4] 设备: {device}（torch {torch.__version__}）")
    batch = U.npz_to_batch(npz, device)

    print(f"[3/4] 加载模型: {args.model_dir} "
          f"(recycles={args.num_recycles}, samples={args.num_samples}, steps={args.diffusion_steps})")
    model = U.build_model(
        args.xfold,
        args.model_dir,
        device=device,
        num_recycles=args.num_recycles,
        num_samples=args.num_samples,
        diffusion_steps=args.diffusion_steps,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      参数量: {n_params:,}")

    # 第三阶段：推理
    print("[4/4] 推理")
    start = time.perf_counter()
    out = U.run_model(model, batch, precision=args.precision)
    elapsed = time.perf_counter() - start

    lddt = float(out["predicted_lddt"].mean()) if "predicted_lddt" in out else float("nan")
    # 模型里 contact_probs 在 distogram 子字典下，run_model 展平后键名带前缀；
    # 兼容两种命名，避免打印 nan 让人误判成输出异常。
    contact = float("nan")
    for key in ("distogram/contact_probs", "contact_probs"):
        if key in out:
            contact = float(out[key].mean())
            break
    print(f"      predicted_lddt_mean = {lddt:.3f}")
    print(f"      contact_probs_mean   = {contact:.6f}")
    print(f"      耗时                 = {elapsed:.2f} s")
    print("      输出张量:")
    for key in sorted(out):
        print(f"        {key:24s} {tuple(out[key].shape)}")

    if args.save_output:
        np.savez(args.save_output, **{k: v.numpy() for k, v in out.items()})
        print(f"      已保存: {args.save_output}")

    # 断言
    finite = all(bool(torch.isfinite(v).all()) for v in out.values())
    ok = True
    if not finite:
        print("FAIL: 输出含 NaN/Inf")
        ok = False
    if "predicted_lddt" not in out:
        print("FAIL: 输出缺少 predicted_lddt")
        ok = False
    if not args.skip_baseline_check:
        if n_params != BASELINE_PARAMS:
            print(f"FAIL: 参数量 {n_params:,} != 基线 {BASELINE_PARAMS:,}")
            ok = False
        if abs(lddt - BASELINE_LDDT) > args.lddt_tolerance:
            print(f"FAIL: predicted_lddt_mean {lddt:.3f} 偏离基线 {BASELINE_LDDT}±{args.lddt_tolerance}")
            ok = False
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
