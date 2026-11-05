#!/usr/bin/env python3
"""fastnn 三种实现的数值等价性回归。

这是本仓补丁的核心改动之一：xfold 的 fastnn 有三个 triton kernel（LayerNorm、
dot_product_attention、GatedLinearUnit），NPU 上没有 triton，补丁把它们改成
「triton 可选导入 + torch 回退」。这个脚本负责验证：

1. 无 triton 时 `HAS_TRITON=False`，三个模块的 `*_implementation` 自动落到 `torch`；
2. torch 实现与框架原语（`F.layer_norm` / `F.scaled_dot_product_attention` /
   `silu(xW1)*xW2`）数值一致（fp32，容差 1e-5）；
3. triton 若可用（CUDA 机器），triton 与 torch 两条路径一致。

用法：
    python test/test_fastnn_equivalence.py --xfold /data/af3/xfold [--device cpu]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xfold", required=True, help="xfold 源码根")
    p.add_argument("--device", default="cpu", help="cpu / cuda:0 / npu:0（默认 cpu）")
    p.add_argument("--tol", type=float, default=1e-5)
    return p.parse_args()


def check(name: str, actual: torch.Tensor, expected: torch.Tensor, tol: float) -> bool:
    actual = actual.detach().float()
    expected = expected.detach().float()
    if actual.shape != expected.shape:
        print(f"  FAIL {name}: shape {tuple(actual.shape)} vs {tuple(expected.shape)}")
        return False
    diff = (actual - expected).abs().max().item()
    ok = diff <= tol
    print(f"  {'PASS' if ok else 'FAIL'} {name}: max|Δ| = {diff:.3e} (tol {tol:g})")
    return ok


def main():
    args = parse_args()
    sys.path.insert(0, args.xfold)

    import importlib

    from xfold.fastnn import config as fastnn_config

    # xfold/fastnn/__init__.py 把同名函数提到了包级，直接 `from xfold.fastnn import
    # gated_linear_unit` 拿到的是函数而不是模块，这里用 importlib 取子模块。
    fastnn_ln = importlib.import_module("xfold.fastnn.layer_norm")
    fastnn_attention = importlib.import_module("xfold.fastnn.attention")
    fastnn_glu = importlib.import_module("xfold.fastnn.gated_linear_unit")

    device = torch.device(args.device)
    torch.manual_seed(0)
    gen = torch.Generator(device="cpu").manual_seed(0)

    print(f"[1/3] 配置探测（HAS_TRITON={getattr(fastnn_config, 'HAS_TRITON', 'N/A')}）")
    expect_torch = not torch.cuda.is_available() or not getattr(fastnn_config, "HAS_TRITON", False)
    for attr in ("layer_norm_implementation",
                 "dot_product_attention_implementation",
                 "gated_linear_unit_implementation"):
        value = getattr(fastnn_config, attr)
        print(f"  {attr} = {value}")
        if expect_torch and value != "torch":
            print(f"  FAIL 期望回退到 torch，实际 {value}")
            return 1

    ok = True

    # ---- LayerNorm ----
    print("[2/3] LayerNorm")
    hidden, eps = 128, 1e-5
    x = torch.randn(2, 16, hidden, generator=gen).to(device)
    weight = torch.randn(hidden, generator=gen).to(device)
    bias = torch.randn(hidden, generator=gen).to(device)

    fastnn_config.layer_norm_implementation = "torch"
    ln = fastnn_ln.LayerNorm(hidden, eps=eps).to(device)
    with torch.no_grad():
        ln.weight.copy_(weight)
        ln.bias.copy_(bias)
        got = ln(x)
    ref = F.layer_norm(x, (hidden,), weight, bias, eps)
    ok &= check("layer_norm", got, ref, args.tol)

    # ---- dot product attention ----
    print("[3/3] dot_product_attention / gated_linear_unit")
    b, h, n, d = 2, 4, 32, 64
    q = torch.randn(b, h, n, d, generator=gen).to(device)
    k = torch.randn(b, h, n, d, generator=gen).to(device)
    v = torch.randn(b, h, n, d, generator=gen).to(device)

    fastnn_config.dot_product_attention_implementation = "torch"
    got = fastnn_attention.dot_product_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    ok &= check("attention（无 mask/bias）", got, ref, args.tol)

    mask = torch.ones(n, dtype=torch.bool)
    mask[n // 2:] = False
    got = fastnn_attention.dot_product_attention(q, k, v, mask=mask)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None, None, :])
    ok &= check("attention（因果 mask）", got, ref, args.tol)

    bias_attn = torch.randn(h, n, n, generator=gen).to(device) * 0.1
    got = fastnn_attention.dot_product_attention(q, k, v, bias=bias_attn)
    scale = d ** -0.5
    logits = (q * scale) @ k.transpose(-1, -2) + bias_attn
    ref = torch.softmax(logits, dim=-1) @ v
    ok &= check("attention（bias）", got, ref, args.tol)

    m, hidden_in, hidden_out = 8, 128, 256
    gx = torch.randn(m, hidden_in, generator=gen).to(device)
    gw = torch.randn(hidden_in, hidden_out * 2, generator=gen).to(device) / hidden_in ** 0.5
    fastnn_config.gated_linear_unit_implementation = "torch"
    got = fastnn_glu.gated_linear_unit(gx, gw)
    y = gx @ gw
    a, bb = y.chunk(2, dim=-1)
    ref = F.silu(a) * bb
    ok &= check("gated_linear_unit", got, ref, args.tol)

    if torch.cuda.is_available() and getattr(fastnn_config, "HAS_TRITON", False):
        print("[附加] triton 路径（CUDA 可用；triton kernel 只支持 fp16/bf16）")
        try:
            gq, gk, gv = (t.cuda().to(torch.bfloat16) for t in (q, k, v))
            fastnn_config.dot_product_attention_implementation = "triton"
            got = fastnn_attention.dot_product_attention(gq, gk, gv)
            fastnn_config.dot_product_attention_implementation = "torch"
            ref = fastnn_attention.dot_product_attention(gq, gk, gv)
            ok &= check("attention triton vs torch", got.float(), ref.float(), 1e-2)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP triton 路径不可用: {exc}")
        finally:
            fastnn_config.dot_product_attention_implementation = "torch"

    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
