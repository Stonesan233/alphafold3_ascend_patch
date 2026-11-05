"""AlphaFold3-Ascend 测试公共工具。

约定：所有脚本都在「已打补丁的工作区」下运行，工作区结构为

    <ws>/alphafold3      google-deepmind/alphafold3 @ c0f97ed + 补丁
    <ws>/xfold           shenggan/xfold @ 22bdeed + 补丁
    <ws>/deps            deps/download_deps.sh 的产物

注意：`alphafold3.cpp` 与 `torch_npu` 同进程加载后再做 pickle 反序列化会 segfault，
所以 **pkl 转 npz 必须放在干净子进程里做**（见 `pkl_to_npz`）。
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np


def add_paths(af3_root: str | os.PathLike, xfold_root: str | os.PathLike) -> None:
    """把上游两个源码树加入 sys.path（顺序：xfold 优先于 alphafold3）。"""
    af3 = Path(af3_root)
    xf = Path(xfold_root)
    for p in (str(af3 / "src"), str(af3), str(xf)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def set_libcifpp_env(deps_root: str | os.PathLike | None) -> None:
    if deps_root:
        os.environ.setdefault(
            "LIBCIFPP_DATA_DIR", str(Path(deps_root) / "libcifpp" / "rsrc")
        )


def pkl_to_npz(
    pkl_path: str | os.PathLike,
    npz_path: str | os.PathLike,
    af3_root: str | os.PathLike,
    deps_root: str | os.PathLike | None = None,
    timeout: int = 300,
) -> str:
    """在干净子进程里把 featurised pkl 转成 npz（不加载 torch_npu）。"""
    script = textwrap.dedent(
        f"""
        import sys, pickle
        import numpy as np
        sys.path.insert(0, {str(Path(af3_root) / 'src')!r})
        with open({str(pkl_path)!r}, 'rb') as f:
            batch = pickle.load(f)
        if isinstance(batch, list):
            batch = batch[0]
        arrays = {{}}
        for k, v in batch.items():
            if isinstance(v, np.ndarray) and v.dtype != object:
                arrays[k] = v
        np.savez({str(npz_path)!r}, **arrays)
        print(len(arrays))
        """
    )
    env = dict(os.environ)
    if deps_root:
        env.setdefault("LIBCIFPP_DATA_DIR", str(Path(deps_root) / "libcifpp" / "rsrc"))
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "pkl 反序列化失败。检查：alphafold3 是否已 `pip install -e`、"
            "LIBCIFPP_DATA_DIR 是否指向 deps/libcifpp/rsrc。\n"
            + proc.stderr[-2000:]
        )
    return str(npz_path)


def npz_to_batch(npz_path: str | os.PathLike, device):
    """npz -> 设备上的 tensor 字典，dtype 按 NPU 约束下采样。"""
    import torch

    data = np.load(npz_path, allow_pickle=False)
    batch = {}
    for key in data.files:
        arr = data[key]
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)  # aclnnMatmul 不支持 double
        elif arr.dtype == np.int64:
            arr = arr.astype(np.int32)
        tensor = torch.from_numpy(arr)
        if tensor.dtype in (torch.float64, torch.bfloat16):
            tensor = tensor.float()
        batch[key] = tensor.to(device)
    return batch


def npu_available() -> bool:
    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except Exception:
        return False


def pick_device(preferred: str | None = None):
    """按 npu > cuda > cpu 选设备；preferred 形如 ``npu:0``。"""
    import torch

    if preferred:
        return torch.device(preferred)
    if npu_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_model(
    xfold_root: str | os.PathLike,
    model_dir: str | os.PathLike,
    *,
    device,
    num_recycles: int = 1,
    num_samples: int = 1,
    diffusion_steps: int = 10,
):
    """构造 xfold 的 AlphaFold3 并加载 af3.bin 权重。"""
    from xfold.alphafold3 import AlphaFold3
    from xfold.fastnn import config as fastnn_config
    from xfold.params import import_jax_weights_

    # NPU 无 triton，强制纯 torch 实现（补丁已自动探测，这里显式再钉一次）
    for attr in (
        "layer_norm_implementation",
        "dot_product_attention_implementation",
        "gated_linear_unit_implementation",
    ):
        if hasattr(fastnn_config, attr):
            setattr(fastnn_config, attr, "torch")

    model = AlphaFold3(
        num_recycles=num_recycles,
        num_samples=num_samples,
        diffusion_steps=diffusion_steps,
    )
    import_jax_weights_(model, Path(model_dir))
    model.eval()
    return model.to(device=device)


def run_model(model, batch, *, precision: str = "fp32"):
    """跑一次前向，返回 float32 的 CPU 张量字典。"""
    import torch

    device = next(model.parameters()).device
    use_bf16 = precision == "bf16" and device.type in ("cuda", "npu")
    with torch.inference_mode():
        if use_bf16:
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                result = model(batch)
        else:
            result = model(batch)
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    def to_cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().float().cpu()
        return None

    out = {}
    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            out[key] = to_cpu(value)
        elif isinstance(value, dict):  # 例如 diffusion_samples / distogram
            for sub_key, sub_value in value.items():
                tensor = to_cpu(sub_value)
                if tensor is not None:
                    out[f"{key}/{sub_key}"] = tensor
    return out


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Kabsch 对齐后的 RMSD（消除整体平移/旋转）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape 不一致: {a.shape} vs {b.shape}")
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    aligned = a @ rot.T
    return float(np.sqrt(np.mean(np.sum((aligned - b) ** 2, axis=1))))


def mean_abs_diff(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    d = np.abs(np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64))
    return float(d.mean()), float(d.max())
