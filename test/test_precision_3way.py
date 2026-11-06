#!/usr/bin/env python3
"""三方精度对比：NPU vs CPU（同框架同 seed）与 NPU vs JAX 参考输出。

对应 docs/Ascend-910B-Perf-Precision-Report.md 的方法：

    对比 1: NPU  vs JAX/GPU 参考   → 总差异（NPU 计算 + 框架 + RNG）
    对比 2: CPU  vs NPU           → NPU 计算精度（同框架同 seed）
    对比 3: CPU  vs JAX 参考      → 框架/RNG 差异（同 seed）

用法：
    # 只比 NPU vs CPU（不需要参考 pkl）
    python test/test_precision_3way.py --af3 ... --xfold ... --deps ... \
        --model-dir /mnt/weight/alphafold3 --device npu:0 --cpu

    # 再叠加官方 JAX 参考输出
    python test/test_precision_3way.py ... --cpu \
        --reference <af3>/src/alphafold3/test_data/alphafold_run_outputs/\
run_alphafold_test_output_bucket_default.pkl

    # 结构级 RMSD（Kabsch），比较两批 *_model.cif
    python test/test_precision_3way.py ... --compare-cif ours/ ref/

说明：
    * 与参考 pkl 比对的是 token 级数组（contact_probs / full_pae / full_pde）与标量
      元数据，这些与原子布局无关，可直接逐元素比对；
    * 3D 坐标用 --compare-cif 走结构级比对（Kabsch 对齐后 RMSD）。
"""

from __future__ import annotations

import argparse
import pickle
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import af3_test_utils as U  # noqa: E402

# 参考实现里 numerical_data 的 token 级数组
REF_ARRAY_KEYS = ("contact_probs", "full_pae", "full_pde")
# 参考实现里的标量元数据
REF_SCALAR_KEYS = (
    "ptm",
    "iptm",
    "ranking_score",
    "predicted_tm_score",
    "interface_predicted_tm_score",
    "predicted_distance_error",
    "fraction_disordered",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--af3", required=False, help="alphafold3 源码根")
    p.add_argument("--xfold", required=False, help="xfold 源码根")
    p.add_argument("--deps", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--pkl", default=None, help="featurised pkl，默认用上游 test_data")
    p.add_argument("--device", default=None)
    p.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--num-recycles", type=int, default=1)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--diffusion-steps", type=int, default=10)
    p.add_argument("--cpu", action="store_true", help="同时在 CPU 上跑一遍做对比")
    p.add_argument("--reference", default=None, help="官方 JAX 参考输出 pkl")
    p.add_argument("--compare-cif", nargs=2, metavar=("OURS", "REF"), default=None,
                   help="比较两个目录下的 *_model.cif，输出 Kabsch RMSD")
    return p.parse_args()


def run_once(args, device_spec, npz_path, af3):
    import torch  # noqa: F401

    device = U.pick_device(device_spec)
    print(f"  -> 推理设备 {device}")
    batch = U.npz_to_batch(npz_path, device)
    model = U.build_model(
        args.xfold,
        args.model_dir,
        device=device,
        num_recycles=args.num_recycles,
        num_samples=args.num_samples,
        diffusion_steps=args.diffusion_steps,
    )
    out = U.run_model(model, batch, precision=args.precision)
    del model
    return out


def pick(out, name):
    """按展平后的键名取张量：distogram 子字典会变成 "<outer>/<inner>"。"""
    if name in out:
        return out[name]
    for key in out:
        if key.split("/")[-1] == name:
            return out[key]
    return None


def compare_arrays(name, a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        print(f"    {name:22s} 跳过（shape {a.shape} vs {b.shape}）")
        return
    mean, mx = U.mean_abs_diff(a, b)
    print(f"    {name:22s} 均值差 {mean:.6f}   最大差 {mx:.6f}")


def load_reference(path):
    """加载官方参考 pkl：list[{seed, inference_results:[InferenceResult]}]。"""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        data = [data]
    entry = data[0]
    results = entry.get("inference_results", entry) if isinstance(entry, dict) else entry
    return entry.get("seed"), list(results)


def compare_cif(ours_dir, ref_dir):
    """结构级比对：CA / C1' 原子的 Kabsch RMSD。"""
    from alphafold3 import structure as af3_structure

    ours = sorted(Path(ours_dir).glob("*_model.cif"))
    refs = sorted(Path(ref_dir).glob("*_model.cif"))
    if not ours or not refs:
        print("  未找到 *_model.cif，跳过")
        return
    print(f"  比较 {len(ours)} vs {len(refs)} 个结构")

    def coords(cif_path):
        st = af3_structure.from_mmcif(str(cif_path))
        atom = np.asarray(st.atom_positions)
        mask = np.asarray(st.atom_mask).astype(bool)
        # 只取 CA（蛋白）与 C1'（核酸），避免侧链差异放大 RMSD
        names = np.asarray(st.atom_name)
        keep = mask.all(axis=-1) & (names == "CA")
        if keep.sum() < 3:
            keep = mask.all(axis=-1)
        return atom[keep]

    for ref_path in refs:
        ref_xyz = coords(ref_path)
        best = None
        for our_path in ours:
            try:
                rmsd = U.kabsch_rmsd(coords(our_path), ref_xyz)
            except ValueError:
                continue
            if best is None or rmsd < best[0]:
                best = (rmsd, our_path.name)
        if best:
            print(f"    {ref_path.name:40s} <-> {best[1]:40s} RMSD {best[0]:8.2f} Å")


def main():
    args = parse_args()

    if args.compare_cif:
        U.add_paths(args.af3, args.xfold or args.af3)
        compare_cif(*args.compare_cif)
        return 0

    if not (args.af3 and args.xfold and args.model_dir):
        sys.exit("需要 --af3/--xfold/--model-dir（或用 --compare-cif 模式）")

    af3 = Path(args.af3)
    U.set_libcifpp_env(args.deps)
    pkl = Path(args.pkl) if args.pkl else af3 / "src/alphafold3/test_data/featurised_example.pkl"

    tmpdir = tempfile.mkdtemp(prefix="af3_prec_")
    npz = Path(tmpdir) / "batch.npz"
    print(f"[1/3] pkl -> npz: {pkl}")
    U.pkl_to_npz(pkl, npz, af3, args.deps)

    U.add_paths(af3, args.xfold)
    print("[2/3] NPU 推理")
    out_npu = run_once(args, args.device, npz, af3)

    out_cpu = None
    if args.cpu or args.device in (None, "cpu"):
        print("[3/3] CPU 推理（同框架同 seed）")
        out_cpu = run_once(args, "cpu", npz, af3)

    print("\n== NPU vs CPU（NPU 计算精度）")
    if out_cpu is None:
        print("    未启用 --cpu，跳过")
    else:
        for key in ("contact_probs", "predicted_lddt", "full_pae", "full_pde"):
            a, b = pick(out_npu, key), pick(out_cpu, key)
            if a is not None and b is not None:
                compare_arrays(key, a.numpy(), b.numpy())

    if args.reference:
        print("\n== NPU vs JAX 参考（总差异）")
        try:
            seed, refs = load_reference(args.reference)
        except Exception as exc:  # noqa: BLE001
            print(f"    参考 pkl 加载失败（需要 alphafold3 已安装）: {exc}")
            return 0
        print(f"    参考 seed={seed}，样本数 {len(refs)}")
        ref0 = refs[0]
        num = getattr(ref0, "numerical_data", {}) or {}
        meta = getattr(ref0, "metadata", {}) or {}
        for key in REF_ARRAY_KEYS:
            ours_t = pick(out_npu, key)
            if key in num and ours_t is not None:
                ours = ours_t.numpy()
                theirs = np.asarray(num[key])
                # 参考是 token 级，我们的可能是全 token+pad，取前 N×N
                n = theirs.shape[-1]
                ours = ours[..., :n, :n]
                compare_arrays(key, ours, theirs)
        for key in REF_SCALAR_KEYS:
            if key in meta:
                print(f"    {key:22s} 参考 {float(meta[key]):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
