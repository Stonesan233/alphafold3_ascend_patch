#!/usr/bin/env bash
# 离线批处理推理：整目录输入 -> 整目录输出，并校验产物齐全。
#
# 这是 AI4S 的实际用法（run_alphafold.py --input_dir），不走任何 HTTP 服务。
#
# 用法：
#   ./test/run_batch_inference.sh \
#       --workspace /data/af3 \
#       --input  /data/af3_inputs \
#       --output /data/af3_outputs \
#       --model-dir /mnt/weight/alphafold3 \
#       --db-dir    /mnt/weight/alphafold3/databases \
#       [--recycles 10] [--samples 5] [--steps 200] [--precision fp32] \
#       [--device npu:0] [--no-data-pipeline]
#
# 输入目录里放 *.json（官方 fold input 格式，参见 alphafold3/examples/）。
# 输出校验：每个 <name>.json 应有 <name>_model.cif 与 <name>_summary_confidences.json。
set -uo pipefail

WS=""; INPUT=""; OUTPUT=""; MODEL_DIR=""; DB_DIR=""
RECYCLES=10; SAMPLES=5; STEPS=200; PRECISION=fp32; DEVICE=""; DATA_PIPE=1

usage() { sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) WS="$2"; shift 2 ;;
    --input)     INPUT="$2"; shift 2 ;;
    --output)    OUTPUT="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --db-dir)    DB_DIR="$2"; shift 2 ;;
    --recycles)  RECYCLES="$2"; shift 2 ;;
    --samples)   SAMPLES="$2"; shift 2 ;;
    --steps)     STEPS="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --device)    DEVICE="$2"; shift 2 ;;
    --no-data-pipeline) DATA_PIPE=0; shift ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$WS" ]]        || { echo "缺少 --workspace" >&2; exit 2; }
[[ -n "$INPUT" ]]     || { echo "缺少 --input" >&2; exit 2; }
[[ -n "$OUTPUT" ]]    || { echo "缺少 --output" >&2; exit 2; }
[[ -n "$MODEL_DIR" ]] || { echo "缺少 --model-dir" >&2; exit 2; }
[[ -d "$INPUT" ]] || { echo "输入目录不存在: $INPUT" >&2; exit 2; }

XFSRC="$WS/xfold"
[[ -d "$XFSRC" ]] || { echo "找不到 $XFSRC" >&2; exit 2; }

export AF3_DEVICE="${DEVICE:-npu:0}"
export LIBCIFPP_DATA_DIR="${LIBCIFPP_DATA_DIR:-$WS/deps/libcifpp/rsrc}"
mkdir -p "$OUTPUT"

echo "设备: $AF3_DEVICE | 精度: $PRECISION | recycles=$RECYCLES samples=$SAMPLES steps=$STEPS"
echo "输入: $INPUT"
echo "输出: $OUTPUT"

ARGS=(
  --input_dir "$INPUT"
  --output_dir "$OUTPUT"
  --model_dir "$MODEL_DIR"
  --num_recycles "$RECYCLES"
  --num_diffusion_samples "$SAMPLES"
  --diffusion_steps "$STEPS"
  --precision "$PRECISION"
)
[[ $DATA_PIPE -eq 1 && -n "$DB_DIR" ]] && ARGS+=(--db_dir "$DB_DIR")
[[ $DATA_PIPE -eq 0 ]] && ARGS+=(--run_data_pipeline=False)

cd "$XFSRC"
start=$(date +%s)
python run_alphafold.py "${ARGS[@]}"
rc=$?
elapsed=$(( $(date +%s) - start ))
echo "run_alphafold.py 退出码 $rc，耗时 ${elapsed}s"

# 产物校验
missing=0
while IFS= read -r -d '' json; do
  name=$(basename "$json" .json)
  for suffix in _model.cif _summary_confidences.json; do
    if [[ ! -s "$OUTPUT/${name}${suffix}" ]]; then
      echo "MISSING $OUTPUT/${name}${suffix}"
      missing=1
    fi
  done
done < <(find "$INPUT" -maxdepth 1 -name '*.json' -print0)

if [[ $missing -eq 0 && $rc -eq 0 ]]; then
  echo "PASS：批处理完成，产物齐全"
  exit 0
fi
echo "FAIL：rc=$rc missing=$missing"
exit 1
