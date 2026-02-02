#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/diffusion_models/select}"
SEQ_LEN_SAMPLE="${SEQ_LEN_SAMPLE:-${ROOT_DIR}/datasets/aspartese/train.csv}"
MAX_LEN="${MAX_LEN:-490}"
MIN_LEN="${MIN_LEN:-460}"
SEQ_NUM="${SEQ_NUM:-500}"
# sample_seq2seq.py expects --seed2, so keep this name for compatibility.
SEED="${SEED:-123}"
SPLIT="${SPLIT:-test}"
STEP="${STEP:-2000}"
TOP_P="${TOP_P:--1}"
BATCH_SIZE="${BATCH_SIZE:-50}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/generation_outputs}"
ORG_MODEL="${ORG_MODEL:-${ROOT_DIR}/diffusion_models/esm_orig/esm2_t30_150M_UR50D.pt}"
MASTER_PORT="${MASTER_PORT:-12354}"
EXTRA_ARGS_STR="${EXTRA_ARGS:-}"
EXTRA_ARGS=()
if [[ -n "${EXTRA_ARGS_STR}" ]]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_ARGS_STR}"
fi

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  --use_env "${ROOT_DIR}/sample_seq2seq.py" \
  --model_diff_path "${MODEL_DIR}" \
  --seq_len_sample "${SEQ_LEN_SAMPLE}" \
  --max_len "${MAX_LEN}" \
  --min_len "${MIN_LEN}" \
  --seq_num "${SEQ_NUM}" \
  --seed2 "${SEED}" \
  --split "${SPLIT}" \
  --step "${STEP}" \
  --top_p "${TOP_P}" \
  --batch_size "${BATCH_SIZE}" \
  --out_dir "${OUT_DIR}" \
  --model_path "${ORG_MODEL}" \
  "${EXTRA_ARGS[@]}"
