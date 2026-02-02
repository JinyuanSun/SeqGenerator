#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET="${DATASET:-aspartese}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/datasets/${DATASET}}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/diffusion_models/esm_orig/esm2_t30_150M_UR50D.pt}"
MODEL_REGRESSION_PATH="${MODEL_REGRESSION_PATH:-${ROOT_DIR}/diffusion_models/esm_orig/esm2_t30_150M_UR50D-contact-regression.pt}"

DIFF_STEPS="${DIFF_STEPS:-2000}"
LR="${LR:-0.00005}"
LEARNING_STEPS="${LEARNING_STEPS:-1000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
SEED="${SEED:-102}"
NOISE_SCHEDULE="${NOISE_SCHEDULE:-sqrt}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MICROBATCH="${MICROBATCH:-64}"
MAX_LEN="${MAX_LEN:-490}"
MIN_LEN="${MIN_LEN:-460}"
SCHEDULE_SAMPLER="${SCHEDULE_SAMPLER:-lossaware}"
NOTES="${NOTES:-run}"
MASTER_PORT="${MASTER_PORT:-12236}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-none}"
EXTRA_ARGS_STR="${EXTRA_ARGS:-}"
EXTRA_ARGS=()
if [[ -n "${EXTRA_ARGS_STR}" ]]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_ARGS_STR}"
fi

TIMESTAMP="$(date +"%Y%m%d-%H%M%S")"
MODEL_DIR="${ROOT_DIR}/diffusion_models"
RUN_NAME="diffuseq_${DATASET}_lr${LR}_t${DIFF_STEPS}_${NOISE_SCHEDULE}_${SCHEDULE_SAMPLER}_seed${SEED}"
if [[ -n "${NOTES}" ]]; then
  RUN_NAME="${RUN_NAME}_${NOTES}_${TIMESTAMP}"
fi
CHECKPOINT_PATH="${MODEL_DIR}/${RUN_NAME}"

mkdir -p "${CHECKPOINT_PATH}"

export OPENAI_LOGDIR="${CHECKPOINT_PATH}"
export TOKENIZERS_PARALLELISM=false

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  --use_env "${ROOT_DIR}/train.py" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --dataset "${DATASET}" \
  --data_dir "${DATA_DIR}" \
  --lr "${LR}" \
  --batch_size "${BATCH_SIZE}" \
  --microbatch "${MICROBATCH}" \
  --diffusion_steps "${DIFF_STEPS}" \
  --noise_schedule "${NOISE_SCHEDULE}" \
  --schedule_sampler "${SCHEDULE_SAMPLER}" \
  --resume_checkpoint "${RESUME_CHECKPOINT}" \
  --seed "${SEED}" \
  --max_len "${MAX_LEN}" \
  --min_len "${MIN_LEN}" \
  --learning_steps "${LEARNING_STEPS}" \
  --save_interval "${SAVE_INTERVAL}" \
  --notes "${NOTES}" \
  --model_path "${MODEL_PATH}" \
  --model_regression_path "${MODEL_REGRESSION_PATH}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
