#!/usr/bin/env bash
set -euo pipefail

# v0 launcher (theory-aligned):
# - minimal/byte port (A5: language is just an interface)
# - recurrent dynamics drives learning (not a big transformer doing the work)
# - offline rollout participates via a single unified functional (A2/A3)
#
# Usage:
#   ENV_NAME=PINNs DEVICE=cuda STEPS=20000 SAVE_DIR=HEI/checkpoints/v0_lang ./HEI/training/scripts/run_v0.sh
#
# Resume:
#   Rerun the same command; it resumes automatically if `SAVE_DIR/last.pt` exists.

ENV_NAME="${ENV_NAME:-PINNs}"
DEVICE="${DEVICE:-cuda}"

SAVE_DIR="${SAVE_DIR:-HEI/checkpoints/v0_lang}"

STEPS="${STEPS:-20000}"
MAX_TOKENS="${MAX_TOKENS:-0}" # 0 = disabled, stop by STEPS

LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}"
DETACH_EVERY="${DETACH_EVERY:-64}"

DIM_Q="${DIM_Q:-512}"
DIM_Z="${DIM_Z:-32}"
NUM_CHARTS="${NUM_CHARTS:-16}"
PORT_COUPLING_TOP_K="${PORT_COUPLING_TOP_K:-4}"
PORT_COUPLING_IMPL="${PORT_COUPLING_IMPL:-grouped}" # grouped|dense

PORT_TRAINABLE="${PORT_TRAINABLE:-1}" # 1=calibrate minimal IO map; 0=fully fixed port (hard to learn language)

NUM_OFFLINE_STEPS="${NUM_OFFLINE_STEPS:-1}"
OFFLINE_DT="${OFFLINE_DT:-0.1}"
OFFLINE_REPLAY_MODE="${OFFLINE_REPLAY_MODE:-prioritized}"
OFFLINE_LOSS_MODE="${OFFLINE_LOSS_MODE:-relu_delta}"
OFFLINE_WEIGHT="${OFFLINE_WEIGHT:-1.0}"
EXPERIENCE_PUSH_N="${EXPERIENCE_PUSH_N:-8}"

LOG_EVERY="${LOG_EVERY:-50}"
SAMPLE_EVERY="${SAMPLE_EVERY:-200}"
SAVE_EVERY="${SAVE_EVERY:-500}"

MAX_SAMPLES_WIKI="${MAX_SAMPLES_WIKI:-200000}"
MAX_SAMPLES_CLUE="${MAX_SAMPLES_CLUE:-200000}"

HF_DATASET="${HF_DATASET:-}"
HF_NAME="${HF_NAME:-}"
HF_SPLIT="${HF_SPLIT:-train}"
HF_TEXT_FIELD="${HF_TEXT_FIELD:-text}"
MAX_SAMPLES_HF="${MAX_SAMPLES_HF:-0}"
HF_WEIGHT="${HF_WEIGHT:-0.0}"

PORT_INIT_STD="${PORT_INIT_STD:-0.02}"

SCHEDULED_SAMPLING_PROB="${SCHEDULED_SAMPLING_PROB:-0.0}"
SCHEDULED_SAMPLING_MODE="${SCHEDULED_SAMPLING_MODE:-sample}"
SCHEDULED_SAMPLING_TOP_K="${SCHEDULED_SAMPLING_TOP_K:-20}"
SCHEDULED_SAMPLING_TEMPERATURE="${SCHEDULED_SAMPLING_TEMPERATURE:-1.0}"

ROUTER_BALANCE_WEIGHT="${ROUTER_BALANCE_WEIGHT:-0.0}"
ROUTER_ENTROPY_WEIGHT="${ROUTER_ENTROPY_WEIGHT:-0.0}"

TRANSPORT_THRESHOLD="${TRANSPORT_THRESHOLD:-0.1}"
CONNECTION_RANK="${CONNECTION_RANK:-0}"

Q_CLIP_NORM="${Q_CLIP_NORM:-100.0}"
P_CLIP_NORM="${P_CLIP_NORM:-100.0}"
S_CLIP_ABS="${S_CLIP_ABS:-100.0}"
SANITIZE_NONFINITE="${SANITIZE_NONFINITE:-1}"

SAMPLER_LOSS_UPDATE_EVERY="${SAMPLER_LOSS_UPDATE_EVERY:-1}"
SAMPLER_LOSS_UPDATE_MODE="${SAMPLER_LOSS_UPDATE_MODE:-per_sample}"

TF32="${TF32:-0}"
AMP="${AMP:-0}"
AMP_DTYPE="${AMP_DTYPE:-bf16}" # bf16|fp16
FUSED_ADAMW="${FUSED_ADAMW:-1}"

PYTHON=(conda run -n "${ENV_NAME}" python)
PYTEST=(conda run -n "${ENV_NAME}" pytest)

echo "[v0] env=${ENV_NAME} device=${DEVICE}"
echo "[v0] save_dir=${SAVE_DIR} steps=${STEPS} max_tokens=${MAX_TOKENS}"

echo "[v0] running minimal gate tests..."
PYTHONPATH=HEI "${PYTEST[@]}" -q \
  HEI/tests/test_checkpoint_io.py \
  HEI/tests/test_offline_learning_alignment.py \
  HEI/tests/test_L1_gate.py

RESUME_ARGS=()
if [[ -f "${SAVE_DIR}/last.pt" ]]; then
  echo "[v0] resume: ${SAVE_DIR}/last.pt"
  RESUME_ARGS=(--resume_dir "${SAVE_DIR}")
fi

echo "[v0] launching training..."
PYTHONPATH=HEI "${PYTHON[@]}" HEI/training/train_active_sampling.py \
  "${RESUME_ARGS[@]}" \
  --device "${DEVICE}" \
  --max_samples_wiki "${MAX_SAMPLES_WIKI}" \
  --max_samples_clue "${MAX_SAMPLES_CLUE}" \
  --hf_dataset "${HF_DATASET}" \
  --hf_name "${HF_NAME}" \
  --hf_split "${HF_SPLIT}" \
  --hf_text_field "${HF_TEXT_FIELD}" \
  --max_samples_hf "${MAX_SAMPLES_HF}" \
  --hf_weight "${HF_WEIGHT}" \
  --tokenizer byte \
  --port_arch minimal \
  --sequence_mode recurrent \
  --tie_io_weights 1 \
  --port_trainable "${PORT_TRAINABLE}" \
  --port_init_std "${PORT_INIT_STD}" \
  --scheduled_sampling_prob "${SCHEDULED_SAMPLING_PROB}" \
  --scheduled_sampling_mode "${SCHEDULED_SAMPLING_MODE}" \
  --scheduled_sampling_top_k "${SCHEDULED_SAMPLING_TOP_K}" \
  --scheduled_sampling_temperature "${SCHEDULED_SAMPLING_TEMPERATURE}" \
  --router_balance_weight "${ROUTER_BALANCE_WEIGHT}" \
  --router_entropy_weight "${ROUTER_ENTROPY_WEIGHT}" \
  --lr "${LR}" \
  --dim_q "${DIM_Q}" \
  --dim_z "${DIM_Z}" \
  --num_charts "${NUM_CHARTS}" \
  --port_coupling_top_k "${PORT_COUPLING_TOP_K}" \
  --port_coupling_impl "${PORT_COUPLING_IMPL}" \
  --transport_threshold "${TRANSPORT_THRESHOLD}" \
  --connection_rank "${CONNECTION_RANK}" \
  --q_clip_norm "${Q_CLIP_NORM}" \
  --p_clip_norm "${P_CLIP_NORM}" \
  --s_clip_abs "${S_CLIP_ABS}" \
  --sanitize_nonfinite "${SANITIZE_NONFINITE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --detach_every "${DETACH_EVERY}" \
  --tf32 "${TF32}" \
  --amp "${AMP}" \
  --amp_dtype "${AMP_DTYPE}" \
  --fused_adamw "${FUSED_ADAMW}" \
  --num_offline_steps "${NUM_OFFLINE_STEPS}" \
  --offline_dt "${OFFLINE_DT}" \
  --offline_replay_mode "${OFFLINE_REPLAY_MODE}" \
  --offline_loss_mode "${OFFLINE_LOSS_MODE}" \
  --offline_weight "${OFFLINE_WEIGHT}" \
  --experience_push_n "${EXPERIENCE_PUSH_N}" \
  --steps "${STEPS}" \
  --max_tokens "${MAX_TOKENS}" \
  --log_every "${LOG_EVERY}" \
  --sample_every "${SAMPLE_EVERY}" \
  --sampler_loss_update_every "${SAMPLER_LOSS_UPDATE_EVERY}" \
  --sampler_loss_update_mode "${SAMPLER_LOSS_UPDATE_MODE}" \
  --save_dir "${SAVE_DIR}" \
  --save_every "${SAVE_EVERY}"
