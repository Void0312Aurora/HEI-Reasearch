#!/usr/bin/env bash
set -euo pipefail

# v0 evaluation runner:
# - gate health checks (finite metrics)
# - protocol-5 closed-loop diagnostics
# - atlas/skill metrics (router usage)
# - L2 holonomy proxy
#
# Usage:
#   ENV_NAME=PINNs DEVICE=cuda SAVE_DIR=HEI/checkpoints/v0_lang ./HEI/training/scripts/eval_v0.sh

ENV_NAME="${ENV_NAME:-PINNs}"
DEVICE="${DEVICE:-cuda}"

SAVE_DIR="${SAVE_DIR:-HEI/checkpoints/v0_lang}"
CKPT="${CKPT:-${SAVE_DIR}/last.pt}"

MAX_SAMPLES_WIKI="${MAX_SAMPLES_WIKI:-200000}"
MAX_SAMPLES_CLUE="${MAX_SAMPLES_CLUE:-200000}"

HF_DATASET="${HF_DATASET:-}"
HF_NAME="${HF_NAME:-}"
HF_SPLIT="${HF_SPLIT:-validation}"
HF_TEXT_FIELD="${HF_TEXT_FIELD:-text}"
MAX_SAMPLES_HF="${MAX_SAMPLES_HF:-0}"
HF_WEIGHT="${HF_WEIGHT:-0.0}"

GATE_EVAL_SAMPLES="${GATE_EVAL_SAMPLES:-512}"
GATE_BATCH_SIZE="${GATE_BATCH_SIZE:-16}"
GATE_MAX_SEQ_LEN="${GATE_MAX_SEQ_LEN:-128}"

LANG_EVAL_SAMPLES="${LANG_EVAL_SAMPLES:-20000}"
LANG_BATCH_SIZE="${LANG_BATCH_SIZE:-16}"
LANG_MAX_SEQ_LEN="${LANG_MAX_SEQ_LEN:-128}"
GEN_LEN="${GEN_LEN:-64}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-20}"

P5_EVAL_SAMPLES="${P5_EVAL_SAMPLES:-2000}"
P5_NUM_PROMPTS="${P5_NUM_PROMPTS:-200}"
P5_PROMPT_CHARS="${P5_PROMPT_CHARS:-4}"
P5_BATCH_SIZE="${P5_BATCH_SIZE:-16}"
P5_MAX_SEQ_LEN="${P5_MAX_SEQ_LEN:-128}"
P5_MAX_NEW_TOKENS="${P5_MAX_NEW_TOKENS:-64}"
P5_TOP_K="${P5_TOP_K:-1}"
P5_TEMPERATURE="${P5_TEMPERATURE:-1.0}"

ATLAS_EVAL_SAMPLES="${ATLAS_EVAL_SAMPLES:-256}"
ATLAS_BATCH_SIZE="${ATLAS_BATCH_SIZE:-8}"
ATLAS_MAX_SEQ_LEN="${ATLAS_MAX_SEQ_LEN:-256}"

HOLONOMY_LOOPS="${HOLONOMY_LOOPS:-64}"
HOLONOMY_DELTA="${HOLONOMY_DELTA:-0.05}"
HOLONOMY_Q_STD="${HOLONOMY_Q_STD:-0.5}"

PYTHON=(conda run -n "${ENV_NAME}" python)
PYTEST=(conda run -n "${ENV_NAME}" pytest)

echo "[v0-eval] env=${ENV_NAME} device=${DEVICE}"
echo "[v0-eval] ckpt=${CKPT}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[v0-eval] ERROR: checkpoint not found: ${CKPT}" >&2
  exit 2
fi

echo "[v0-eval] running gate tests..."
PYTHONPATH=HEI "${PYTEST[@]}" -q \
  HEI/tests/test_checkpoint_io.py \
  HEI/tests/test_offline_learning_alignment.py \
  HEI/tests/test_L1_gate.py

echo "[v0-eval] gate health metrics..."
PYTHONPATH=HEI "${PYTHON[@]}" HEI/training/eval_v0_gate.py \
  --ckpt "${CKPT}" \
  --device "${DEVICE}" \
  --eval_samples "${GATE_EVAL_SAMPLES}" \
  --batch_size "${GATE_BATCH_SIZE}" \
  --max_seq_len "${GATE_MAX_SEQ_LEN}" \
  --max_samples_wiki "${MAX_SAMPLES_WIKI}" \
  --max_samples_clue "${MAX_SAMPLES_CLUE}" \
  --hf_dataset "${HF_DATASET}" \
  --hf_name "${HF_NAME}" \
  --hf_split "${HF_SPLIT}" \
  --hf_text_field "${HF_TEXT_FIELD}" \
  --max_samples_hf "${MAX_SAMPLES_HF}"

echo "[v0-eval] atlas/skill metrics..."
PYTHONPATH=HEI "${PYTHON[@]}" HEI/training/eval_v0_skill_atlas.py \
  --ckpt "${CKPT}" \
  --device "${DEVICE}" \
  --eval_samples "${ATLAS_EVAL_SAMPLES}" \
  --batch_size "${ATLAS_BATCH_SIZE}" \
  --max_seq_len "${ATLAS_MAX_SEQ_LEN}" \
  --max_samples_wiki "${MAX_SAMPLES_WIKI}" \
  --max_samples_clue "${MAX_SAMPLES_CLUE}" \
  --hf_dataset "${HF_DATASET}" \
  --hf_name "${HF_NAME}" \
  --hf_split "${HF_SPLIT}" \
  --hf_text_field "${HF_TEXT_FIELD}" \
  --max_samples_hf "${MAX_SAMPLES_HF}" \
  --hf_weight "${HF_WEIGHT}"

echo "[v0-eval] L2 holonomy proxy..."
PYTHONPATH=HEI "${PYTHON[@]}" HEI/training/eval_v0_l2_holonomy.py \
  --ckpt "${CKPT}" \
  --device "${DEVICE}" \
  --num_loops "${HOLONOMY_LOOPS}" \
  --delta "${HOLONOMY_DELTA}" \
  --q_std "${HOLONOMY_Q_STD}"

echo "[v0-eval] protocol-5 closed-loop diagnostics..."
PYTHONPATH=HEI "${PYTHON[@]}" HEI/training/eval_protocol5_closed_loop.py \
  --ckpt "${CKPT}" \
  --device "${DEVICE}" \
  --eval_samples "${P5_EVAL_SAMPLES}" \
  --num_prompts "${P5_NUM_PROMPTS}" \
  --prompt_chars "${P5_PROMPT_CHARS}" \
  --batch_size "${P5_BATCH_SIZE}" \
  --max_seq_len "${P5_MAX_SEQ_LEN}" \
  --max_new_tokens "${P5_MAX_NEW_TOKENS}" \
  --temperature "${P5_TEMPERATURE}" \
  --top_k "${P5_TOP_K}" \
  --max_samples_wiki "${MAX_SAMPLES_WIKI}" \
  --max_samples_clue "${MAX_SAMPLES_CLUE}" \
  --hf_dataset "${HF_DATASET}" \
  --hf_name "${HF_NAME}" \
  --hf_split "${HF_SPLIT}" \
  --hf_text_field "${HF_TEXT_FIELD}" \
  --max_samples_hf "${MAX_SAMPLES_HF}" \
  --hf_weight "${HF_WEIGHT}"

echo "[v0-eval] language skill quick eval..."
PYTHONPATH=HEI "${PYTHON[@]}" HEI/training/eval_language_skill.py \
  --ckpt "${CKPT}" \
  --device "${DEVICE}" \
  --eval_samples "${LANG_EVAL_SAMPLES}" \
  --batch_size "${LANG_BATCH_SIZE}" \
  --max_seq_len "${LANG_MAX_SEQ_LEN}" \
  --gen_len "${GEN_LEN}" \
  --temperature "${TEMPERATURE}" \
  --top_k "${TOP_K}" \
  --max_samples_wiki "${MAX_SAMPLES_WIKI}" \
  --max_samples_clue "${MAX_SAMPLES_CLUE}" \
  --hf_dataset "${HF_DATASET}" \
  --hf_name "${HF_NAME}" \
  --hf_split "${HF_SPLIT}" \
  --hf_text_field "${HF_TEXT_FIELD}" \
  --max_samples_hf "${MAX_SAMPLES_HF}"

echo "[v0-eval] done."
