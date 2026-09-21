#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WFM_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${COSMOS_ROOT:?Set COSMOS_ROOT to the cloned cosmos-transfer1 repository}"
NUM_GPU="${NUM_GPU:-2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${COSMOS_ROOT}/checkpoints}"
: "${BATCH_INPUT_PATH:?Set BATCH_INPUT_PATH to a rewritten Cosmos batch JSONL spec}"
: "${VIDEO_SAVE_FOLDER:?Set VIDEO_SAVE_FOLDER to the generated-RGB output directory}"

if [[ ! "${NUM_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPU must be a positive integer" >&2
  exit 2
fi
if [[ ! -f "${COSMOS_ROOT}/cosmos_transfer1/diffusion/inference/transfer.py" ]]; then
  echo "COSMOS_ROOT is not a cosmos-transfer1 checkout: ${COSMOS_ROOT}" >&2
  exit 2
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((NUM_GPU - 1))")"
fi
export CUDA_VISIBLE_DEVICES

mkdir -p "${VIDEO_SAVE_FOLDER}"
cd "${COSMOS_ROOT}"
PYTHONPATH="${COSMOS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" torchrun \
  --nproc_per_node="${NUM_GPU}" \
  --nnodes=1 \
  --node_rank=0 \
  cosmos_transfer1/diffusion/inference/transfer.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --video_save_folder "${VIDEO_SAVE_FOLDER}" \
  --controlnet_specs "${WFM_ROOT}/specs/deafult_depth_gen_spec.json" \
  --sigma_max 80 \
  --fps 30 \
  --num_gpus "${NUM_GPU}" \
  --batch_input_path "${BATCH_INPUT_PATH}"
