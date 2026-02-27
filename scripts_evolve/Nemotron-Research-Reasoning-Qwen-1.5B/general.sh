#!/bin/bash
# General training script with configurable parameters
# Usage: ./scripts_evolve/${MODEL_NAME}/general.sh WANDB_PROJECT RUN_NAME INITIAL_PROGRAM EVALUATOR_FILE CONFIG_YAML SAVE_PATH IS_TRAINING LAZY_OUTPUT_PENALTY REWARD_PROCESS_TYPE SEED

if [ $# -ne 10 ]; then
  echo "Usage: $0 WANDB_PROJECT RUN_NAME INITIAL_PROGRAM EVALUATOR_FILE CONFIG_YAML SAVE_PATH IS_TRAINING LAZY_OUTPUT_PENALTY REWARD_PROCESS_TYPE SEED"
  echo ""
  echo "Required parameters:"
  echo "  WANDB_PROJECT        - Weights & Biases project name"
  echo "  RUN_NAME             - Experiment run name"
  echo "  INITIAL_PROGRAM      - Path to initial program file"
  echo "  EVALUATOR_FILE       - Path to evaluator file"
  echo "  CONFIG_YAML          - Path to config YAML file"
  echo "  SAVE_PATH            - Save directory path"
  echo "  IS_TRAINING          - True for training, False for inference-only"
  echo "  LAZY_OUTPUT_PENALTY  - Lazy output penalty level (1 or 2)"
  echo "  REWARD_PROCESS_TYPE  - Reward processing type (original_reward, rl_normalized_reward, etc.)"
  echo "  SEED                 - Random seed for reproducibility"
  exit 1
fi

WANDB_PROJECT=$1
RUN_NAME=$2
INITIAL_PROGRAM=$3
EVALUATOR_FILE=$4
CONFIG_YAML=$5
SAVE_PATH=$6
IS_TRAINING=$7
LAZY_OUTPUT_PENALTY=$8
REWARD_PROCESS_TYPE=$9
SEED=${10}

SAVE_SHM_DIR="${SAVE_PATH}/shm"
CKPT_DIR="${SAVE_PATH}/${RUN_NAME}"
RECORD_PATH="${SAVE_PATH}/${RUN_NAME}/records"
MODEL_NAME="Nemotron-Research-Reasoning-Qwen-1.5B"

# Determine debug-rollout-only mode based on IS_TRAINING
if [ "$IS_TRAINING" = "False" ] || [ "$IS_TRAINING" = "false" ]; then
    DEBUG_ROLLOUT_ONLY="--debug-rollout-only"
    echo "Inference-only mode enabled (IS_TRAINING=$IS_TRAINING)"
else
    DEBUG_ROLLOUT_ONLY=""
    echo "Normal training mode (IS_TRAINING=$IS_TRAINING)"
fi

# Create checkpoint directory
mkdir -p "${CKPT_DIR}"

# Always use a fresh local Ray session.
unset RAY_ADDRESS
rm -f /tmp/ray/ray_current_cluster

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
# pkill -9 python
sleep 3
pkill -9 ray
# pkill -9 python

set -ex

# Raise open-file limit to prevent Ray grpc/raylet FD exhaustion.
ulimit -n 1048576 || ulimit -n 65535
echo "[limits] nofile soft limit: $(ulimit -n)"

export PYTHONBUFFERED=16
export TOKENIZERS_PARALLELISM=false

# NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l || echo 0)
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
echo "NVLINK_COUNT: $NVLINK_COUNT"
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source scripts/models/deepseek-r1-distill-qwen-1.5B.sh

# Auto-size Ray/SLIME GPU allocation to avoid hanging on oversized placement groups.
TOTAL_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${TOTAL_GPUS}" -ge 8 ]; then
  RAY_NUM_GPUS=8
  ACTOR_NUM_GPUS=4
  ROLLOUT_NUM_GPUS=4
elif [ "${TOTAL_GPUS}" -ge 4 ]; then
  RAY_NUM_GPUS=4
  ACTOR_NUM_GPUS=2
  ROLLOUT_NUM_GPUS=2
elif [ "${TOTAL_GPUS}" -ge 2 ]; then
  RAY_NUM_GPUS=2
  ACTOR_NUM_GPUS=1
  ROLLOUT_NUM_GPUS=1
else
  echo "Need at least 2 GPUs, but detected ${TOTAL_GPUS}."
  exit 1
fi
echo "[gpu] detected=${TOTAL_GPUS}, ray=${RAY_NUM_GPUS}, actor=${ACTOR_NUM_GPUS}, rollout=${ROLLOUT_NUM_GPUS}"

CKPT_ARGS=(
   --hf-checkpoint "${SAVE_SHM_DIR}/${MODEL_NAME}"
   --ref-load "${SAVE_SHM_DIR}/${MODEL_NAME}_torch_dist"
   --load "${CKPT_DIR}/"
   --save "${CKPT_DIR}/"
   --save-interval 10
)

ROLLOUT_ARGS=(
  --disable-rollout-global-dataset
  --data-source-path slime.rollout.data_source.RolloutDataSourceWithBuffer
  --evolving-gym
  --evolving-gym-initial-program "${INITIAL_PROGRAM}"
  --evolving-gym-evaluator-file "${EVALUATOR_FILE}"
  --evolving-gym-config-path "${CONFIG_YAML}"
  --evolving-gym-max-concurrent-evals 16
  --evolving-gym-log-prompts
  --evolving-gym-record
  --evolving-gym-record-dir "${RECORD_PATH}"
  --evolving-gym-lazy-output-penalty-level "${LAZY_OUTPUT_PENALTY}"
  --evolving-gym-seed ${SEED}
  --evolving-gym-reward-process-type "${REWARD_PROCESS_TYPE}"

  --apply-chat-template

  --rm-type evolving-gym
  --reward-key reward

  --num-rollout 10000
  --rollout-batch-size 32
  --n-samples-per-prompt 16
  --rollout-max-response-len 16384
  --rollout-temperature 1.0

  --over-sampling-batch-size 32
  # --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
  --partial-rollout

  --num-steps-per-rollout 1
  --wandb-always-use-train-step
  --balance-data
)


PERF_ARGS=(
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 2
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1

  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1

  --use-dynamic-batch-size
  --max-tokens-per-gpu 2048
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28

  --use-tis
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

WANDB_ARGS=(
  --use-wandb
  --wandb-team ${WANDB_ENTITY}
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${RUN_NAME}"
  --wandb-key "${WANDB_API_KEY}"
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.8
  --sglang-server-concurrency 256
)

MISC_ARGS=(
  ${DEBUG_ROLLOUT_ONLY}
  --seed ${SEED}
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

# Start Ray (training/inference separation: don't use --colocate; use train_async.py)
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# Keep Ray temp path very short; AF_UNIX sockets must stay <= 107 chars.
RAY_TMP_DIR="$(mktemp -d /tmp/ray.XXXXXX)"
echo "[ray] temp dir: ${RAY_TMP_DIR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${RAY_NUM_GPUS} --temp-dir "${RAY_TMP_DIR}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Disable Triton
export TRITON_DISABLE=1



export FAST_MOUNT=$SAVE_PATH/fast_mount
export HF_DATASETS_CACHE=$FAST_MOUNT/hf/datasets
export DATASETS_CACHE=$HF_DATASETS_CACHE
export DATASETS_TMPDIR=$FAST_MOUNT/tmp
export PYARROW_TMP_DIR=$FAST_MOUNT/tmp

mkdir -p "$HF_DATASETS_CACHE" "$DATASETS_TMPDIR"
echo "[disk] HF_DATASETS_CACHE=$HF_DATASETS_CACHE TMPDIR=$TMPDIR"

export PYTHONPATH="/root/Megatron-LM/"
export CUDA_DEVICE_MAX_CONNECTIONS="1"
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
export DATASETS_CACHE="${HF_DATASETS_CACHE}"
export WANDB_GROUP="${RUN_NAME}"
export SLIME_PG_CPU_PER_BUNDLE="${SLIME_PG_CPU_PER_BUNDLE:-0.5}"

python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node ${ACTOR_NUM_GPUS} \
  --rollout-num-gpus ${ROLLOUT_NUM_GPUS} \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${DISTRIBUTED_ARGS[@]} \
  ${WANDB_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${MISC_ARGS[@]}
