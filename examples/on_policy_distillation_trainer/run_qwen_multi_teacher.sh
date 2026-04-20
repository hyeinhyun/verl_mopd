#!/usr/bin/env bash
# Multi-teacher on-policy distillation example (2 teachers, multiple datasets per teacher).
#
# Two teachers on a dedicated teacher pool. Each sample is routed by its `data_source`
# column. One teacher may serve many datasets via the `keys` list:
#   - math_teacher  (key=openai/gsm8k, keys=[math500, aime])
#   - code_teacher  (key=eurus_code, keys=[humaneval, mbpp])
#
# Data preparation:
#   python scripts/add_data_source.py \
#       -i data/gsm8k/train.parquet     -s "openai/gsm8k" \
#       -i data/math500/train.parquet   -s "math500" \
#       -i data/aime/train.parquet      -s "aime" \
#       -i data/eurus/train.parquet     -s "eurus_code" \
#       -i data/humaneval/train.parquet -s "humaneval" \
#       -i data/mbpp/train.parquet      -s "mbpp" \
#       --merge data/multi_teacher_train.parquet
#
# Pool constraint: sum(num_replicas * TP) == n_gpus_per_node * nnodes.
set -xeuo pipefail

############################ Quick Config ############################

ROLLOUT_NAME="vllm"

STUDENT_MODEL=Qwen/Qwen2.5-0.5B
MATH_TEACHER_MODEL=Qwen/Qwen2.5-Math-7B
CODE_TEACHER_MODEL=Qwen/Qwen2.5-Coder-7B

USE_POLICY_GRADIENT=True
DISTILLATION_LOSS_MODE="k1"
USE_FUSED_KERNELS=False

DISTILLATION_LOSS_MAX_CLAMP=10.0
DISTILLATION_LOG_PROB_MIN_CLAMP=-10.0

PROJECT_NAME='verl_mopd_multi_teacher'

MAX_PROMPT=1024
MAX_RESPONSE_LENGTH=2048
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))
TRAIN_PROMPT_BSZ=128
STUDENT_MICRO_BATCH_SIZE_PER_GPU=1
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
USE_DYNAMIC_BSZ=True

STUDENT_WORLD_SIZE=4

# Teacher pool: dedicated GPUs for all teachers.
MATH_TEACHER_NUM_REPLICAS=2
CODE_TEACHER_NUM_REPLICAS=2
TP=1
TEACHER_POOL_WORLD_SIZE=$(( (MATH_TEACHER_NUM_REPLICAS + CODE_TEACHER_NUM_REPLICAS) * TP ))

# Routing column: samples are dispatched to teachers by this column.
TEACHER_KEY_COLUMN=data_source

# Each teacher's routing values. `key` is the canonical identifier;
# `keys` lists additional data_source values this teacher handles.
MATH_TEACHER_KEY="openai/gsm8k"
MATH_TEACHER_EXTRA_KEYS='[math500, aime]'
CODE_TEACHER_KEY="eurus_code"
CODE_TEACHER_EXTRA_KEYS='[humaneval, mbpp]'

SP=1

EXP_NAME="fsdp/student-${STUDENT_MODEL}/multi-teacher/loss-${DISTILLATION_LOSS_MODE}/pg-${USE_POLICY_GRADIENT}"

ENFORCE_EAGER=True

############################ Paths ############################

# Use a single merged parquet with data_source column pre-tagged.
# See header comment for how to create it with scripts/add_data_source.py.
TRAIN_FILES="['${DATA_PATH}/multi_teacher_train.parquet']"
TEST_FILES="['${DATA_PATH}/multi_teacher_test.parquet']"

############################ Parameter Groups ############################

DATA=(
    data.train_files="$TRAIN_FILES"
    data.val_files="$TEST_FILES"
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.train_batch_size=$TRAIN_PROMPT_BSZ
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=True
)

MODEL=(
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER
)

teacher_block() {
    local NAME=$1 KEY=$2 EXTRA_KEYS=$3 MODEL_PATH=$4 NUM_REPLICAS=$5
    echo "+distillation.teacher_models.${NAME}.key=${KEY}"
    echo "+distillation.teacher_models.${NAME}.keys=${EXTRA_KEYS}"
    echo "+distillation.teacher_models.${NAME}.model_path=${MODEL_PATH}"
    echo "+distillation.teacher_models.${NAME}.num_replicas=${NUM_REPLICAS}"
    echo "+distillation.teacher_models.${NAME}.inference.name=${ROLLOUT_NAME}"
    echo "+distillation.teacher_models.${NAME}.inference.tensor_model_parallel_size=${TP}"
    echo "+distillation.teacher_models.${NAME}.inference.gpu_memory_utilization=0.8"
    echo "+distillation.teacher_models.${NAME}.inference.enforce_eager=${ENFORCE_EAGER}"
    echo "+distillation.teacher_models.${NAME}.inference.max_model_len=${MAX_NUM_TOKENS}"
    echo "+distillation.teacher_models.${NAME}.inference.max_num_batched_tokens=${MAX_NUM_TOKENS}"
    echo "+distillation.teacher_models.${NAME}.inference.max_num_seqs=${MAX_NUM_TOKENS}"
}

DISTILLATION=(
    distillation.enabled=True
    distillation.n_gpus_per_node=$TEACHER_POOL_WORLD_SIZE
    distillation.nnodes=1
    distillation.teacher_key=$TEACHER_KEY_COLUMN

    # Math teacher: serves openai/gsm8k + math500 + aime
    $(teacher_block math_teacher "$MATH_TEACHER_KEY" "$MATH_TEACHER_EXTRA_KEYS" "$MATH_TEACHER_MODEL" $MATH_TEACHER_NUM_REPLICAS)

    # Code teacher: serves eurus_code + humaneval + mbpp
    $(teacher_block code_teacher "$CODE_TEACHER_KEY" "$CODE_TEACHER_EXTRA_KEYS" "$CODE_TEACHER_MODEL" $CODE_TEACHER_NUM_REPLICAS)

    distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
    distillation.distillation_loss.topk=64
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=$USE_POLICY_GRADIENT
    distillation.distillation_loss.loss_max_clamp=$DISTILLATION_LOSS_MAX_CLAMP
    distillation.distillation_loss.log_prob_min_clamp=$DISTILLATION_LOG_PROB_MIN_CLAMP
)

STUDENT=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.n=1
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXP_NAME
    trainer.n_gpus_per_node=$STUDENT_WORLD_SIZE
    trainer.nnodes=1
    trainer.save_freq=200
    trainer.test_freq=5
    trainer.total_epochs=15
    trainer.val_before_train=False
    trainer.use_legacy_worker_impl=disable
    trainer.resume_mode=disable
    trainer.log_val_generations=5
)

############################ Launch ############################

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${DISTILLATION[@]}" \
    "${ROLLOUT[@]}" \
    "${STUDENT[@]}" \
    "${TRAINER[@]}" \
    "$@"
