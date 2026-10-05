#!/usr/bin/env bash
set -euo pipefail
# Run inside the existing, version-pinned training container.
cd "${TRAIN_WORKSPACE:?Set TRAIN_WORKSPACE}"
: "${OUTPUT_DIR:?Use a separate output directory for each probe}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE=4 MASTER_PORT="${MASTER_PORT:-29661}"
export ACCELERATE_USE_FSDP=true FSDP_CPU_RAM_EFFICIENT_LOADING=true FSDP_VERSION=2
export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=1800
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True DECISION_PAD_MULTIPLE=128
batch="${MICROBATCH:-1}"
case "$batch" in 1|2|4) ;; *) echo 'MICROBATCH must be 1, 2 or 4' >&2; exit 2;; esac
accumulation=$((4 / batch))
callback=decision_throughput
evaluation_plugin=()
if [[ "${DECISION_EVAL_ONLY:-0}" == 1 ]]; then
  callback=decision_checkpoint_eval
  evaluation_plugin=("$(dirname "$PROBE_PLUGIN")/checkpoint_eval.py")
fi
extra=()
if [[ -n "${RESUME_FROM:-}" ]]; then extra+=(--resume_from_checkpoint "$RESUME_FROM"); fi
swift sft \
  --model "${MODEL_PATH:?Set MODEL_PATH}" --model_type qwen3_5_moe \
  --external_plugins "$PWD/decision_plugin.py" "$PWD/checkpoint_fence.py" \
    "$PWD/cpu_offload.py" "$PWD/lazy_cpu_init.py" "$PWD/checkpoint_space.py" \
    "${PROBE_PLUGIN:?Set PROBE_PLUGIN}" "$(dirname "$PROBE_PLUGIN")/decision_sparse_logits.py" "${evaluation_plugin[@]}" \
  --callbacks "$callback" \
  --template intern_decision_training --new_special_tokens '<decision>' \
  --tuner_type full --freeze_vit true --freeze_aligner true --freeze_llm false \
  --dataset "${TRAIN_DATA:-$PWD/decision_data/train.jsonl}" \
  --val_dataset "$PWD/decision_data/validation.jsonl" \
  --remove_unused_columns false --strict true --split_dataset_ratio 0 \
  --enable_thinking false --max_length 8192 --truncation_strategy delete \
  --packing false --padding_free false --attn_impl sdpa \
  --torch_dtype float32 --bf16 true --fp16 false --fsdp "$PWD/fsdp2-sharded.json" \
  --gradient_checkpointing false --use_logits_to_keep true --accelerator_config "$PWD/accelerator.json" \
  --per_device_train_batch_size "$batch" --gradient_accumulation_steps "$accumulation" \
  --per_device_eval_batch_size 1 --learning_rate 2e-6 \
  --weight_decay 0.01 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --max_steps "${TOTAL_STEPS:-600}" \
  --save_strategy "${SAVE_STRATEGY:-no}" --save_steps "${SAVE_STEPS:-150}" \
  --eval_strategy "${EVAL_STRATEGY:-no}" --eval_steps 150 \
  --load_best_model_at_end false --save_only_model false --save_total_limit 1 \
  --logging_steps 1 --report_to none --seed 42 --data_seed 42 \
  --dataset_num_proc 1 --dataloader_num_workers 0 \
  --output_dir "$OUTPUT_DIR" --add_version false "${extra[@]}"
