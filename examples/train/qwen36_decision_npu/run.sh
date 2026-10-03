#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 MASTER_PORT=29651
export ACCELERATE_USE_FSDP=true FSDP_CPU_RAM_EFFICIENT_LOADING=true FSDP_VERSION=2
export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=/workspace:${PYTHONPATH:-}
export HCCL_CONNECT_TIMEOUT=300 HCCL_EXEC_TIMEOUT=1800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True DECISION_PAD_MULTIPLE=128
stage=${1:-probe-cpu}
model=/models/Qwen3.6-35B-A3B
data=/workspace/decision_data/train.jsonl
fsdp_config=/workspace/fsdp2.json
case "$stage" in
  tiny) model=/workspace/tiny-model; data=/workspace/decision_data/tiny.jsonl; extra=(--max_steps 2 --save_steps 1 --eval_strategy no) ;;
  tiny-resume) model=/workspace/tiny-model; data=/workspace/decision_data/tiny.jsonl; extra=(--max_steps 3 --save_steps 1 --eval_strategy no --resume_from_checkpoint /workspace/outputs/tiny/checkpoint-2) ;;
  probe|probe-cpu) extra=(--max_steps 2 --save_strategy no --eval_strategy no) ;;
  tiny-sharded) model=/workspace/tiny-model; data=/workspace/decision_data/tiny.jsonl; fsdp_config=/workspace/fsdp2-sharded.json; extra=(--max_steps 2 --save_steps 2 --eval_strategy no) ;;
  tiny-sharded-resume) model=/workspace/tiny-model; data=/workspace/decision_data/tiny.jsonl; fsdp_config=/workspace/fsdp2-sharded.json; extra=(--max_steps 3 --save_steps 3 --eval_strategy no --resume_from_checkpoint /workspace/outputs/tiny-sharded/checkpoint-2) ;;
  smoke) extra=(--max_steps 2 --save_steps 2 --eval_strategy no) ;;
  resume) extra=(--max_steps 3 --save_steps 3 --eval_strategy no --resume_from_checkpoint /workspace/outputs/smoke/checkpoint-2) ;;
  train) extra=(--num_train_epochs 2 --save_steps 150 --eval_strategy steps --eval_steps 150 --load_best_model_at_end true --metric_for_best_model loss --greater_is_better false) ;;
  *) exit 2 ;;
esac
swift sft \
  --model "$model" --model_type qwen3_5_moe \
  --external_plugins /workspace/decision_plugin.py /workspace/checkpoint_fence.py \
  --template intern_decision_training --new_special_tokens '<decision>' \
  --tuner_type full --freeze_vit true --freeze_aligner true --freeze_llm false \
  --dataset "$data" --val_dataset /workspace/decision_data/validation.jsonl \
  --remove_unused_columns false --strict true --split_dataset_ratio 0 \
  --enable_thinking false --max_length 8192 --truncation_strategy delete \
  --packing false --padding_free false --attn_impl sdpa \
  --torch_dtype bfloat16 --bf16 true --fsdp "$fsdp_config" \
  --gradient_checkpointing false --use_logits_to_keep true \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --per_device_eval_batch_size 1 --learning_rate 2e-6 \
  --weight_decay 0.01 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --save_only_model false --save_total_limit 1 \
  --logging_steps 1 --report_to none --seed 42 --data_seed 42 \
  --dataset_num_proc 1 --dataloader_num_workers 0 \
  --output_dir "/workspace/outputs/$stage" --add_version false "${extra[@]}"
