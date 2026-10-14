#!/usr/bin/env bash
# Text decision LoRA through the standard ScoringTrainer.
set -euo pipefail
: "${MODEL_PATH:?Set a local StartLux checkpoint}"
: "${STARTLUX_TRAIN_FILE:?Set a local training JSONL file}"
: "${STARTLUX_SOURCE:?Set the official StartLux source checkout}"
: "${OUTPUT_DIR:?Choose a new output directory}"
export PYTHONPATH="$STARTLUX_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
swift sft \
  --external_plugins examples/train/decision/startlux.py \
  --model "$MODEL_PATH" --model_type startlux_decision \
  --task_type decision --loss_type scoring --dataset startlux_decision \
  --tuner_type lora --target_modules q_proj v_proj \
  --torch_dtype bfloat16 --attn_impl sdpa \
  --lora_rank 8 --lora_alpha 16 --learning_rate 1e-5 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --gradient_checkpointing true --max_length 1024 \
  --max_steps "${MAX_STEPS:-2}" --save_steps "${MAX_STEPS:-2}" \
  --logging_steps 1 --split_dataset_ratio 0 --dataloader_num_workers 0 \
  --report_to none --output_dir "$OUTPUT_DIR" "$@"
