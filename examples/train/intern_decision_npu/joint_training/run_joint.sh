#!/usr/bin/env bash
set -euo pipefail
cd "${TRAIN_WORKSPACE:?Set TRAIN_WORKSPACE}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export DECISION_PAD_MULTIPLE=128
export DECISION_DISABLE_THP=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
unset ACCELERATE_USE_FSDP FSDP_VERSION NPROC_PER_NODE
# One visible physical card, full language-backbone training. Final weights only;
# this short experiment does not promise optimizer-state resume.
swift sft \
  --model "${MODEL_PATH:?Set MODEL_PATH}" --model_type qwen3_5 \
  --external_plugins "$PWD/decision_plugin.py" "$PWD/joint_validation.py" "$PWD/cpu_adamw.py" \
  --callbacks joint_validation --template intern_decision_training \
  --new_special_tokens '<decision>' --tuner_type full \
  --freeze_vit true --freeze_aligner true --freeze_llm false \
  --dataset "$PWD/data/train.jsonl" --dataset_shuffle false --train_dataloader_shuffle false \
  --remove_unused_columns false --strict true --split_dataset_ratio 0 \
  --enable_thinking false --max_length 8192 --truncation_strategy delete \
  --packing false --padding_free false --attn_impl sdpa \
  --torch_dtype float32 --bf16 true --fp16 false --gradient_checkpointing true --optim adamw_torch \
  --optimizer decision_cpu_adamw \
  --use_logits_to_keep true --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${ACCUMULATION:-4}" \
  --learning_rate 1e-6 --weight_decay 0.01 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --max_steps "${TOTAL_STEPS:-120}" --save_strategy no \
  --save_only_model true --save_total_limit 1 --eval_strategy no \
  --logging_steps 1 --report_to none --seed 42 --data_seed 42 \
  --dataset_num_proc 1 --dataloader_num_workers 0 --dataloader_pin_memory false \
  --output_dir "${OUTPUT_DIR:?Set a fresh output directory}" --add_version false
