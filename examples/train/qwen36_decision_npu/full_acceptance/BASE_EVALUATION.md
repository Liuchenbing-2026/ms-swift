# Qwen3.6-35B-A3B untrained decision baseline

Evaluate the original checkpoint before decision fine-tuning, using the same
seven-suite decision contract as the trained checkpoints. This measures the
fixed decision-template baseline, not unrestricted chat or reasoning performance.

Use the pinned training image from the parent README in a new container. Mount
original model weights and prepared evaluation data read-only; write results to
a new task directory. The evaluator checks dataset hashes, complete row/field
coverage and separation from the historical training records. Labels are never
included in the input. All seven suites contain 10,751 rows and 12,351 decisions;
the final score is the unweighted mean of the seven suite accuracies.

The optional inference recipe uses four allocated NPUs, BF16 model loading and
FSDP without CPU offload. Disable training autocast for this BF16-only
evaluation, avoiding Accelerate creating FP32 master parameters. These settings reduce evaluation memory and transfer
cost; they do not change weights on disk. They differ from the historical
FP32-load/BF16-compute training evaluation, so do not claim bitwise
numerical equivalence. Run the existing small-model execution check first.

```bash
export TRAIN_WORKSPACE='<TASK_WORKSPACE>'
export MODEL_PATH='<ORIGINAL_CHECKPOINT>'
export ASCEND_RT_VISIBLE_DEVICES='<FOUR_ASSIGNED_NPU_IDS>'
export NPROC_PER_NODE=4 EVAL_LOAD_DTYPE=bfloat16 EVAL_AUTOCAST_BF16=false
export ACCEPTANCE_CODE="$TRAIN_WORKSPACE/full_acceptance"
export PROBE_PLUGIN="$TRAIN_WORKSPACE/throughput_tuning/throughput_probe.py"
export FULL_EVAL_PLUGIN="$ACCEPTANCE_CODE/checkpoint_eval.py"
export FSDP_CONFIG="$ACCEPTANCE_CODE/fsdp2-inference.json"
export OUTPUT_DIR='<FRESH_OUTPUT_DIR>'
export DECISION_EVAL_CONFIG='<NEW_BASE_CONFIG_JSON>'
python "$ACCEPTANCE_CODE/prepare_base_config.py" \
  --source '<EXISTING_VERIFIED_FULL_CONFIG_JSON>' \
  --output "$DECISION_EVAL_CONFIG" --batch-size 8
export RESUME_FROM='' DECISION_EVAL_ONLY=1 DECISION_PROBE_UPDATES=0
export DECISION_SPARSE_LOGITS=1 MICROBATCH=1
export SAVE_STRATEGY=no EVAL_STRATEGY=no MASTER_PORT='<FREE_MASTER_PORT>'
bash "$ACCEPTANCE_CODE/run_evaluation.sh"
```

The SWIFT launcher initializes the model and the evaluator then exits before the
first optimizer update. Accept only a completed `evaluation.json` with
`checkpoint_step == 0`, `optimizer_updates_performed == 0` and complete seven-suite
counts. `TOTAL_STEPS` in the inherited launcher is not a completed training count.
The runtime adds the `<decision>` marker with fixed seed 42; this is the same
masked-answer input contract used for the paired trained-model comparison.
Do not describe this as the untouched tokenizer's normal chat protocol.

Retain per-rank predictions, source/data/config fingerprints, model identity and
logs locally. Training results and machine details are not published with code.
A failed process or incomplete suite must not be averaged as a completed result.

A two-device BF16 attempt exhausted NPU memory while distributing the full
state dict, before scoring started. It is not a validated deployment recipe.
The four-device run remains pending until all seven suites pass; the small-model
check alone does not validate the real 35B memory requirement.
