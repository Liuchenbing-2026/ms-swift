# Qwen3.8-27B decision baseline evaluation

Evaluate the original Qwen3.8-27B checkpoint using the same seven-suite candidate-logit protocol as the Qwen3.6-35B-A3B baseline. This evaluates text-only, fixed decision prompts with thinking disabled. It is not a trained Decider model and does not install its decision head or bidirectional attention.

Use the environment, prepared datasets, plugins, and isolation instructions in [BASE_EVALUATION.md](BASE_EVALUATION.md). Select a new output directory and an independently allocated four-device container. Keep the checkpoint mounted read-only. The only launcher change is making MODEL_TYPE configurable; the default retains the earlier MoE behavior.

```bash
export MODEL_PATH=/models/Qwen3.8-27B
export MODEL_TYPE=qwen3_5
export NPROC_PER_NODE=4
export ASCEND_RT_VISIBLE_DEVICES='<ASSIGNED_NPU_IDS>'
export MASTER_PORT=29841
export EVAL_LOAD_DTYPE=bfloat16
export EVAL_AUTOCAST_BF16=false
export DECISION_EVAL_ONLY=1
export RESUME_FROM=
export OUTPUT_DIR=/workspace/results/base-seven
export DECISION_EVAL_CONFIG=/workspace/base-seven.json
# Set TRAIN_WORKSPACE, PROBE_PLUGIN, FULL_EVAL_PLUGIN and FSDP_CONFIG
# using BASE_EVALUATION.md before running this command.
bash run_evaluation.sh
```

The callback exits before the first optimizer update. It preserves ordinary causal attention and reads legal answer-symbol logits immediately before each decision marker, with all typed fields present in the input. The existing template registers the decision marker at load time; this is protocol setup, not decision fine-tuning. The source weights are not overwritten. Candidate order, dataset hashes, padding, and aggregation remain fixed.

Acceptance requires seven complete suites, 10,751 rows, 12,351 decisions, zero optimizer updates, and finite valid predictions. Report the seven-suite unweighted mean, not pooled accuracy. Do not infer a result from startup success. Record checkpoint file hashes and runtime software with the private experiment evidence. Hardware validation is pending until the completed report is available.
