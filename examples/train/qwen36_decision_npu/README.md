# Qwen3.6-35B-A3B text decision training on Ascend

Development recipe for full language-parameter decision fine-tuning, with the visual encoder frozen. It uses the model's Qwen3.5-MoE architecture, native chat template with thinking disabled, and masked causal cross-entropy at decision positions. This is not a LoRA recipe.

Status: synthetic same-architecture four-NPU forward/backward, CPU offload, checkpoint save and resume have passed. Full-size model validation is in progress; this branch does not claim completed quality or performance acceptance. Numerical experiment results are kept outside the repository.

Mount this directory as `/workspace` and the downloaded model as `/models/Qwen3.6-35B-A3B`. Use an isolated container and an explicitly assigned set of four NPUs. The tested software family is PyTorch/torch_npu 2.10, Transformers 5.15.1, Accelerate 1.14.0 and CANN 9.1. Twinkle additionally uses PEFT 0.19.0 and NumPy 1.26.4 on this stack.

Download public data using `download_data.py`, then run `prepare_data.py` and `prepare_inputs.py`. The case-separated train, validation, calibration and test split is fixed; only train enters optimizer updates. Targets are carried separately from input messages. No model weights or dataset files are included.

FSDP2 CPU offload is enabled because the full parameter and optimizer state must be accounted for, rather than only the active MoE parameters. Confirm sufficient host RAM and checkpoint storage before a full run. Preserve full optimizer/scheduler/RNG state when testing recovery.

SWIFT: `bash run.sh probe-cpu` checks two real-weight updates without saving; `bash run.sh smoke` and `bash run.sh resume` check full checkpoint recovery; `bash run.sh train` is the full training entry. Export `ACCELERATE_USE_FSDP=true` and `FSDP_CPU_RAM_EFFICIENT_LOADING=true` before model construction so the model is not first materialized on a single NPU.


For isolated integration checks, `make_tiny_model.py` creates a small random model. Its results do not establish full-model quality.
