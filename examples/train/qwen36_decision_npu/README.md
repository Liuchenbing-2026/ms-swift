# Qwen3.6-35B-A3B text decision training on Ascend

Development recipe for full language-parameter decision fine-tuning, with the visual encoder frozen. It uses the model's Qwen3.5-MoE architecture, native chat template with thinking disabled, and masked causal cross-entropy at decision positions. This is not a LoRA recipe.

Status: synthetic same-architecture four-NPU forward/backward, CPU offload, checkpoint save and resume have passed. Full-size model validation is in progress; this branch does not claim completed quality or performance acceptance. Numerical experiment results are kept outside the repository.

Mount this directory as `/workspace` and the downloaded model as `/models/Qwen3.6-35B-A3B`. Use an isolated container and an explicitly assigned set of four NPUs. The tested software family is PyTorch/torch_npu 2.10, Transformers 5.15.1, Accelerate 1.14.0 and CANN 9.1. The recipe uses NumPy 1.26.4 on this stack.

Download public data using `download_data.py`, then run `prepare_data.py` and `prepare_inputs.py`. The case-separated train, validation, calibration and test split is fixed; only train enters optimizer updates. Targets are carried separately from input messages. No model weights or dataset files are included.

FSDP2 CPU offload is enabled because the full parameter and optimizer state must be accounted for, rather than only the active MoE parameters. Confirm sufficient host RAM and checkpoint storage before a full run. Preserve full optimizer/scheduler/RNG state when testing recovery.

SWIFT: `bash run.sh probe-pageable` checks two real-weight updates without saving. `bash run.sh tiny-sharded` and `bash run.sh tiny-sharded-resume` validate sharded checkpoint recovery using the synthetic model. `bash run.sh smoke` and `bash run.sh resume` exercise full-size checkpoint recovery when storage is available. `bash run.sh train` runs two epochs with periodic validation and sharded checkpoints; the final fixed-step checkpoint is used rather than automatic best-checkpoint retention. Export `ACCELERATE_USE_FSDP=true` and `FSDP_CPU_RAM_EFFICIENT_LOADING=true` before model construction so the model is not first materialized on a single NPU.


For isolated integration checks, `make_tiny_model.py` creates a small random model. Its results do not establish full-model quality.

The large-model initialization exposed two distinct allocation failures before the first training update: direct placement of the entire model on one NPU, and the NPU host allocator rejecting large pinned-memory allocations during CPU-offloaded parameter assignment. Early FSDP environment configuration avoids the first. `cpu_offload.py` selects the supported `CPUOffloadPolicy(pin_memory=False)` for this recipe to address the second. Pageable transfers may be slower; this is an explicit memory tradeoff. FP32 initial loading with `--bf16 true --fp16 false` keeps master parameters in FP32 and compute in BF16, avoiding a second whole-model cast.

The pageable-offload synthetic four-NPU training, sharded save, and resume checks passed. The corresponding full-size run is still under validation. `checkpoint_space.py` waits before writing when the output filesystem lacks room for the estimated complete parameter and Adam state plus a margin. It does not delete checkpoints or guarantee that storage will become available; provision enough space for both an existing checkpoint and its replacement. PyTorch/Accelerate versions are pinned above because checkpoint internals are version sensitive.
