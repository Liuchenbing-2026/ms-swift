"""Bounded, checkpoint-resumed training measurements for decision fine-tuning.

Load as a SWIFT external plugin and select --callbacks decision_throughput.
This plugin does not change the loss, model, optimizer, or sampler.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from types import MethodType

import torch
from transformers import TrainerCallback
from swift.callbacks import callbacks_map


class DecisionThroughput(TrainerCallback):
    def __init__(self, args, trainer):
        self.trainer = trainer
        self.limit = int(os.environ.get("DECISION_PROBE_UPDATES", "0"))
        self.audit = os.environ.get("DECISION_PROBE_AUDIT", "1") == "1"
        self.root = Path(args.output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.rank = int(os.environ.get("RANK", "0"))
        self.samples = []
        self.original_step = trainer.training_step
        owner = self

        def measured_step(instance, model, inputs, *positional, **keywords):
            if owner.audit:
                ids = inputs["input_ids"].detach().cpu()
                labels = inputs["labels"].detach().cpu()
                mask = inputs.get("attention_mask")
                mask = mask.detach().cpu() if mask is not None else None
                for i in range(ids.shape[0]):
                    keep = mask[i].bool() if mask is not None else torch.ones_like(ids[i], dtype=torch.bool)
                    record = [ids[i][keep].tolist(), labels[i][keep].tolist()]
                    owner.samples.append(hashlib.sha256(json.dumps(record).encode()).hexdigest())
            return owner.original_step(model, inputs, *positional, **keywords)

        trainer.training_step = MethodType(measured_step, trainer)

    def write(self, record):
        with (self.root / f"throughput-rank{self.rank}.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_step = state.global_step
        self.write({"event": "begin", "step": self.start_step,
                    "microbatch": args.per_device_train_batch_size,
                    "accumulation": args.gradient_accumulation_steps,
                    "max_steps": state.max_steps,
                    "world_size": args.world_size})

    def on_step_begin(self, args, state, control, **kwargs):
        torch.npu.synchronize()
        self.started = time.monotonic()
        self.samples = []
        torch.npu.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs):
        torch.npu.synchronize()
        self.write({"event": "update", "step": state.global_step,
                    "seconds": time.monotonic() - self.started,
                    "samples": self.samples,
                    "peak_allocated_bytes": torch.npu.max_memory_allocated(),
                    "peak_reserved_bytes": torch.npu.max_memory_reserved()})
        if self.limit and state.global_step >= self.start_step + self.limit:
            control.should_training_stop = True
            # Do not force a save: a probe must not overwrite its resume point.
        return control


callbacks_map["decision_throughput"] = DecisionThroughput
