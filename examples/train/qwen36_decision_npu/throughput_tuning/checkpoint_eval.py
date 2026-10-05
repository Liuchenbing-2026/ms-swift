"""Read-only candidate-scored evaluation after SWIFT restores FSDP state."""
from collections.abc import Mapping
from datetime import timedelta
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import re

import torch
import torch.distributed as dist
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import TrainOutput
from swift.callbacks import callbacks_map


class EvaluationComplete(Exception):
    """Exit the prepared training loop before its first optimizer update."""


def candidates(row):
    if len(row["decision_targets"]) != 1:
        raise ValueError("This evaluator requires single-question compiled rows")
    assistant = row["messages"][-1]
    if assistant["role"] != "assistant" or json.loads(assistant["content"]) != {row["field"]: "<decision>"}:
        raise ValueError("Assistant input must contain only the decision skeleton")
    users = [m["content"] for m in row["messages"] if m["role"] == "user"]
    if len(users) != 1 or not isinstance(users[0], str):
        raise ValueError("Expected the text-only decision template")
    schema = users[0].split("## Decision schema\n")
    if len(schema) != 2:
        raise ValueError("Decision schema missing or ambiguous")
    symbols = re.findall(r"^    ([A-Za-z0-9]) = ", schema[1], flags=re.MULTILINE)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Invalid candidate symbols")
    if row["decision_targets"][0] not in symbols:
        raise ValueError("Target is not a legal candidate")
    return symbols


class CheckpointEvaluation(TrainerCallback):
    def __init__(self, args, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        config_path = Path(os.environ["DECISION_EVAL_CONFIG"])
        config = json.loads(config_path.read_text())
        rank, world = dist.get_rank(), dist.get_world_size()
        group = dist.new_group(backend="gloo", timeout=timedelta(minutes=30))
        model, tokenizer = self.trainer.model, self.trainer.template.tokenizer
        model.eval()
        tokenizer.padding_side = "right"
        marker = tokenizer.encode("<decision>", add_special_tokens=False)
        if len(marker) != 1:
            raise ValueError("Decision marker must be one token")
        root = Path(args.output_dir)
        reports = {}
        training_ids = {json.loads(line)["case_id"] for line in Path(config["training_data"]).read_text().splitlines()}
        seen_ids = set()
        for suite in config["suites"]:
            path = Path(suite["data"])
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if len(rows) != suite["expected_rows"]:
                raise ValueError("Unexpected suite size")
            ids = {row["case_id"] for row in rows}
            if ids & (training_ids | seen_ids):
                raise ValueError("Training/validation/test cases overlap")
            seen_ids |= ids
            keys = [(row["case_id"], row["field"]) for row in rows]
            if len(set(keys)) != len(keys):
                raise ValueError("Duplicate evaluation decisions")
            jobs = []
            for row in rows:
                symbols = candidates(row)
                token_ids = [tokenizer.encode(s, add_special_tokens=False) for s in symbols]
                if any(len(ids) != 1 for ids in token_ids):
                    raise ValueError("Answer symbol must be one token")
                ids = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                    add_generation_prompt=False, enable_thinking=False)
                if isinstance(ids, Mapping):
                    ids = ids["input_ids"]
                positions = [i for i, token in enumerate(ids) if token == marker[0]]
                if len(positions) != 1 or positions[0] == 0 or len(ids) > 8192:
                    raise ValueError("Invalid or overlength evaluation prompt")
                jobs.append((ids, positions[0] - 1, symbols, [i[0] for i in token_ids]))
            batch_size = int(config.get("batch_size", 4))
            local = []
            with torch.no_grad():
                for start in range(0, len(jobs), batch_size * world):
                    indices = [start + rank * batch_size + j for j in range(batch_size)]
                    # Equal forward-call counts across FSDP ranks; padded jobs
                    # are discarded and do not count toward coverage.
                    selected = [jobs[i if i < len(jobs) else 0] for i in indices]
                    batch = tokenizer.pad({"input_ids": [j[0] for j in selected]},
                        padding=True, pad_to_multiple_of=128, return_tensors="pt").to(args.device)
                    keep = sorted({j[1] for j in selected})
                    logits = model(**batch, use_cache=False,
                        logits_to_keep=torch.tensor(keep, device=args.device)).logits
                    for offset, index in enumerate(indices):
                        if index >= len(jobs):
                            continue
                        job = selected[offset]
                        scores = logits[offset, keep.index(job[1]), job[3]].float().cpu()
                        if not torch.isfinite(scores).all():
                            raise ValueError("Nonfinite candidate scores")
                        row = rows[index]
                        local.append({"index": index, "case_id": row["case_id"], "field": row["field"],
                            "gold": row["decision_targets"][0], "prediction": job[2][int(scores.argmax())],
                            "symbols": job[2], "probabilities": scores.softmax(-1).tolist(),
                            "input_sha256": hashlib.sha256(json.dumps(job[0]).encode()).hexdigest()})
            name = suite["name"]
            (root / f"{name}-rank{rank}.json").write_text(json.dumps(local))
            dist.monitored_barrier(group=group, timeout=timedelta(minutes=30))
            if rank == 0:
                merged = [row for r in range(world) for row in json.loads((root / f"{name}-rank{r}.json").read_text())]
                if sorted(r["index"] for r in merged) != list(range(len(rows))):
                    raise ValueError("Evaluation coverage mismatch")
                correct = sum(r["gold"] == r["prediction"] for r in merged)
                reports[name] = {"correct": correct, "total": len(rows), "accuracy": correct / len(rows),
                    "role": suite["role"], "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "input_hash": hashlib.sha256(json.dumps([r["input_sha256"] for r in sorted(
                        merged, key=lambda r: r["index"])]).encode()).hexdigest()}
            dist.monitored_barrier(group=group, timeout=timedelta(minutes=30))
        if rank == 0:
            report = {"status": "complete", "checkpoint_step": state.global_step,
                "resume_from": os.environ.get("RESUME_FROM", ""), "suites": reports,
                "optimizer_updates_performed": 0,
                "protocol": "Single question; causal pre-marker logits; candidate argmax; no truncation",
                "test_use": "Report only; never tune or select checkpoints on held-out test"}
            (root / "evaluation.json").write_text(json.dumps(report, indent=2))
        dist.monitored_barrier(group=group, timeout=timedelta(minutes=30))
        raise EvaluationComplete()


if os.environ.get("DECISION_EVAL_ONLY") == "1":
    original_train = Trainer.train

    @wraps(original_train)
    def evaluate_in_prepared_loop(self, *args, **kwargs):
        try:
            return original_train(self, *args, **kwargs)
        except EvaluationComplete:
            return TrainOutput(self.state.global_step, 0.0, {"checkpoint_evaluation_only": True})

    Trainer.train = evaluate_in_prepared_loop
    callbacks_map["decision_checkpoint_eval"] = CheckpointEvaluation
