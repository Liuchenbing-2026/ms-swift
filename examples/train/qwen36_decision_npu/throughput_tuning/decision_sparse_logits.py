"""Retain only decision-predicting positions across a padded mini-batch.

Cross entropy still uses the full vocabulary. Only positions with no target
in any row are removed, preserving ordinary causal next-token alignment.
Enable explicitly using DECISION_SPARSE_LOGITS=1 as a SWIFT external plugin.
"""
import os

import torch
from torch.nn import functional as F


def prepare_sparse_decisions(inputs):
    labels = inputs["labels"]
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError("Expected padded two-dimensional causal labels")
    if (labels[:, 0] != -100).any():
        raise ValueError("The first input token cannot be a supervised decision")
    target_positions = (labels != -100).any(dim=0)
    if not target_positions.any():
        raise ValueError("No decision targets in batch")
    inputs["labels"] = F.pad(labels[:, target_positions], (1, 0), value=-100)
    if inputs.get("loss_scale") is not None:
        inputs["loss_scale"] = F.pad(inputs["loss_scale"][:, target_positions], (1, 0), value=0)
    # The trailing position predicts an ignored sentinel, just as the existing
    # single-example SWIFT path does. Every other position predicts its label.
    inputs["logits_to_keep"] = F.pad(target_positions[1:], (0, 1), value=True)


def install():
    from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
    from swift.utils import is_mp

    original = Seq2SeqTrainer.prepare_logits_to_keep

    def prepare(self, inputs):
        if (type(self.template).__name__ == "DecisionTemplate"
                and self.template.sequence_parallel_size == 1
                and inputs["labels"].shape[0] > 1 and not is_mp()):
            prepare_sparse_decisions(inputs)
        else:
            original(self, inputs)

    Seq2SeqTrainer.prepare_logits_to_keep = prepare


if os.environ.get("DECISION_SPARSE_LOGITS", "0") == "1":
    install()
