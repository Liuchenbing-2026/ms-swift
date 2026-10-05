"""Check causal full-vocabulary loss and all gradients against dense logits."""
import argparse
import json

import torch
from torch.nn import functional as F
from decision_sparse_logits import prepare_sparse_decisions


def check(device, dtype):
    torch.manual_seed(71)
    hidden = torch.randn(4, 9, 8, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(13, 8, device=device, dtype=dtype, requires_grad=True)
    labels = torch.full((4, 9), -100, device=device, dtype=torch.long)
    # Unequal positions, multiple decisions, final-token label, and an ignored row.
    labels[0, 1], labels[0, 8], labels[1, 4], labels[2, 6] = 2, 3, 5, 7
    dense = F.linear(hidden, weight)
    loss = F.cross_entropy(dense[:, :-1].float().reshape(-1, 13), labels[:, 1:].reshape(-1))
    gradients = torch.autograd.grad(loss, (hidden, weight))
    inputs = {"labels": labels.clone(), "loss_scale": torch.ones_like(labels, dtype=torch.float)}
    prepare_sparse_decisions(inputs)
    kept = F.linear(hidden[:, inputs["logits_to_keep"]], weight)
    reduced_loss = F.cross_entropy(kept[:, :-1].float().reshape(-1, 13), inputs["labels"][:, 1:].reshape(-1))
    reduced_gradients = torch.autograd.grad(reduced_loss, (hidden, weight))
    tolerance = 0.002 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(loss, reduced_loss, rtol=tolerance, atol=tolerance)
    for expected, actual in zip(gradients, reduced_gradients):
        torch.testing.assert_close(expected, actual, rtol=tolerance, atol=tolerance)
    assert (inputs["labels"] != -100).sum() == (labels != -100).sum()
    return {"dtype": str(dtype), "loss_abs_error": float((loss-reduced_loss).detach().abs()),
            "hidden_gradient_max_abs": float((gradients[0]-reduced_gradients[0]).abs().max()),
            "weight_gradient_max_abs": float((gradients[1]-reduced_gradients[1]).abs().max()),
            "dense_positions": 9, "kept_positions": int(inputs["logits_to_keep"].sum())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
        torch.npu.set_device(args.device)
    print(json.dumps([check(args.device, dtype) for dtype in (torch.float32, torch.bfloat16)]))
