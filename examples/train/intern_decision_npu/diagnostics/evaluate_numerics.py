"""Fixed-weight, fixed-input NPU diagnostic; no training or calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch_npu
from transformers.models.qwen3_5 import modeling_qwen3_5 as architecture
from src.inference.engine import DecisionEngine
from src.eval.jev import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["fused-bf16", "reference-bf16", "reference-fp32"], required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or Path(str(output) + ".audit.json").exists():
        raise FileExistsError("Use a fresh output path")
    torch.manual_seed(42)
    torch.npu.set_device("npu:0")
    engine = DecisionEngine(args.checkpoint, backend="npu" if args.mode == "fused-bf16" else "hf",
        device="npu:0", dtype="float32" if args.mode == "reference-fp32" else "bfloat16",
        attn_implementation="sdpa", max_length=8192)
    layers = 0
    if args.mode != "fused-bf16":
        for module in engine.backend.model.modules():
            if isinstance(module, architecture.Qwen3_5GatedDeltaNet):
                module.chunk_gated_delta_rule = architecture.torch_chunk_gated_delta_rule
                layers += 1
        if not layers:
            raise ValueError("No reference GDN layers found")
    engine.backend_name = args.mode
    engine.backend.synchronize = lambda: torch.npu.synchronize()
    # Hash exact label-free token inputs before inference; never truncate.
    from src.eval.jev import _canonical_public
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    inputs = []
    for row in rows:
        _, batch, positions = engine.backend.encode(_canonical_public(row))
        inputs.append({"id": row["id"], "tokens": batch["input_ids"].tolist(), "positions": positions.tolist()})
    audit = {"mode": args.mode, "rows": len(rows), "batch_size": 1,
        "input_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "reference_layers": layers, "temperature": 1, "status": "running",
        "scope": "Inference numerical diagnostic; not a training-gradient equivalence test"}
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path = Path(str(output) + ".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2))
    evaluate(engine, args.data, output, batch_size=1)
    metrics = json.loads(Path(str(output) + ".metrics.json").read_text())
    if metrics["rows"] != len(rows) or metrics["total"] != len(rows):
        raise ValueError("Expected one decision per row with full coverage")
    audit["status"] = "complete"
    audit_path.write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
