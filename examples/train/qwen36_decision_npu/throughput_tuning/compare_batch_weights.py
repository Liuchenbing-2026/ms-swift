"""Compare small-model resumed updates; not an accuracy benchmark."""
import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


def load(path):
    reader = dcp.FileSystemReader(Path(path) / "pytorch_model_fsdp_0")
    metadata = reader.read_metadata()
    state = {key: torch.empty(value.size, dtype=value.properties.dtype)
             for key, value in metadata.state_dict_metadata.items()
             if isinstance(value, TensorStorageMetadata)}
    dcp.load(state, storage_reader=reader)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    initial, baseline, candidate = map(load, (args.initial, args.baseline, args.candidate))
    assert initial.keys() == baseline.keys() == candidate.keys()
    squared_error = squared_update = 0.0
    max_abs = 0.0
    for key, base in baseline.items():
        other, old = candidate[key], initial[key]
        assert torch.isfinite(base).all() and torch.isfinite(other).all()
        error = (other.float() - base.float()).double()
        update = (base.float() - old.float()).double()
        squared_error += float(error.square().sum())
        squared_update += float(update.square().sum())
        if error.numel():
            max_abs = max(max_abs, float(error.abs().max()))
    assert squared_update > 0, "No measurable optimization update"
    ratio = (squared_error / squared_update) ** 0.5
    result = {"tensor_count": len(baseline), "max_abs": max_abs,
              "update_relative_l2": ratio, "threshold": 0.10,
              "passed": ratio <= 0.10,
              "scope": "Small model, four resumed updates; final quality unverified"}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    assert result["passed"], "Candidate weight update differs materially"


if __name__ == "__main__":
    main()
