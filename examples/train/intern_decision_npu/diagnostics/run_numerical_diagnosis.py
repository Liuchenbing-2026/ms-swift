"""Run isolated fixed-weight comparisons using an explicit private plan."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    root = Path(plan["host_output"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "state.json").exists():
        raise FileExistsError("Use a fresh diagnostic run directory")
    state = {"status": "running", "stages": {}}
    def persist():
        state["observed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        temporary = root / "state.tmp"
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(root / "state.json")
    persist()
    try:
        for model, checkpoint in plan["models"].items():
            for mode in ("fused-bf16", "reference-bf16", "reference-fp32"):
                name = model + "-" + mode
                command = ["docker", "exec", "-w", plan["source"], plan["container"],
                    "timeout", "--signal=TERM", "--kill-after=30s", "3600", plan["python"],
                    "-u", plan["script"], "--checkpoint", checkpoint,
                    "--data", plan["data"], "--mode", mode,
                    "--output", plan["container_output"] + "/" + name + ".jsonl"]
                state["stage"] = name
                state["stages"][name] = {"command": command}
                persist()
                with (root / (name + ".log")).open("x") as log:
                    p = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    state["child_pid"] = p.pid
                    persist()
                    code = p.wait()
                state["stages"][name]["returncode"] = code
                persist()
                if code:
                    raise RuntimeError(name + " failed; no overlapping job launched")
        comparisons = {}
        for model in plan["models"]:
            audit, predictions, metrics = {}, {}, {}
            for mode in ("fused-bf16", "reference-bf16", "reference-fp32"):
                path = root / (model + "-" + mode + ".jsonl")
                audit[mode] = json.loads(Path(str(path) + ".audit.json").read_text())
                predictions[mode] = {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}
                metrics[mode] = json.loads(Path(str(path) + ".metrics.json").read_text())
                assert audit[mode]["status"] == "complete"
                assert metrics[mode]["rows"] == plan["expected_rows"]
            assert len({a["input_sha256"] for a in audit.values()}) == 1
            assert len({a["data_sha256"] for a in audit.values()}) == 1
            baseline = predictions["fused-bf16"]
            comparisons[model] = {"metrics": metrics, "paired": {}}
            for mode in ("reference-bf16", "reference-fp32"):
                other = predictions[mode]
                assert baseline.keys() == other.keys()
                changes = []
                for key, row in baseline.items():
                    assert row["scores"].keys() == other[key]["scores"].keys()
                    for field, score in row["scores"].items():
                        alternate = other[key]["scores"][field]
                        changes.append({"id": key, "field": field, "before": score["correct"],
                                        "after": alternate["correct"]})
                comparisons[model]["paired"][mode] = {
                    "wrong_to_right": sum(not r["before"] and r["after"] for r in changes),
                    "right_to_wrong": sum(r["before"] and not r["after"] for r in changes),
                    "decisions": len(changes)}
        (root / "comparison.json").write_text(json.dumps(comparisons, indent=2))
        state["status"] = "complete"
    except Exception as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        persist()
        # The plan must name a container created exclusively for this experiment.
        if plan.get("stop_exclusive_container", False):
            subprocess.run(["docker", "stop", plan["container"]], check=True)


if __name__ == "__main__":
    main()
