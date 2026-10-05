"""Compare resumed runs, then continue from the unchanged checkpoint.

Host-side driver. The JSON plan supplies owned process identities and paths.
No process is stopped until the requested checkpoint is complete and stable.
Trial outputs are separate; a failed candidate never replaces the checkpoint.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time


def identity(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def output_processes(output):
    """Find only processes carrying this run's exact output environment."""
    marker = ("OUTPUT_DIR=" + output).encode()
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if marker in (proc / "environ").read_bytes().split(b"\0"):
                found.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return found


def checkpoint_files(path, step):
    path = Path(path)
    state = json.loads((path / "trainer_state.json").read_text())
    if state["global_step"] != step:
        raise ValueError("Unexpected checkpoint step")
    required = ["scheduler.pt", "trainer_state.json"]
    required += [f"rng_state_{rank}.pth" for rank in range(4)]
    for group in ("pytorch_model_fsdp_0", "optimizer_0"):
        required += [f"{group}/.metadata"]
        required += [f"{group}/__{rank}_0.distcp" for rank in range(4)]
    stats = {}
    for name in required:
        st = (path / name).stat()
        if st.st_size == 0:
            raise ValueError("Empty checkpoint file: " + name)
        stats[name] = (st.st_size, st.st_mtime_ns)
    return stats


def summarize(folder, start, updates=4):
    folder = Path(folder)
    ranks = []
    for rank in range(4):
        rows = [json.loads(line) for line in (folder / f"throughput-rank{rank}.jsonl").read_text().splitlines()]
        begins = [r for r in rows if r["event"] == "begin"]
        assert len(begins) == 1 and begins[0]["step"] == start
        assert begins[0]["microbatch"] * begins[0]["accumulation"] * begins[0]["world_size"] == 16
        rows = [r for r in rows if r["event"] == "update"]
        assert [r["step"] for r in rows] == list(range(start + 1, start + updates + 1))
        ranks.append(rows)
    metrics = {int(r["global_step/max_steps"].split("/")[0]): r
               for r in map(json.loads, (folder / "logging.jsonl").read_text().splitlines()) if "loss" in r}
    timings, samples, losses, norms = [], [], [], []
    for index in range(updates):
        timing = max(rows[index]["seconds"] for rows in ranks)
        assert math.isfinite(timing) and timing > 0
        timings.append(timing)
        hashes = [h for rows in ranks for h in rows[index]["samples"]]
        assert len(hashes) == 16
        samples.append(sorted(hashes))
        row = metrics[start + index + 1]
        loss, norm = float(row["loss"]), float(row["grad_norm"])
        assert math.isfinite(loss) and math.isfinite(norm) and norm > 0
        losses.append(loss)
        norms.append(norm)
    return {"seconds": timings, "median_seconds": statistics.median(timings[1:]),
            "samples": samples, "losses": losses, "gradient_norms": norms,
            "peak_allocated_bytes": max(r["peak_allocated_bytes"] for rows in ranks for r in rows)}


def admissible(base, trial):
    if base["samples"] != trial["samples"]:
        return False, "different global samples after resume"
    for key, tolerance in (("losses", 0.05), ("gradient_norms", 0.10)):
        if any(abs(a - b) > max(1e-5, abs(a) * tolerance) for a, b in zip(base[key], trial[key])):
            return False, "numerical diagnostic mismatch: " + key
    return True, "finite, matching samples, bounded numerical differences"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    root = Path(plan["host_output"])
    root.mkdir(parents=True, exist_ok=True)
    status = {"status": "waiting_for_checkpoint", "trials": {}}

    def persist(**fields):
        status.update(fields, observed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        temporary = root / "state.tmp"
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(root / "state.json")

    def run(batch, name, probe=True, evaluation=False, initial=False):
        env = dict(plan["environment"])
        env.update(MICROBATCH=str(batch), RESUME_FROM=plan["container_checkpoint"],
                   OUTPUT_DIR=plan["container_output"] + "/" + name if probe else plan["continuation_output"],
                   DECISION_PROBE_UPDATES="4" if probe else "0",
                   DECISION_PROBE_AUDIT="1" if probe else "0",
                   SAVE_STRATEGY="no" if probe else "steps",
                   EVAL_STRATEGY="no" if probe else "steps",
                   DECISION_EVAL_ONLY="1" if evaluation else "0")
        if evaluation:
            env["DECISION_EVAL_CONFIG"] = plan["evaluation_config"]
        if initial:
            env["RESUME_FROM"] = ""
        # A previous timed-out elastic worker must not overlap a new job.
        for output in status.get("launched_outputs", []):
            if output_processes(output):
                raise RuntimeError("Previous owned trial still has live processes: " + output)
        status.setdefault("launched_outputs", []).append(env["OUTPUT_DIR"])
        command = ["docker", "exec"]
        for key, value in env.items():
            command += ["-e", key + "=" + str(value)]
        command += [plan["container"]]
        if probe:
            command += ["timeout", "--signal=TERM", "--kill-after=60s", "7200"]
        command += ["bash", plan["container_launcher"]]
        phase = "evaluating" if evaluation else "trial" if probe else "continuing"
        persist(status=phase, command=command, microbatch=batch)
        with (root / (name + ".log")).open("x") as handle:
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            persist(child_pid=process.pid)
            rc = process.wait()
        for _ in range(60):
            if not output_processes(env["OUTPUT_DIR"]):
                break
            time.sleep(1)
        else:
            raise RuntimeError("Trial workers remain; refusing any overlapping continuation")
        if rc:
            raise RuntimeError(f"{name} exited {rc}; inspect its log")
        if evaluation:
            report = json.loads((root / name / "evaluation.json").read_text())
            assert report["status"] == "complete" and report["optimizer_updates_performed"] == 0
            assert report["checkpoint_step"] == (0 if initial else plan["checkpoint_step"])
            return report
        if probe:
            return summarize(root / name, plan["checkpoint_step"])

    persist()
    checkpoint = plan["host_checkpoint"]
    original_stopped = False
    try:
        deadline = time.monotonic() + plan.get("wait_seconds", 86400)
        while time.monotonic() < deadline:
            if any(identity(int(pid)) != expected for pid, expected in plan["owned_pids"].items()):
                raise RuntimeError("Original training process identity changed; refusing to take over")
            try:
                before = checkpoint_files(checkpoint, plan["checkpoint_step"])
            except (FileNotFoundError, ValueError, json.JSONDecodeError):
                time.sleep(10)
                continue
            time.sleep(15)
            if checkpoint_files(checkpoint, plan["checkpoint_step"]) == before:
                break
        else:
            raise TimeoutError("Checkpoint deadline exceeded; original training left untouched")
        # Only the explicitly owned torchrun launcher is signalled. torchrun
        # terminates its own workers. No container-wide or device-wide kill.
        launcher = int(plan["launcher_pid"])
        assert identity(launcher) == plan["owned_pids"][str(launcher)]
        command = Path(f"/proc/{launcher}/cmdline").read_bytes()
        assert b"torch.distributed.run" in command and plan["original_master_port"].encode() in command
        persist(status="stopping_original_after_save", checkpoint_inventory=before)
        os.kill(launcher, signal.SIGTERM)
        original_stopped = True
        for _ in range(120):
            if all(identity(int(pid)) != expected for pid, expected in plan["owned_pids"].items()):
                break
            time.sleep(1)
        else:
            original_stopped = False  # Do not launch overlapping jobs.
            raise RuntimeError("Original workers did not exit; no new job launched")
        if plan.get("evaluation_config"):
            status["evaluations"] = {}
            for name, initial in (("eval-initial", True), ("eval-checkpoint", False)):
                status["evaluations"][name] = run(4, name, evaluation=True, initial=initial)
                persist()
            before = status["evaluations"]["eval-initial"]["suites"]
            after = status["evaluations"]["eval-checkpoint"]["suites"]
            assert before.keys() == after.keys()
            for name in before:
                for key in ("total", "role", "data_sha256", "input_hash"):
                    assert before[name][key] == after[name][key], "Evaluation protocols differ"
            status["evaluation_comparison"] = {name: {
                "role": after[name]["role"], "initial_correct": before[name]["correct"],
                "checkpoint_correct": after[name]["correct"], "total": after[name]["total"],
                "accuracy_delta": after[name]["accuracy"] - before[name]["accuracy"]}
                for name in before}
            persist()
        baseline = run(1, "baseline")
        status["trials"]["1"] = baseline
        selected, best = 1, baseline["median_seconds"]
        for batch in (2, 4):
            try:
                trial = run(batch, "batch" + str(batch))
                valid, reason = admissible(baseline, trial)
                trial.update(admissible=valid, reason=reason)
                status["trials"][str(batch)] = trial
                if valid and trial["median_seconds"] < min(best, baseline["median_seconds"] * 0.9):
                    selected, best = batch, trial["median_seconds"]
            except Exception as error:
                status["trials"][str(batch)] = {"error": str(error)}
            persist()
        persist(selected_microbatch=selected, status="comparison_complete",
                limitation="Short-run numerical gates do not replace final accuracy evaluation")
        original_stopped = False  # A continuation error must not trigger a duplicate restart.
        run(selected, "continuation", probe=False)
        persist(status="training_finished_pending_independent_evaluation")
    except Exception as error:
        persist(status="failed", error=str(error))
        if original_stopped:
            persist(status="restoring_original_configuration")
            run(1, "fallback-continuation", probe=False)
            persist(status="training_finished_pending_independent_evaluation")
        else:
            raise


if __name__ == "__main__":
    main()
