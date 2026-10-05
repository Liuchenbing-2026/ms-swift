"""Reclaim managed orphan shards only after Trainer rotates their checkpoint.

The current checkpoint is never removed. New checkpoint shards are moved using
the previously verified hash-before-link helper, outside throughput trials.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from relocate_checkpoint_files import persist, relocate


def latest_training_update(plan):
    if not plan.get("training_log"):
        return json.loads(Path(plan["progress"]).read_text())["latest_training_update"]
    # Read the live continuation log rather than a terminated supervisor's snapshot.
    for line in reversed(Path(plan["training_log"]).read_text().splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # The writer may still be appending the last record.
        if all(key in row for key in ("global_step/max_steps", "loss", "grad_norm")):
            return row
    raise ValueError("No complete training update in the continuation log")


def complete_checkpoint(path, step):
    path = Path(path)
    if json.loads((path / "trainer_state.json").read_text())["global_step"] != step:
        raise ValueError("Checkpoint step mismatch")
    names = ["scheduler.pt", "trainer_state.json"] + [f"rng_state_{r}.pth" for r in range(4)]
    for group in ("optimizer_0", "pytorch_model_fsdp_0"):
        names += [f"{group}/.metadata"] + [f"{group}/__{r}_0.distcp" for r in range(4)]
    return {name: (path / name).stat().st_size for name in names
            if (path / name).stat().st_size > 0}


def reclaim_orphans(manifest, training_root, spill_root, state_path):
    """Delete only unreferenced spill copies of already removed checkpoints."""
    training_root, spill_root = Path(training_root).absolute(), Path(spill_root).absolute()
    record = json.loads(Path(manifest).read_text())
    if record["status"] != "complete" or not record["files"]:
        raise ValueError("A complete relocation manifest is required")
    candidates = []
    for row in record["files"]:
        source, destination = Path(row["source"]), Path(row["destination"])
        relative = source.relative_to(training_root)
        if len(relative.parts) != 3 or not relative.parts[0].startswith("checkpoint-"):
            raise ValueError("Unexpected source layout")
        if not relative.parts[0][11:].isdigit():
            raise ValueError("Unexpected checkpoint name")
        if destination != spill_root / relative or not row["copy_verified"] or not row["link_installed"]:
            raise ValueError("Destination is not a verified managed copy")
        # Directory removal is done by Trainer's configured retention policy,
        # never by this helper. A retained checkpoint makes cleanup ineligible.
        old_checkpoint = training_root / relative.parts[0]
        if old_checkpoint.exists() or old_checkpoint.is_symlink():
            raise ValueError("Trainer still retains the old checkpoint")
        if destination.is_symlink():
            raise ValueError("Spill file unexpectedly became a symlink")
        if destination.exists():
            stat = destination.stat()
            if stat.st_size != row["bytes"] or stat.st_nlink != 1:
                raise ValueError("Managed copy size or hardlink count changed")
            if destination.resolve() != destination:
                raise ValueError("Spill path has unexpected symlink parents")
        candidates.append(destination)
    for path in training_root.parent.rglob("*"):
        if path.is_symlink() and path.resolve() in candidates:
            raise ValueError("Another retained output references an old spill file")
    report = {"status": "reclaiming", "manifest": str(manifest), "removed": []}
    persist(state_path, report)
    for destination in candidates:
        if destination.exists():
            destination.unlink()
            fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        report["removed"].append(str(destination))
        persist(state_path, report)
    report["status"] = "complete"
    persist(state_path, report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    training = Path(plan["training_root"])
    spill = Path(plan["spill_root"])
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "state.json"
    if state_path.exists():
        raise ValueError("State already exists; inspect interrupted actions before restarting")
    if training.stat().st_dev == spill.stat().st_dev:
        raise ValueError("Spill storage must be mounted on a different filesystem")
    manifest = Path(plan["initial_manifest"])
    state = {"status": "waiting", "completed_steps": []}
    deadline = time.monotonic() + plan.get("timeout_seconds", 259200)
    try:
        for step in plan["steps"]:
            checkpoint = training / f"checkpoint-{step}"
            while time.monotonic() < deadline:
                tuning = json.loads(Path(plan["tuning_state"]).read_text())
                latest = latest_training_update(plan)
                logged = int(latest["global_step/max_steps"].split("/")[0])
                phase = tuning["status"]
                state.update(status="waiting", target_step=step, logged_step=logged,
                             tuning_phase=phase,
                             observed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                persist(state_path, state)
                final = step == plan["steps"][-1]
                eligible = (logged > step and phase == "continuing") or (
                    final and logged >= step and phase == "training_finished_pending_independent_evaluation")
                if eligible:
                    try:
                        inventory = complete_checkpoint(checkpoint, step)
                        if len(inventory) != 16:
                            raise ValueError("Incomplete checkpoint")
                        time.sleep(15)
                        if complete_checkpoint(checkpoint, step) == inventory:
                            break
                    except (FileNotFoundError, ValueError, json.JSONDecodeError):
                        pass
                time.sleep(30)
            else:
                raise TimeoutError("Checkpoint rotation deadline exceeded")
            # A resumed training update must be finite before reclaiming any
            # old spill copies. This is not a full final-quality evaluation.
            import math
            if not all(math.isfinite(float(latest[k])) for k in ("loss", "grad_norm")):
                raise ValueError("Nonfinite resumed update")
            state.update(status="reclaiming", target_step=step)
            persist(state_path, state)
            reclaim_orphans(manifest, training, spill, output / f"reclaim-before-{step}.json")
            if not final:
                manifest = output / f"relocation-{step}.json"
                state["status"] = "relocating"
                persist(state_path, state)
                relocate(checkpoint, spill, manifest, plan["files"], int(plan["reserve_gib"] * 1024**3))
            state["completed_steps"].append(step)
            state.update(training_free_bytes=shutil.disk_usage(training).free,
                         spill_free_bytes=shutil.disk_usage(spill).free)
            persist(state_path, state)
        state["status"] = "complete"
        persist(state_path, state)
    except Exception as error:
        state.update(status="failed", error=str(error))
        persist(state_path, state)
        raise


if __name__ == "__main__":
    main()
