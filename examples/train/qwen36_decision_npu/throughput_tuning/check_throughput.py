"""CPU checks for trial admission and incomplete checkpoint rejection."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import tune_microbatch
from tune_microbatch import admissible, checkpoint_files, summarize


class TrialChecks(unittest.TestCase):
    def test_owned_checkpoint_takeover_and_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint-300"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text('{"global_step":300}')
            names = ["scheduler.pt"] + [f"rng_state_{r}.pth" for r in range(4)]
            for group in ("pytorch_model_fsdp_0", "optimizer_0"):
                names += [group + "/.metadata"] + [f"{group}/__{r}_0.distcp" for r in range(4)]
            for name in names:
                p = checkpoint / name
                p.parent.mkdir(exist_ok=True)
                p.write_bytes(b"fixture")
            plan = {"host_output": str(root / "results"), "host_checkpoint": str(checkpoint),
                    "checkpoint_step": 300, "launcher_pid": 123,
                    "owned_pids": {"123": "identity"}, "original_master_port": "29651",
                    "container": "fixture", "container_output": "/trial",
                    "container_checkpoint": "/checkpoint", "continuation_output": "/continue",
                    "container_launcher": "/launcher.sh", "environment": {}}
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            alive, commands = [True], []

            class Completed:
                pid = 456

                def __init__(self, command, **kwargs):
                    commands.append(command)

                def wait(self):
                    return 0

            def summary(folder, step):
                batch = {"baseline": 1, "batch2": 2, "batch4": 4}[folder.name]
                return {"median_seconds": {1: 10, 2: 6, 4: 8}[batch],
                        "samples": [["same"]], "losses": [1], "gradient_norms": [2]}

            def stop(pid, sig):
                self.assertEqual(pid, 123)
                alive[0] = False

            with patch("sys.argv", ["tune_microbatch.py", "--plan", str(path)]), \
                    patch.object(tune_microbatch, "identity", lambda pid: "identity" if alive[0] else None), \
                    patch.object(tune_microbatch, "output_processes", lambda output: []), \
                    patch.object(tune_microbatch, "summarize", summary), \
                    patch.object(tune_microbatch.subprocess, "Popen", Completed), \
                    patch.object(tune_microbatch.os, "kill", stop), \
                    patch.object(tune_microbatch.time, "sleep", lambda seconds: None), \
                    patch.object(Path, "read_bytes", return_value=b"torch.distributed.run\0--master_port\0" + b"29651"):
                tune_microbatch.main()
            result = json.loads((root / "results/state.json").read_text())
            self.assertEqual(result["selected_microbatch"], 2)
            self.assertEqual(result["status"], "training_finished_pending_independent_evaluation")
            self.assertEqual(len(commands), 4)
            self.assertIn("MICROBATCH=2", commands[-1])
            self.assertIn("DECISION_PROBE_UPDATES=0", commands[-1])

    def test_numeric_and_sample_gates(self):
        baseline = {"samples": [["a", "b"]], "losses": [1.0], "gradient_norms": [2.0]}
        self.assertTrue(admissible(baseline, deepcopy(baseline))[0])
        for field, value in (("samples", [["a", "c"]]), ("losses", [1.2]),
                             ("gradient_norms", [2.3])):
            candidate = deepcopy(baseline)
            candidate[field] = value
            self.assertFalse(admissible(baseline, candidate)[0])

    def test_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "trainer_state.json").write_text('{"global_step": 300}')
            with self.assertRaises(FileNotFoundError):
                checkpoint_files(path, 300)
            with self.assertRaises(ValueError):
                checkpoint_files(path, 150)

    def test_all_ranks_warmup_and_finite_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for rank in range(4):
                rows = [{"event": "begin", "step": 300, "microbatch": 2,
                         "accumulation": 2, "world_size": 4}]
                rows += [{"event": "update", "step": 301 + i,
                          "seconds": (999 if i == 0 else 10 + i) + rank,
                          "samples": [str(rank * 4 + j) for j in range(4)],
                          "peak_allocated_bytes": 1024} for i in range(4)]
                (path / f"throughput-rank{rank}.jsonl").write_text("\n".join(map(json.dumps, rows)))
            metrics = [{"global_step/max_steps": f"{301+i}/600", "loss": 1,
                        "grad_norm": 2} for i in range(4)]
            log = path / "logging.jsonl"
            log.write_text("\n".join(map(json.dumps, metrics)))
            self.assertEqual(summarize(path, 300)["median_seconds"], 15)
            metrics[-1]["grad_norm"] = float("nan")
            log.write_text("\n".join(map(json.dumps, metrics)))
            with self.assertRaises(AssertionError):
                summarize(path, 300)


if __name__ == "__main__":
    unittest.main()
