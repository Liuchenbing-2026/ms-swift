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
    def test_handoff_does_not_adopt_live_controller(self):
        with patch.object(tune_microbatch, "identity", return_value="same"):
            with self.assertRaises(RuntimeError):
                tune_microbatch.wait_for_handoff({"handoff": {
                    "controller_pid": 1, "controller_identity": "same"}}, lambda **kw: None)

    def test_reuse_requires_matching_checkpoint_and_stopped_complete_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "state.json"
            source.write_text(json.dumps({"checkpoint_inventory": {"weight": [5, 10]}}))
            for rank in range(4):
                (root / f"throughput-rank{rank}.jsonl").write_text("fixture")
            log = root / "logging.jsonl"
            log.write_text(json.dumps({"train_runtime": 10, "global_step/max_steps": "304/600"}))
            spec = {"host_output": str(root), "container_output": "/probe", "controller_state": str(source)}
            with patch.object(tune_microbatch, "output_processes", return_value=[]) as active, \
                    patch.object(tune_microbatch, "summarize", return_value={}):
                got = tune_microbatch.reuse_trial(spec, 300, {"weight": (5, 10)})
                self.assertEqual(len(got["reused_evidence"]), 5)
                with self.assertRaises(ValueError):
                    tune_microbatch.reuse_trial(spec, 300, {"weight": (6, 10)})
                active.return_value = [100]
                with self.assertRaises(RuntimeError):
                    tune_microbatch.reuse_trial(spec, 300, {"weight": (5, 10)})
                active.return_value = []
                log.write_text("{}")
                with self.assertRaises(ValueError):
                    tune_microbatch.reuse_trial(spec, 300, {"weight": (5, 10)})

    def test_selected_recompute_configuration_reaches_continuation(self):
        self.check_owned_run(False, saved=True, recompute=True)

    def test_idle_resume_rejects_busy_unknown_and_owned_workers(self):
        plan = {"owned_pids": {"123": "old"}, "previous_outputs": ["/old"],
                "continuation_output": "/continue", "physical_devices": [0, 1, 2, 3]}
        free = "\n".join(f"| No running processes found in NPU {i} |" for i in range(4))
        with patch.object(tune_microbatch, "identity", return_value=None), \
                patch.object(tune_microbatch, "output_processes", return_value=[]), \
                patch.object(tune_microbatch.subprocess, "check_output", return_value=free) as report:
            tune_microbatch.check_idle_resume(plan)
            report.return_value = free.replace("NPU 3", "NPU 30")
            with self.assertRaises(RuntimeError):
                tune_microbatch.check_idle_resume(plan)
            report.return_value = free
            with patch.object(tune_microbatch, "identity", return_value="old"):
                with self.assertRaises(RuntimeError):
                    tune_microbatch.check_idle_resume(plan)
            with patch.object(tune_microbatch, "output_processes", return_value=[123]):
                with self.assertRaises(RuntimeError):
                    tune_microbatch.check_idle_resume(plan)

    def test_saved_checkpoint_resume(self):
        self.check_owned_run(False, saved=True)

    def test_fallback_failure_is_not_reported_as_running(self):
        self.check_owned_run(False, saved=True, fail=True)

    def test_owned_checkpoint_takeover_and_selection(self):
        self.check_owned_run(False)

    def test_evaluation_precedes_trials_without_updates(self):
        self.check_owned_run(True)

    def check_owned_run(self, evaluate, saved=False, fail=False, recompute=False):
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
            if recompute:
                plan["no_recompute_config"] = "/no-recompute.json"
            if saved:
                plan.update(resume_saved_checkpoint=True, previous_outputs=["/old"],
                            physical_devices=[0, 1, 2, 3])
            if evaluate:
                plan["evaluation_config"] = "/evaluation.json"
                for name, step in (("eval-initial", 0), ("eval-checkpoint", 300)):
                    folder = root / "results" / name
                    folder.mkdir(parents=True)
                    report = {"status": "complete", "checkpoint_step": step,
                              "optimizer_updates_performed": 0,
                              "suites": {"validation": {"correct": 1, "total": 2,
                                  "accuracy": 0.5, "role": "validation",
                                  "data_sha256": "same", "input_hash": "same"}}}
                    (folder / "evaluation.json").write_text(json.dumps(report))
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            alive, commands = [not saved], []

            class Completed:
                pid = 456

                def __init__(self, command, **kwargs):
                    commands.append(command)

                def wait(self):
                    return 1 if fail else 0

            def summary(folder, step):
                batch = {"baseline": 1, "batch2": 2, "batch4": 4, "no-recompute": 8}[folder.name]
                return {"median_seconds": {1: 10, 2: 6, 4: 8, 8: 4}[batch],
                        "samples": [["same"]], "losses": [1], "gradient_norms": [2]}

            def stop(pid, sig):
                self.assertEqual(pid, 123)
                alive[0] = False

            with patch("sys.argv", ["tune_microbatch.py", "--plan", str(path)]), \
                    patch.object(tune_microbatch, "identity", lambda pid: "identity" if alive[0] else None), \
                    patch.object(tune_microbatch, "output_processes", lambda output: []), \
                    patch.object(tune_microbatch, "summarize", summary), \
                    patch.object(tune_microbatch.subprocess, "Popen", Completed), \
                    patch.object(tune_microbatch.subprocess, "check_output", return_value="\n".join(
                        f"| No running processes found in NPU {i} |" for i in range(4))), \
                    patch.object(tune_microbatch.os, "kill", stop), \
                    patch.object(tune_microbatch.time, "sleep", lambda seconds: None), \
                    patch.object(Path, "read_bytes", return_value=b"torch.distributed.run\0--master_port\0" + b"29651"):
                if fail:
                    with self.assertRaises(RuntimeError):
                        tune_microbatch.main()
                else:
                    tune_microbatch.main()
            result = json.loads((root / "results/state.json").read_text())
            if fail:
                self.assertEqual(result["status"], "failed")
                self.assertIn("fallback-continuation exited", result["fallback_error"])
                self.assertEqual(len(commands), 2)
                return
            self.assertEqual(result["selected_microbatch"], 2)
            self.assertEqual(result["status"], "training_finished_pending_independent_evaluation")
            self.assertEqual(len(commands), (6 if evaluate else 4) + int(recompute))
            if recompute:
                self.assertEqual(result["selected_fsdp_config"], "/no-recompute.json")
                self.assertIn("FSDP_CONFIG=/no-recompute.json", commands[-1])
            if evaluate:
                self.assertIn("RESUME_FROM=", commands[0])
                self.assertIn("RESUME_FROM=/checkpoint", commands[1])
                for command in commands[:2]:
                    self.assertIn("DECISION_EVAL_ONLY=1", command)
                    self.assertIn("SAVE_STRATEGY=no", command)
                self.assertEqual(result["evaluation_comparison"]["validation"]["accuracy_delta"], 0)
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
