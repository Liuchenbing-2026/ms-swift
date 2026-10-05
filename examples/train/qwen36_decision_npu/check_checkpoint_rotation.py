"""Exercise retention and reference guards using disposable checkpoint files."""
import json
from pathlib import Path
import tempfile
import unittest

from relocate_checkpoint_files import relocate
from rotate_checkpoint_storage import reclaim_orphans, latest_training_update


class RotationChecks(unittest.TestCase):
    def test_live_log_ignores_partial_and_evaluation_rows(self):
        path = self.root / "logging.jsonl"
        expected = {"global_step/max_steps": "301/600", "loss": 1.0, "grad_norm": 2.0}
        path.write_text(json.dumps(expected) + '\n{"eval_loss":1}\n{"partial":')
        self.assertEqual(latest_training_update({"training_log": str(path)}), expected)
        path.write_text('{"eval_loss":1}\n')
        with self.assertRaises(ValueError):
            latest_training_update({"training_log": str(path)})

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.training = self.root / "outputs/train"
        self.checkpoint = self.training / "checkpoint-150"
        self.checkpoint.mkdir(parents=True)
        self.spill = self.root / "spill"
        self.spill.mkdir()
        self.filename = "optimizer_0/__1_0.distcp"
        source = self.checkpoint / self.filename
        source.parent.mkdir()
        source.write_bytes(b"immutable checkpoint fixture")
        self.manifest = self.root / "relocation.json"
        relocate(self.checkpoint, self.spill, self.manifest, [self.filename], 0)
        self.destination = self.spill / self.checkpoint.name / self.filename
        self.report = self.root / "reclaim.json"

    def tearDown(self):
        self.temporary.cleanup()

    def rotate_fixture(self):
        (self.checkpoint / self.filename).unlink()
        (self.checkpoint / "optimizer_0").rmdir()
        self.checkpoint.rmdir()

    def reclaim(self):
        reclaim_orphans(self.manifest, self.training, self.spill, self.report)

    def test_retained_checkpoint_is_never_removed(self):
        with self.assertRaisesRegex(ValueError, "retains"):
            self.reclaim()
        self.assertTrue((self.checkpoint / self.filename).is_file())
        self.assertTrue(self.destination.is_file())

    def test_other_output_reference_blocks_reclaim(self):
        self.rotate_fixture()
        alias = self.training.parent / "retained-alias"
        alias.symlink_to(self.destination)
        with self.assertRaisesRegex(ValueError, "references"):
            self.reclaim()
        self.assertTrue(self.destination.exists())

    def test_changed_managed_file_is_not_deleted(self):
        self.rotate_fixture()
        self.destination.write_bytes(b"unexpected replacement content")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.reclaim()
        self.assertTrue(self.destination.exists())

    def test_reclaims_only_orphan_and_is_idempotent(self):
        self.rotate_fixture()
        unrelated = self.spill / "unrelated"
        unrelated.write_bytes(b"retain")
        self.reclaim()
        self.assertFalse(self.destination.exists())
        self.assertTrue(unrelated.exists())
        self.assertTrue(self.destination.parent.exists())
        self.assertEqual(json.loads(self.report.read_text())["status"], "complete")
        self.reclaim()


if __name__ == "__main__":
    unittest.main()
