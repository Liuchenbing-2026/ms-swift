import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import run_multitask


class RunnerChecks(unittest.TestCase):
    def test_prior_failure_never_starts_container(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);prior = root / 'prior.json';prior.write_text('{"status":"failed"}')
            plan = root / 'plan.json';plan.write_text(json.dumps({'host_workspace': str(root), 'prior_acceptance': str(prior)}))
            with patch.object(sys, 'argv', ['run', '--plan', str(plan)]), \
                    patch.object(run_multitask.subprocess, 'run') as command:
                with self.assertRaisesRegex(RuntimeError, 'Prior acceptance failed'):
                    run_multitask.main()
                command.assert_not_called()
            self.assertEqual(json.loads((root / 'state.json').read_text())['status'], 'failed')

    def test_duplicate_state_never_starts_container(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);(root / 'state.json').write_text('{"status":"running"}')
            plan = root / 'plan.json';plan.write_text(json.dumps({'host_workspace': str(root)}))
            with patch.object(sys, 'argv', ['run', '--plan', str(plan)]), \
                    patch.object(run_multitask.subprocess, 'run') as command:
                with self.assertRaises(FileExistsError):run_multitask.main()
                command.assert_not_called()


if __name__ == '__main__':
    unittest.main()
