"""CPU-only checks for acceptance sequencing and failure isolation."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_acceptance


class AcceptanceTests(unittest.TestCase):
    def execute(self, training_status='validation_complete', busy=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'pipeline-state.json').write_text(json.dumps({'status': training_status}))
            checkpoint = root / 'experiment/checkpoint-120';checkpoint.mkdir(parents=True)
            (checkpoint / 'model.safetensors').write_bytes(b'synthetic-checkpoint')
            (root / 'experiment/export.json').write_text(json.dumps({'step': 120, 'dtype': 'bfloat16'}))
            suites = {'typed_decisions': {'n': 2000}, **{str(i): {'n': 1} for i in range(48)}}
            counts = [(48,48), (72,72), (111,111), (7600,7600), (310,310), (400,2000), (2210,2210)]
            results = {'business': {'decisions': 2000}, 'broad49': {'overall': {'n': 17416}, 'suites': suites},
                       'official7': {'datasets': {str(i): {'rows': r, 'total': t} for i,(r,t) in enumerate(counts)}}}
            stages = []
            for name,result in results.items():
                path = root / (name + '.json');path.write_text(json.dumps(result))
                stages.append({'name': name, 'container': 'owned-evaluation', 'command': ['evaluate', name], 'result': str(path)})
            plan = {'host_workspace': str(root), 'training_container': 'owned-training', 'checkpoint_step': 120,
                    'free_device_check': ['free-device'], 'stages': stages}
            path = root / 'plan.json';path.write_text(json.dumps(plan))
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=int(busy and command == ['free-device']), stderr='busy' if busy else '')
            process = SimpleNamespace(pid=17, wait=lambda **kw: 0)
            with patch.object(sys, 'argv', ['acceptance', '--plan', str(path)]), \
                 patch.object(run_acceptance.subprocess, 'check_output', return_value='false\n'), \
                 patch.object(run_acceptance.subprocess, 'run', side_effect=run), \
                 patch.object(run_acceptance.subprocess, 'Popen', return_value=process) as popen:
                if busy or training_status == 'failed':
                    with self.assertRaises(RuntimeError):
                        run_acceptance.main()
                else:
                    run_acceptance.main()
                state = json.loads((root / 'final-evaluation/state.json').read_text())
                return state, calls, popen.call_count

    def test_completed_fixed_checkpoint_runs_all_suites(self):
        state, calls, launched = self.execute()
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(launched, 3)
        self.assertEqual(len(state['checkpoint_sha256']), 1)
        self.assertFalse(state['test_used_for_selection'])

    def test_failed_training_does_not_start_evaluation(self):
        state, calls, launched = self.execute('failed')
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(launched, 0)
        self.assertEqual(calls, [])

    def test_occupied_device_does_not_stop_foreign_work(self):
        state, calls, launched = self.execute(busy=True)
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(launched, 0)
        self.assertEqual(calls, [['free-device']])


if __name__ == '__main__':
    unittest.main()
