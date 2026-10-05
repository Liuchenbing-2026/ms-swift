"""Wait for prior acceptance, then run one fixed validation-only training trial."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def main():
    p = argparse.ArgumentParser();p.add_argument('--plan', type=Path, required=True);a = p.parse_args()
    plan = json.loads(a.plan.read_text());root = Path(plan['host_workspace'])
    state_path = root / 'state.json'
    if state_path.exists():raise FileExistsError('Do not launch a duplicate trial')
    state = {'status': 'waiting_for_prior_acceptance', 'stages': {}, 'test_used_for_selection': False}
    started = False
    def persist():
        state['observed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        temp = root / 'state.tmp';temp.write_text(json.dumps(state, indent=2));temp.replace(state_path)
    def run(name, command, env=None):
        cmd = ['docker', 'exec', '-w', '/workspace']
        for key, value in {**plan['environment'], **(env or {})}.items():cmd += ['-e', key + '=' + str(value)]
        cmd += [plan['container'], 'timeout', '--signal=TERM', '--kill-after=60s', '28800'] + command
        state.update(status='running', stage=name);state['stages'][name] = {'command': cmd};persist()
        with (root / (name + '.log')).open('x') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            state['child_pid'] = proc.pid;persist();rc = proc.wait()
        state['stages'][name]['returncode'] = rc;persist()
        if rc:raise RuntimeError(name + ' failed; preserve evidence')
    persist()
    try:
        deadline = time.monotonic() + plan.get('wait_seconds', 14400)
        while True:
            prior = json.loads(Path(plan['prior_acceptance']).read_text())
            if prior['status'] == 'failed':raise RuntimeError('Prior acceptance failed; do not hide it with new training')
            if prior['status'] == 'complete':break
            if time.monotonic() > deadline:raise TimeoutError('Prior acceptance did not finish')
            time.sleep(30)
        for path, expected in plan['pinned_files'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, 'Pinned input/code changed: ' + path
        weight_bytes = sum(Path(path).stat().st_size for path in plan['checkpoint_shards'])
        assert shutil.disk_usage(root).free >= weight_bytes + plan['reserve_gib'] * 1024**3, 'Insufficient reserved export space'
        subprocess.run(plan['free_device_check'], check=True)
        running = subprocess.check_output(['docker', 'inspect', '-f', '{{.State.Running}}', plan['container']], text=True).strip()
        assert running == 'false', 'Container already running'
        subprocess.run(['docker', 'start', plan['container']], check=True);started = True
        run('token-audit', ['python', '/workspace/audit_mix_tokens.py', '--data', '/workspace/data', '--tokenizer', plan['checkpoint']])
        if plan.get('smoke_checkpoint'):
            run('smoke-training', ['bash', '/workspace/run_multitask.sh'], {
                'MODEL_PATH': plan['smoke_checkpoint'], 'OUTPUT_DIR': '/workspace/smoke', 'TOTAL_STEPS': 2,
                'MULTITASK_DATA': '/workspace/data', 'TRAIN_WORKSPACE': '/workspace'})
            smoke = json.loads((root / 'smoke/export.json').read_text());assert smoke['step'] == 2
            run('smoke-reload', ['python', '/workspace/multitask_validation.py', '--checkpoint', smoke['checkpoint'],
                                '--data-dir', '/workspace/data', '--output', '/workspace/smoke-reload'])
            for source in ('business', 'tool', 'safety'):
                memory = json.loads((root / 'smoke' / ('validation-after-' + source + '.json')).read_text())
                loaded = json.loads((root / 'smoke-reload' / ('validation-' + source + '.json')).read_text())
                assert memory['data_sha256'] == loaded['data_sha256']
                for mode in ('joint', 'single'):
                    left, right = memory[mode]['predictions'], loaded[mode]['predictions']
                    assert len(left) == len(right)
                    assert all(all(x[k] == y[k] for k in ('case_id', 'field', 'input_hash', 'prediction'))
                               for x, y in zip(left, right)), 'Small-model reload mismatch'
        run('baseline', ['python', '/workspace/multitask_validation.py', '--checkpoint', plan['checkpoint'],
                         '--data-dir', '/workspace/data', '--output', '/workspace/baseline'])
        run('training', ['bash', '/workspace/run_multitask.sh'], {
            'MODEL_PATH': plan['checkpoint'], 'OUTPUT_DIR': '/workspace/experiment', 'TOTAL_STEPS': plan['steps'],
            'MULTITASK_DATA': '/workspace/data', 'TRAIN_WORKSPACE': '/workspace'})
        export = json.loads((root / 'experiment/export.json').read_text());assert export['step'] == plan['steps']
        run('reload', ['python', '/workspace/multitask_validation.py', '--checkpoint', export['checkpoint'],
                       '--data-dir', '/workspace/data', '--output', '/workspace/reload'])
        comparisons = {}
        for source in ('business', 'tool', 'safety'):
            before = json.loads((root / 'baseline' / ('validation-' + source + '.json')).read_text())
            after = json.loads((root / 'reload' / ('validation-' + source + '.json')).read_text())
            memory = json.loads((root / 'experiment' / ('validation-after-' + source + '.json')).read_text())
            assert before['data_sha256'] == after['data_sha256'] == memory['data_sha256']
            comparisons[source] = {}
            for mode in ('joint', 'single'):
                assert before[mode]['total'] == after[mode]['total'] == memory[mode]['total']
                for old, new, saved in zip(before[mode]['predictions'], after[mode]['predictions'], memory[mode]['predictions']):
                    assert all(old[k] == new[k] == saved[k] for k in ('case_id', 'field', 'gold', 'input_hash'))
                    assert new['prediction'] == saved['prediction'], 'Reload changed decisions'
                comparisons[source][mode] = {'before_correct': before[mode]['correct'], 'after_correct': after[mode]['correct'], 'total': after[mode]['total']}
        (root / 'comparison.json').write_text(json.dumps(comparisons, indent=2))
        state.update(status='validation_complete', limitation='Fixed data intervention; independent final acceptance still required')
    except Exception as error:
        state.update(status='failed', error=str(error));raise
    finally:
        persist()
        if started:subprocess.run(['docker', 'stop', '--timeout', '30', plan['container']], check=True)


if __name__ == '__main__':
    main()
