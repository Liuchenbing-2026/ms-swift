"""Run a validation-only joint-field experiment, preserving prior weights."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    root = Path(plan['host_workspace'])
    if (root / 'pipeline-state.json').exists():
        raise FileExistsError('Do not start duplicate pipelines')
    state = {'status': 'running', 'stages': {}}

    def persist():
        state['observed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        temporary = root / 'pipeline-state.tmp'
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(root / 'pipeline-state.json')

    def run(name, command, environment=None):
        cmd = ['docker', 'exec', '-w', '/workspace']
        for key, value in (environment or {}).items():
            cmd += ['-e', key + '=' + str(value)]
        cmd += [plan['container'], 'timeout', '--signal=TERM', '--kill-after=60s', '28800'] + command
        state['stage'] = name
        state['stages'][name] = {'command': cmd}
        persist()
        with (root / (name + '.log')).open('x') as stream:
            process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            state['child_pid'] = process.pid
            persist()
            code = process.wait()
        state['stages'][name]['returncode'] = code
        persist()
        if code:
            raise RuntimeError(name + ' failed; inspect preserved logs')

    try:
        assert shutil.disk_usage(root).free >= 14 * 1024**3, 'Final BF16 artifact needs reserved disk space'
        run('baseline', ['python', '/workspace/joint_validation.py', '--checkpoint', plan['checkpoint'],
            '--data', plan['validation_data'], '--output', '/workspace/baseline.json'])
        environment = {'TRAIN_WORKSPACE': '/workspace', 'MODEL_PATH': plan['checkpoint'],
            'OUTPUT_DIR': '/workspace/experiment', 'TOTAL_STEPS': str(plan['steps']), 'ACCUMULATION': '4',
            'JOINT_VALIDATION_DATA': plan['validation_data'], 'TRITON_CACHE_DIR': '/workspace/prior-triton-cache'}
        run('training', ['bash', '/workspace/launches/real.sh'], environment)
        export = json.loads((root / 'experiment/export.json').read_text())
        assert export['step'] == plan['steps'] and export['dtype'] == 'bfloat16'
        run('reload-validation', ['python', '/workspace/joint_validation.py', '--checkpoint', export['checkpoint'],
            '--data', plan['validation_data'], '--output', '/workspace/reload-validation.json'])
        before = json.loads((root / 'baseline.json').read_text())
        after = json.loads((root / 'reload-validation.json').read_text())
        memory = json.loads((root / 'experiment/validation-after.json').read_text())
        assert before['data_sha256'] == after['data_sha256'] == memory['data_sha256']
        comparison = {}
        for mode in ('joint', 'single'):
            assert before[mode]['total'] == after[mode]['total'] == memory[mode]['total'] == 600
            keys = ('case_id', 'field', 'gold', 'input_hash')
            for a, b, c in zip(before[mode]['predictions'], after[mode]['predictions'], memory[mode]['predictions']):
                assert all(a[k] == b[k] == c[k] for k in keys)
                assert b['prediction'] == c['prediction'], 'Reload changed decisions'
            comparison[mode] = {'before_correct': before[mode]['correct'], 'after_correct': after[mode]['correct'],
                                'total': 600, 'delta': after[mode]['accuracy'] - before[mode]['accuracy']}
        (root / 'comparison.json').write_text(json.dumps(comparison, indent=2))
        state.update(status='validation_complete', limitation='No test or seven-suite acceptance in this stage')
    except Exception as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        persist()
        subprocess.run(['docker', 'stop', plan['container']], check=True)


if __name__ == '__main__':
    main()
