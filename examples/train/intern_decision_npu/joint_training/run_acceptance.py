"""Run predeclared held-out evaluations once a fixed training run completes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path(plan['host_workspace'])
    output = root / 'final-evaluation'
    output.mkdir(exist_ok=False)
    state = {'status': 'waiting_for_fixed_checkpoint', 'stages': {}, 'test_used_for_selection': False}
    owned = set()

    def persist():
        state['observed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        temporary = output / 'state.tmp'
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(output / 'state.json')

    def running(container):
        return subprocess.check_output(['docker', 'inspect', '-f', '{{.State.Running}}', container], text=True).strip() == 'true'

    def run(stage):
        name, container = stage['name'], stage['container']
        for path, expected in plan.get('pinned_files', {}).items():
            assert sha256(Path(path)) == expected, 'Pinned evaluation input or source changed: ' + path
        if running(container):
            raise RuntimeError('Container unexpectedly running before acceptance: ' + container)
        # The device must still be free; never terminate unowned processes.
        probe = subprocess.run(plan['free_device_check'], capture_output=True, text=True, timeout=30)
        if probe.returncode:
            raise RuntimeError('Reserved evaluation device is occupied or probe failed: ' + probe.stderr[-500:])
        subprocess.run(['docker', 'start', container], check=True)
        owned.add(container)
        state.update(status='running', stage=name)
        state['stages'][name] = {'command': stage['command']}
        persist()
        try:
            with (output / (name + '.log')).open('x') as log:
                process = subprocess.Popen(stage['command'], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                state['stages'][name]['pid'] = process.pid;persist()
                try:
                    code = process.wait(timeout=stage.get('timeout_seconds', 14400))
                except subprocess.TimeoutExpired:
                    subprocess.run(['docker', 'stop', '--time', '30', container], check=True)
                    process.wait(timeout=60)
                    raise RuntimeError(name + ' timed out')
            state['stages'][name]['returncode'] = code
            if code:
                raise RuntimeError(name + ' failed; inspect local log')
            result = json.loads(Path(stage['result']).read_text())
            if name == 'business':
                assert result['decisions'] == 2000
            elif name == 'broad49':
                assert result['overall']['n'] == 17416 and len(result['suites']) == 49
                assert result['suites']['typed_decisions']['n'] == 2000
            elif name == 'official7':
                suites = result['datasets']
                assert len(suites) == 7 and sum(x['rows'] for x in suites.values()) == 10751
                assert sum(x['total'] for x in suites.values()) == 12351
            else:
                raise ValueError('Unknown predeclared suite')
            state['stages'][name]['status'] = 'complete'
            persist()
        finally:
            subprocess.run(['docker', 'stop', '--time', '30', container], check=True)
            owned.remove(container)

    persist()
    try:
        deadline = time.monotonic() + plan.get('wait_seconds', 43200)
        while True:
            training = json.loads((root / 'pipeline-state.json').read_text())
            if training['status'] == 'failed':
                raise RuntimeError('Training/validation pipeline failed; do not evaluate a partial checkpoint')
            if training['status'] == 'validation_complete' and not running(plan['training_container']):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Timed out waiting for fixed completed checkpoint')
            time.sleep(15)
        export = json.loads((root / 'experiment/export.json').read_text())
        assert export['step'] == plan['checkpoint_step'] and export['dtype'] == 'bfloat16'
        checkpoint = root / 'experiment' / ('checkpoint-' + str(plan['checkpoint_step']))
        index = checkpoint / 'model.safetensors.index.json'
        if index.exists():
            shards = {checkpoint / name for name in json.loads(index.read_text())['weight_map'].values()}
        else:
            shards = {checkpoint / 'model.safetensors'}
        assert shards and all(p.is_file() and p.stat().st_size > 0 for p in shards)
        state['checkpoint_sha256'] = {p.name: sha256(p) for p in sorted(shards)}
        persist()
        assert [stage['name'] for stage in plan['stages']] == ['business', 'broad49', 'official7']
        for stage in plan['stages']:
            run(stage)
        assert set(state['stages']) == {'business', 'broad49', 'official7'}
        state.update(status='complete', limitation='Fixed checkpoint assessment only; no test-guided training or selection')
    except Exception as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        for container in owned:
            subprocess.run(['docker', 'stop', '--time', '30', container], check=True)
        persist()


if __name__ == '__main__':
    main()
