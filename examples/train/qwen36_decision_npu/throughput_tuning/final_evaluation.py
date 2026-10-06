"""Wait for fixed-budget training and storage rotation, then assess saved weights."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from tune_microbatch import check_idle_resume, checkpoint_files, output_processes


def ready(training, storage):
    if training['status'] == 'failed' or storage['status'] == 'failed':
        raise RuntimeError('Training or storage failed; preserve its evidence')
    return (training['status'] == 'training_finished_pending_independent_evaluation'
            and storage['status'] == 'complete')


def compare(previous, current):
    assert current['status'] == 'complete' and current['optimizer_updates_performed'] == 0
    assert previous['suites'].keys() == current['suites'].keys()
    result = {}
    for name, after in current['suites'].items():
        before = previous['suites'][name]
        for key in ['total', 'role', 'data_sha256', 'input_hash']:
            assert before[key] == after[key], 'Evaluation protocols changed: ' + key
        result[name] = {'before_correct': before['correct'], 'after_correct': after['correct'],
                        'total': after['total'], 'role': after['role']}
    return result


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--plan', required=True)
    plan = json.loads(Path(parser.parse_args().plan).read_text())
    root = Path(plan['host_output']);root.mkdir(exist_ok=False)
    state = {'status': 'waiting_for_training_and_storage', 'test_used_for_selection': False}
    def persist():
        state['observed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        tmp = root / 'state.tmp';tmp.write_text(json.dumps(state, indent=2));tmp.replace(root / 'state.json')
    persist()
    try:
        deadline = time.monotonic() + plan.get('wait_seconds', 86400)
        while not ready(json.loads(Path(plan['training_state']).read_text()),
                        json.loads(Path(plan['storage_state']).read_text())):
            if time.monotonic() > deadline:raise TimeoutError('Fixed training did not finish in time')
            time.sleep(30)
        check_idle_resume(plan)
        before = checkpoint_files(plan['host_checkpoint'], plan['checkpoint_step'])
        time.sleep(15)
        assert before == checkpoint_files(plan['host_checkpoint'], plan['checkpoint_step'])
        for name, digest in plan['pinned_files'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, 'Pinned source/input changed'
        env = dict(plan['environment'])
        env.update(MICROBATCH='4', RESUME_FROM=plan['container_checkpoint'], OUTPUT_DIR=plan['container_output'],
                   DECISION_EVAL_ONLY='1', DECISION_EVAL_CONFIG=plan['evaluation_config'],
                   DECISION_PROBE_UPDATES='0', SAVE_STRATEGY='no', EVAL_STRATEGY='no')
        command = ['docker', 'exec']
        for key, value in env.items():command += ['-e', key + '=' + str(value)]
        command += [plan['container'], 'timeout', '--signal=TERM', '--kill-after=60s',
                    str(plan.get('evaluation_timeout', 14400)), 'bash', plan['container_launcher']]
        state.update(status='evaluating', command=command, checkpoint_inventory=before);persist()
        with (root / 'evaluation.log').open('x') as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            state['child_pid'] = proc.pid;persist();rc = proc.wait()
        if rc:raise RuntimeError('Evaluation exited ' + str(rc))
        if output_processes(plan['container_output']):raise RuntimeError('Evaluation workers still active')
        after = checkpoint_files(plan['host_checkpoint'], plan['checkpoint_step'])
        assert before == after, 'Checkpoint changed during read-only evaluation'
        report = json.loads((root / 'evaluation.json').read_text())
        assert report['checkpoint_step'] == plan['checkpoint_step']
        state['comparison'] = compare(json.loads(Path(plan['previous_evaluation']).read_text()), report)
        state.update(status='complete', limitation='Business and validation assessment; seven-suite and broad acceptance remain separate')
    except Exception as error:
        state.update(status='failed', error=str(error));raise
    finally:
        persist()


if __name__ == '__main__':
    main()
