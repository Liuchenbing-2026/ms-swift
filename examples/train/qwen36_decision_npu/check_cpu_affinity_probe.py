"""CPU-only checks using an owned disposable process; no NPU required."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import cpu_affinity_probe as probe


def main():
    command = ('import threading,time; '
               'threading.Thread(target=lambda:time.sleep(90),daemon=True).start(); '
               'print("ready",flush=True); time.sleep(90)')
    child = subprocess.Popen([sys.executable, '-c', command], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        original = sorted(os.sched_getaffinity(child.pid))
        target = original[:max(1, len(original) // 2)]
        rank = probe.snapshot(child.pid, target)
        probe.apply(rank)
        assert sorted(os.sched_getaffinity(child.pid)) == target
        probe.restore(rank)
        assert sorted(os.sched_getaffinity(child.pid)) == original
        wrong = dict(rank, identity=str(int(rank['identity']) + 1))
        try:
            probe.apply(wrong)
        except RuntimeError:
            pass
        else:
            raise AssertionError('PID reuse guard failed')
        assert sorted(os.sched_getaffinity(child.pid)) == original
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / 'logging.jsonl'
            def row(step):
                return {'loss': 1.0, 'grad_norm': 1.0,
                        'global_step/max_steps': f'{step}/100', 'elapsed_time': f'{step}s'}
            log.write_text(json.dumps(row(0)) + '\n')
            args = argparse.Namespace(state=root / 'state.json', rank=[
                str(child.pid) + ':' + ','.join(map(str, target))], log=log,
                updates=1, save_every=100, poll=0.02, timeout=15, stall_timeout=5)
            stop = threading.Event()
            def advance():
                for step in range(1, 10):
                    if stop.wait(0.2):
                        break
                    with log.open('a') as handle:
                        handle.write(json.dumps(row(step)) + '\n')
            writer = threading.Thread(target=advance)
            writer.start()
            try:
                probe.run(args)
            finally:
                stop.set()
                writer.join()
            state = json.loads(args.state.read_text())
            assert state['status'] == 'complete'
            assert set(state['windows']) == {'baseline_before', 'candidate', 'baseline_after'}
            assert not state['final_restore_errors']
            assert sorted(os.sched_getaffinity(child.pid)) == original
            # Missing progress also restores the original process configuration.
            log.write_text(json.dumps(row(0)) + '\n')
            args.state = root / 'stall.json'
            args.stall_timeout = 0.2
            def one_update():
                time.sleep(0.05)
                with log.open('a') as handle:
                    handle.write(json.dumps(row(1)) + '\n')
            writer = threading.Thread(target=one_update)
            writer.start()
            try:
                probe.run(args)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Progress timeout did not fire')
            writer.join()
            failed = json.loads(args.state.read_text())
            assert failed['phase'] == 'candidate'
            assert failed['status'] == 'failed' and not failed['final_restore_errors']
            assert sorted(os.sched_getaffinity(child.pid)) == original
        print('PASS: apply/restore, PID identity, phase transitions and progress timeout')
    finally:
        child.terminate()
        child.wait(timeout=10)


if __name__ == '__main__':
    main()
