"""Bounded, reversible CPU-affinity probe for an already running training job.

No model or accelerator setting is changed. Only explicitly named processes
and their threads are managed. State contains local process identities; keep it
private. Sequential timing windows are diagnostic, not a controlled benchmark.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import time


def start_identity(pid):
    text = Path(f'/proc/{pid}/stat').read_text()
    return text.rsplit(')', 1)[1].split()[19]


def persist(path, state):
    state['observed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(state, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def snapshot(pid, target):
    main = set(os.sched_getaffinity(pid))
    if not target or not set(target) <= main:
        raise ValueError('Target CPUs must be a nonempty subset of the original mask')
    threads = {}
    for entry in Path(f'/proc/{pid}/task').iterdir():
        tid = int(entry.name)
        try:
            threads[str(tid)] = {'identity': start_identity(tid),
                                 'cpus': sorted(os.sched_getaffinity(tid))}
        except (FileNotFoundError, ProcessLookupError):
            continue
    return {'pid': pid, 'identity': start_identity(pid), 'original': sorted(main),
            'target': sorted(target), 'threads': threads}


def assert_identity(rank):
    if start_identity(rank['pid']) != rank['identity']:
        raise RuntimeError('PID identity changed; refusing to modify another process')


def apply(rank):
    assert_identity(rank)
    for tid, previous in rank['threads'].items():
        try:
            if start_identity(int(tid)) != previous['identity']:
                continue
            # Preserve any special affinity established by the runtime itself.
            if previous['cpus'] == rank['original']:
                os.sched_setaffinity(int(tid), rank['target'])
        except (FileNotFoundError, ProcessLookupError):
            continue


def restore(rank):
    try:
        assert_identity(rank)
    except FileNotFoundError:
        return 'process_exited'
    for entry in Path(f'/proc/{rank["pid"]}/task').iterdir():
        tid = int(entry.name)
        try:
            previous = rank['threads'].get(str(tid))
            if previous and start_identity(tid) == previous['identity']:
                os.sched_setaffinity(tid, previous['cpus'])
                if sorted(os.sched_getaffinity(tid)) != previous['cpus']:
                    raise RuntimeError('Original affinity was not restored')
            elif sorted(os.sched_getaffinity(tid)) == rank['target']:
                # Newly created threads may have inherited the probe mask.
                os.sched_setaffinity(tid, rank['original'])
        except (FileNotFoundError, ProcessLookupError):
            continue
    return 'restored'


def read_updates(path):
    updates = {}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # A concurrent final line may still be being written.
        if 'loss' in row and 'global_step/max_steps' in row:
            step = int(row['global_step/max_steps'].split('/')[0])
            updates[step] = row
    if not updates:
        raise RuntimeError('No valid training updates found')
    latest = updates[max(updates)]
    for key in ('loss', 'grad_norm'):
        if key in latest and not math.isfinite(float(latest[key])):
            raise RuntimeError('Nonfinite training metric observed; restore original affinity')
    return updates


def elapsed(row):
    units = {'d': 86400, 'h': 3600, 'm': 60, 's': 1}
    return sum(int(value) * units[unit]
               for value, unit in re.findall(r'(\d+)([dhms])', row['elapsed_time']))


def window(updates, start, end):
    if any(step not in updates for step in range(start, end + 1)):
        raise RuntimeError('Incomplete timing window')
    values = [elapsed(updates[step]) - elapsed(updates[step - 1])
              for step in range(start + 1, end + 1)]
    if not values or min(values) <= 0:
        raise RuntimeError('Invalid timing intervals')
    return {'start_step': start, 'end_step': end, 'seconds_per_update': values,
            'median_seconds': statistics.median(values), 'mean_seconds': statistics.mean(values)}


def run(args):
    if args.state.exists():
        raise ValueError('State exists; inspect it or use --restore')
    ranks = []
    assigned = set()
    for spec in args.rank:
        pid, mask = spec.split(':', 1)
        target = set()
        for part in mask.split(','):
            ends = part.split('-')
            target.update(range(int(ends[0]), int(ends[-1]) + 1))
        if assigned & target:
            raise ValueError('Rank CPU allocations overlap')
        assigned |= target
        ranks.append(snapshot(int(pid), target))
    updates = read_updates(args.log)
    initial = max(updates)
    # Avoid measuring across validation/checkpoint boundaries.
    remaining = args.save_every - initial % args.save_every
    if remaining <= 2 * (args.updates + 1) + 2:
        raise RuntimeError('Too close to a checkpoint boundary for this probe')
    state = {'status': 'running', 'phase': 'await_step_boundary', 'ranks': ranks,
             'initial_step': initial, 'log': str(args.log), 'windows': {},
             'scope': 'Sequential CPU-affinity diagnostic; no model/config change'}
    persist(args.state, state)  # Recovery information must precede any mutation.
    started = last_advance = time.monotonic()
    last_step = initial
    candidate_start = baseline_start = None
    try:
        while time.monotonic() - started < args.timeout:
            updates = read_updates(args.log)
            current = max(updates)
            for rank in ranks:
                assert_identity(rank)
            if current > last_step:
                last_step = current
                last_advance = time.monotonic()
            if time.monotonic() - last_advance > args.stall_timeout:
                raise RuntimeError('Training progress timeout; reverting probe')
            if candidate_start is None and current > initial:
                state['windows']['baseline_before'] = window(
                    updates, current - args.updates, current)
                # Refresh thread identities immediately before binding.
                ranks = [snapshot(r['pid'], r['target']) for r in ranks]
                state['ranks'] = ranks
                state.update(phase='applying_candidate', last_step=current)
                persist(args.state, state)
                for rank in ranks:
                    apply(rank)
                candidate_start = current + 1  # Exclude the transition update.
                state.update(phase='candidate', candidate_timing_start=candidate_start)
            elif candidate_start is not None and baseline_start is None and current >= candidate_start + args.updates:
                state['windows']['candidate'] = window(updates, candidate_start, current)
                state['restore_results'] = [restore(rank) for rank in ranks]
                baseline_start = current + 1
                state.update(phase='baseline_after', baseline_timing_start=baseline_start)
            elif baseline_start is not None and current >= baseline_start + args.updates:
                state['windows']['baseline_after'] = window(updates, baseline_start, current)
                state.update(status='complete', phase='original_affinity_restored')
                break
            state['last_step'] = current
            persist(args.state, state)
            time.sleep(args.poll)
        else:
            raise RuntimeError('Overall probe timeout')
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        failures = []
        for rank in ranks:
            try:
                restore(rank)
            except Exception as error:
                failures.append(repr(error))
        state['final_restore_errors'] = failures
        if failures:
            state['status'] = 'restore_attention_required'
        persist(args.state, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--rank', action='append', default=[], help='PID:CPU_RANGE')
    parser.add_argument('--log', type=Path)
    parser.add_argument('--updates', type=int, default=3)
    parser.add_argument('--save-every', type=int, default=150)
    parser.add_argument('--timeout', type=int, default=7200)
    parser.add_argument('--stall-timeout', type=int, default=1500)
    parser.add_argument('--poll', type=int, default=5)
    args = parser.parse_args()
    if args.restore:
        state = json.loads(args.state.read_text())
        results = [restore(rank) for rank in state['ranks']]
        state.update(manual_restore_results=results)
        persist(args.state, state)
        print(results)
        return
    if not args.rank or args.log is None or min(args.updates, args.save_every, args.poll) <= 0:
        parser.error('Provide rank mappings, training log and positive intervals')
    def interrupted(signum, frame):
        raise InterruptedError(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    run(args)


if __name__ == '__main__':
    main()
