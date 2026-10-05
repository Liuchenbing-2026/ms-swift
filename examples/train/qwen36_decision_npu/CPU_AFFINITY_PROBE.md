# Live CPU-affinity diagnostic

This optional host-side experiment tests CPU scheduling locality without
restarting training or changing its model, optimizer, data, global batch or
NPU allocation. It does not migrate existing memory pages, pin pageable
buffers, enable eight-card training, or remove CPU offload.

Inspect the device-to-CPU topology and each rank's NUMA memory placement first.
Choose disjoint CPU sets near each rank's NPU, allowing enough cores for its
existing OpenMP/runtime threads. Local NUMA memory may be too small to hold all
CPU-offloaded states; CPU affinity alone cannot establish local memory access.

Only supply PIDs belonging to this task. The helper records process/thread
start identities and original masks before changing affinity. Threads with a
pre-existing special mask are preserved. It restores original masks after the
candidate window, on progress timeout, on SIGINT/SIGTERM, and on ordinary
errors. SIGKILL or host failure cannot run a cleanup handler; retain the state
file and use the explicit restore command if needed.

## Commands on the training host

Run the CPU-only checks first. They create an isolated short-lived process and
exercise mask restoration, process-identity protection, complete phase
transitions and a stall after the candidate mask is applied.

```bash
python3 check_cpu_affinity_probe.py

# Fill in the actual owned rank PIDs and verified CPU sets.
python3 cpu_affinity_probe.py \
  --rank "${RANK0_PID}:${RANK0_CPUS}" \
  --rank "${RANK1_PID}:${RANK1_CPUS}" \
  --rank "${RANK2_PID}:${RANK2_CPUS}" \
  --rank "${RANK3_PID}:${RANK3_CPUS}" \
  --log /path/to/outputs/train/logging.jsonl \
  --state /path/to/new-affinity-probe.json \
  --updates 3 --save-every 150 --timeout 7200 --stall-timeout 1500

# Manual recovery: verifies process identity before restoring masks.
python3 cpu_affinity_probe.py \
  --state /path/to/new-affinity-probe.json --restore
```

The example reads the existing SWIFT JSONL log. It waits for the next update
boundary, records the preceding original-affinity window, applies the probe,
discards one transition update, measures three full updates, restores the
original affinity, discards another transition update and records the next
three original-affinity updates. It refuses to start near the configured
checkpoint boundary. A missing update, process exit or nonfinite metric
terminates the probe and triggers restoration; it never stops training.

Use a persistent launcher when running over SSH. The state file records phase,
update ranges, elapsed-time differences and restoration errors. It contains
local PIDs and timing data; keep it private. Inspect `final_restore_errors`
before considering the experiment finished.

## Interpretation and limits

These sequential windows use different training examples and can be affected
by concurrent CPU/storage load. They are diagnostic evidence, not proof of
causal speedup or end-to-end model quality. Do not select a permanent setting
from a single small window. Keep actual checkpoint restoration, numerical
validation, final accuracy and eight-card scaling as separate acceptance steps.
No production speedup is claimed before the live probe completes.
