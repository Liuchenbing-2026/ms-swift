# Checkpoint storage across two filesystems

These optional host-side tools preserve every byte of selected immutable DCP
checkpoint shards while moving them to another persistent filesystem. Training
code, optimizer state values and the checkpoint format are unchanged. The source
paths become relative symlinks. Keep the destination storage for as long as any
checkpoint references it.

Requirements: Linux root, Docker, Python 3.10+, Linux 5.2+ on aarch64/x86_64 for
the live-container mount helper, and an independently allocated destination.
Check `df -B1` on both filesystems. Do not use tmpfs for persistent checkpoints.
Do not move a checkpoint still being written. Confirm training has advanced
past its save step and no other process writes the selected files.

Run the following on the host. Replace each path and container name with the
isolated training environment. `DECISION_WORK` is its host workspace, mounted
at `/workspace` in the container; `DECISION_STORE` is on the other filesystem.

```bash
DECISION_WORK=/path/to/training-workspace
DECISION_STORE=/path/on/other-filesystem/checkpoint-spill
DECISION_CONTAINER=task19-qwen36-swift
python3 mount_checkpoint_storage.py \
  --workspace "$DECISION_WORK" --storage "$DECISION_STORE" \
  --container "$DECISION_CONTAINER"
python3 relocate_checkpoint_files.py \
  --checkpoint "$DECISION_WORK/outputs/train/checkpoint-150" \
  --storage "$DECISION_WORK/.checkpoint-spill" \
  --manifest "$DECISION_WORK/checkpoint-spill-150.json" \
  --reserve-gib 18 \
  optimizer_0/__1_0.distcp optimizer_0/__2_0.distcp \
  pytorch_model_fsdp_0/__1_0.distcp
```

The file list is an example for a four-rank checkpoint. Inventory the actual
checkpoint first. The reserve is a minimum remaining destination capacity;
choose it for the host, and ensure the source filesystem will also satisfy the
training checkpoint-space guard after relocation. The script aborts if the
reserve is reached, a destination already exists, or the source changes.
It does not lower or bypass the training space guard.

For each file, the script copies to a new file, fsyncs it, rereads it to verify
SHA256, checks that the source has not changed, and atomically installs a
relative symlink. JSON records distinguish verified copies and installed links.
Failures preserve the original until the atomic switch; completed earlier
files remain relocated. Inspect the manifest and any `.partial` files before
retrying. Never automatically overwrite an existing manifest or destination.

## Container restart and host reboot

The live-container helper adds mounts only; it does not restart training or
modify Docker's saved container configuration. Host mounts are not registered
in fstab. After a host reboot, recreate the host bind mount before opening the
checkpoint, and rerun the helper for an existing restarted container. For a
new container, explicitly add the following to its original `docker run`
command in addition to the ordinary workspace mount:

```bash
--mount type=bind,src="$DECISION_STORE",dst=/workspace/.checkpoint-spill
```

Preserve the host mount at `$DECISION_WORK/.checkpoint-spill` as well. Before
resuming, verify all symlinks and manifest hashes through the container, then
perform actual framework restoration. Byte-identical files and readable paths
do not prove optimizer/RNG/sample-position restoration or final model quality.

The relocation manifest contains local paths and hashes; keep it local. Do not
publish checkpoints or raw metrics. Do not delete spill files merely because
an old checkpoint directory has been rotated. First establish that no retained
checkpoint references them and a newer complete recovery point is validated.

## Validation scope

The relocation helper has been checked with synthetic file round trips and an
insufficient-space rejection. Host/container mount mapping has been exercised
on the running training environment. Full model restoration and subsequent
checkpoint rotation require separate validation. These utilities do not claim
to improve training step time.

## Follow Trainer retention without deleting a retained checkpoint

`rotate_checkpoint_storage.py` is an optional companion to the task19
microbatch tuning driver. It waits until the throughput trials have ended and
the continuation has advanced beyond a newly saved checkpoint. It inventories
all 16 model/optimizer/metadata/scheduler/RNG/trainer files twice. It never
removes checkpoint directories: the configured Trainer retention policy must
already have removed the old checkpoint directory before its managed spill
copies are eligible for reclamation.

The initial relocation manifest is an allowlist. Every deleted file must have
been copied and linked successfully, still have its recorded size and a single
hardlink, and be inside the expected spill directory. Any remaining reference
from the training output tree prevents reclamation. Unrelated files, retained
checkpoints, and the spill directories themselves are preserved.

After reclaiming eligible orphan files, the existing SHA256-verified relocation
helper moves the configured shards of the new checkpoint, preserving the same
reserve. This happens outside timed throughput trials. At the final checkpoint,
only obsolete spill copies are reclaimed; the final checkpoint is not moved.
Space used by unrelated workloads can still exhaust the reserve and cause a
safe failure. The helper does not bypass the training storage guard, restart
training, or prove that a future checkpoint has been successfully reloaded.

The actual training image, dependencies and original launch/evaluation commands
remain those in task19's `镜像分支与执行命令.md`; this host-side utility needs only
Python 3.10+ and the relocation helper from the same branch, with the existing
host/container mounts already checked. No image export or new package is needed.

```bash
python3 check_checkpoint_rotation.py
python3 rotate_checkpoint_storage.py --plan /secure/local-storage-plan.json
```

Example plan (replace paths with the already verified isolated workspace):

```json
{
  "training_root": "/work/task19/outputs/train",
  "spill_root": "/work/task19/.checkpoint-spill",
  "output": "/work/task19/storage-rotation",
  "initial_manifest": "/work/task19/checkpoint-spill-150.json",
  "progress": "/work/task19/progress.json",
  "tuning_state": "/work/task19/throughput-tuning/real-trials/state.json",
  "steps": [300, 450, 600],
  "files": ["optimizer_0/__1_0.distcp", "optimizer_0/__2_0.distcp", "pytorch_model_fsdp_0/__1_0.distcp"],
  "reserve_gib": 18,
  "timeout_seconds": 259200
}
```

Keep the plan and all manifests local. `state.json` distinguishes waiting,
reclaiming, relocating, completion and failure. Existing state prevents an
unreviewed restart after a partial copy. Inspect both state and per-file
relocation records before retrying; never start a second keeper over the same
storage. CPU tests exercise retained-checkpoint protection, remaining symlink
references, unexpected file changes, and idempotent orphan reclamation. They
do not substitute for a complete real-model rotation cycle.
