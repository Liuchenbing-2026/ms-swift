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
