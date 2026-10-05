# Resume recovery after a host pinned-memory allocation failure

Use the image digest, dependency versions, model revision and dataset split recorded in CHECKPOINT_EVALUATION.md and THROUGHPUT_TUNING.md. This change disables only DataLoader pinning; it does not change FSDP optimizer offload, the global batch, training objective, schedule, or checkpoint format.

The observed failure was an aclrtMallocHostWithCfg allocation in DataLoader pin_memory after successful full checkpoint loading. The launcher now passes --dataloader_pin_memory false. A failed fallback now records status=failed instead of leaving status=continuing.

To resume an already stopped job, copy the original private plan to a new path, select fresh host_output and container_output directories, and set:

```json
{
  "resume_saved_checkpoint": true,
  "physical_devices": [0, 1, 2, 3],
  "previous_outputs": ["/workspace/prior-trial-output"]
}
```

Retain owned_pids from the original job and include every previous probe/continuation output in previous_outputs. The controller checks that those workers have exited, the four requested devices are idle, and all model, optimizer, scheduler and RNG files remain stable. It sends no signal in this mode. Remove evaluation_config only when the paired evaluations have already completed and their evidence is preserved. Keep the same complete checkpoint and training continuation directory.

```bash
python check_throughput.py
bash -n run_throughput.sh
python tune_microbatch.py --plan "$PRIVATE_RESUME_PLAN"
```

The timing protocol remains four updates per candidate from the same checkpoint, identical global sample fingerprints, finite loss/gradients, and bounded numerical differences. Continue with a faster candidate only when the existing admission rules pass. Short probes do not establish final accuracy. Update any storage supervisor to follow the new controller state and live training log; preserve its old state and relocation manifests. Do not launch overlapping supervisors or remove the current recovery checkpoint.

Validation: eight CPU checks cover saved-checkpoint takeover, occupied/unknown devices, surviving workers, candidate admission and failed-fallback state reporting. Real-model recovery and speed remain to be measured after deploying this revision.
