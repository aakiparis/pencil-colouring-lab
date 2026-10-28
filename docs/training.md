# LoRA training and inspection

The original training entry points, patched official trainer and patch are
preserved together. `stage-training` copies them into a new workspace, supplies
resource IDs and substitutes the local storage client. It retains the trainer,
model revision, DDP strategy and recorded hyperparameters. It removes the old
automatic controller: explicitly review calibration before submitting the pilot.

The launcher's default action is a local plan. `--launch` uploads a package to S3
and submits the job. It injects `bootstrap.sh`, `train_worker.py` and
`requirements.txt` into the pinned PyTorch image. The bootstrap creates an
environment on the machine's local `/tmp` disk. The worker verifies the package,
frozen files and injected-code checksums, downloads the model to the shared
cache, and starts Accelerate with the configured GPU world size.

`distributed_train.py` maps the configuration to official trainer arguments.
The local `trainer_ddp.py` carries the all-rank checkpoint/resume and validation
changes; `trainer-ddp.patch` and upstream checksums document their provenance.
It is already patched **before** packaging. The worker does not apply a patch.

Only LoRA parameters are updated. Accelerate synchronises gradients between
replicas; it does not train several independent adapters. The optimiser update
count is not a count of unique images. Model preparation, latent caching,
checkpoint upload and previews add time beyond optimiser steps. Use the actual
calibration timing and memory report rather than a fixed duration estimate.

## Progress and checkpoints

`lab.py status-training` reads the cloud job state, metrics, complete-checkpoint
markers and held-out preview index, and downloads verified previews. Open the
resulting `training-review/review.html`. Preview generation follows the trainer's
epoch schedule, while saved checkpoints follow the update interval: a preview
at one update is not a saved checkpoint from a nearby update.

Review the calibration image as well as its completion report. Then run
`lab.py approve-calibration ... --confirm-preview-reviewed`. The pilot is a
separate run from the original base weights. To recover a failed/preempted job,
inspect its failure and complete-checkpoint manifest, then use the original
launcher's explicit `--retry`; it refuses a duplicate running job. Never delete
job records merely to silence that guard.

## Retrieve the final adapter

This prints the exact object key and downloads the verified final adapter after
a completed pilot. Run it from the repository root, inside the local environment:

```sh
python tools/fetch_adapter.py \
  --workspace workspaces/training-demo --cloud local/cloud.json \
  --output local/pytorch_lora_weights.safetensors
```

Use that file with `lab.py prepare --adapter ... --checkpoint 200 --strength 1`
when it really is the recorded final update. Keep the model revision, adapter
hash, strength, prompt version, seed and decoding settings together. The
historical accepted adapter hash is in
`recipes/historical/benchmark-v2/qwen21_lora200_detailed/run.json`; newly trained
weights are not expected to have that hash.

Evaluate base + detailed prompt, LoRA + short prompt and LoRA + detailed prompt
on the same validation inputs/seeds. The preserved validation comparison scripts
are in `archive/original/lora-v1/evaluation/`; they are historical, not automatically
staged portable commands. Use fresh inference manifests to compare new runs.
Reserve a genuinely untouched test set until the choices are fixed. Training
loss and a single attractive preview do not establish improvement.
