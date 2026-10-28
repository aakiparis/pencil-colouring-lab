# Pencil colouring lab

Scripts and recipes for comparing image-editing models and training a paired
photo-to-pencil LoRA on Nebius. This repository preserves the experiment and
adds portable entry points for new data.

The selected recipe is **Qwen-Image-2.1 + update-200 LoRA, strength 1, detailed
background v4 prompt, full-frame VAE decoding**. The short prompt alone did not
reproduce the accepted result reliably. Keep the adapter **and** detailed prompt
as part of the inference recipe.

Background and findings:

- [Benchmarking open-source image models before fine-tuning](https://medium.com/@kiparisandrew/how-do-you-benchmark-open-source-image-models-before-fine-tuning-a9b2431bb10f)
- [Does fine-tuning beat the best prompt?](https://medium.com/@kiparisandrew/does-model-fine-tuning-beat-the-best-prompt-760c8c1dde55)

## What is included

| Path | Purpose |
| --- | --- |
| `lab.py` | Portable preparation, inference plans/submission, output synchronisation and training staging |
| `tools/provision_storage.py` | Plan/create a bucket and shared model cache in an existing project |
| `archive/original/` | Unmodified original scripts, review UI, launchers, trainer and patch; historical machine/account paths remain |
| `archive/source-inventory.json` | Original source checksums |
| `recipes/historical/` | Model revisions, per-image prompts/seeds, training configuration, target selection and provenance |
| `prompts/` | Reusable short, single-image background-v4 and composite prompts |
| `examples/` | Dataset and cloud configuration templates |
| `tests/` | CPU guardrail and packaging checks |

**Not included:** images, generated outputs, model weights, the trained adapter,
credentials, private cloud job responses or a hosted service. This is a runnable
workflow with supplied assets, not a self-contained copy of the private dataset.
Archived scripts are reference material; start with `lab.py` for new runs.

## Local setup

Use Python 3.12, the Nebius CLI and a GPU environment for actual model execution.
The local preparation and checks do not require CUDA.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-local.txt
python lab.py verify-archive
python -m unittest discover -s tests -v
mkdir -p local workspaces
cp examples/cloud.example.json local/cloud.json
cp examples/dataset.example.json local/dataset.json
```

Fill in `local/cloud.json` using [the resource guide](docs/nebius.md). Sign in to
Nebius separately and configure local S3 credentials using the standard AWS
credential chain. No token or secret value belongs in a tracked file.

## Prepare and run inference

Edit `local/dataset.json` to reference files under your data directory. Each pair
has a unique ID, one or two inputs, and an optional reference target. Training
also requires explicit groups, identities and splits; see [the format](docs/dataset.md).

```sh
python lab.py prepare \
  --manifest local/dataset.json --data-root /absolute/path/to/images \
  --output workspaces/demo --seed 42 \
  --adapter /absolute/path/to/pytorch_lora_weights.safetensors \
  --checkpoint 200 --strength 1
```

Omit `--adapter` and `--checkpoint` to compare the base model. The detailed
background-v4 prompt is the default; `--short-prompt` is an explicit comparison.
Use a **new workspace** to change seed, prompt or adapter. Preparation creates a
stable input archive, exact run JSON, executable worker, runtime requirements,
asset checksums and preparation record. It allocates no compute.

Plan first; add `--submit` to actually upload and allocate a job:

```sh
python lab.py infer --workspace workspaces/demo --cloud local/cloud.json --mode warm
python lab.py infer --workspace workspaces/demo --cloud local/cloud.json --mode warm --submit
# Wait for successful warm completion in Nebius before the next command.
python lab.py infer --workspace workspaces/demo --cloud local/cloud.json --mode smoke --gpus 1 --submit
# Inspect the smoke output, then submit the full batch.
python lab.py infer --workspace workspaces/demo --cloud local/cloud.json --mode batch --gpus 8 --submit
python lab.py sync --workspace workspaces/demo --cloud local/cloud.json
python -m http.server 8093 --directory workspaces
```

Open `http://127.0.0.1:8093/demo/review.html`. Synchronisation downloads raw outputs,
checks completed-image hashes and creates an original/reference/result page.
Smoke and batch share result keys within one immutable run: batch may reuse the
matching smoke image. Run `sync` again to refresh; it does not poll indefinitely.
The historical richer comparison UI and feedback persistence remain in the archive.

The portable runner supports the selected Qwen pipeline. Earlier Qwen Edit and
FLUX launchers/configurations are preserved for investigation; they have **not**
been converted into portable end-to-end commands.

## Train a new adapter

Review targets, identity grouping, split membership and geometry first. `--freeze`
is an explicit acknowledgement of that review and the documented white-letterbox
normalisation. Use a separate workspace for this example:

```sh
python lab.py prepare \
  --manifest local/dataset.json --data-root /absolute/path/to/images \
  --output workspaces/training-demo --freeze
python lab.py stage-training --workspace workspaces/training-demo --cloud local/cloud.json
python workspaces/training-demo/training/verify_preparation.py
python workspaces/training-demo/training/verify_distributed.py
python workspaces/training-demo/training/launch.py calibrate
# Append --launch to the preceding command to submit the calibration job.
```

Calibration measures a separate short run; the pilot starts from the frozen base,
not from the calibration adapter. The preserved update counts and hyperparameters
are in `recipes/historical/lora-v1/pilot.json`. Do not edit a staged run once launched.

```sh
python lab.py status-training --workspace workspaces/training-demo --cloud local/cloud.json
# View http://127.0.0.1:8093/training-demo/training-review/review.html
# Once complete, inspect previews, memory, timing and errors, then acknowledge review:
python lab.py approve-calibration --workspace workspaces/training-demo \
  --cloud local/cloud.json --confirm-preview-reviewed
python workspaces/training-demo/training/launch.py pilot --launch
python lab.py status-training --workspace workspaces/training-demo --cloud local/cloud.json
```

`status-training` is a one-off refresh. Calibration approval validates the matching
completion checksum, run identity, measured timing and GPU count. It does not
judge image quality for you. No automatic controller is started or copied into
the portable training workspace.

After completion, obtain the adapter from the job's S3 prefix recorded in
`training-review/status.json` / `training/pilot.json` and its run key. Verify it
against `pilot-completion.json`'s `adapter_sha256`, then pass it to `prepare
--adapter` for a new inference run. A base model is still needed; the adapter is
not a standalone model. The [training guide](docs/training.md) explains validation,
checkpoints, retrieval and the distinction between previews and saved updates.

## Reproduction limits and checks

[Reproducibility notes](docs/reproducibility.md) explain the exact historical
pins, article/code differences and the checks performed. Dependencies are pinned
at the top level, **not a complete transitive lock**. Matching a seed alone does
not guarantee identical output across hardware or software changes.

CPU tests and synthetic paired export checks verify preparation and contracts.
The newly portable job packaging has not been rerun on paid GPUs. The archive
contains the scripts from the completed experiment; this is not a claim that every
historical backup is independently runnable. Review [third-party notices](THIRD_PARTY_NOTICES.md)
and model terms before reuse or serving.
