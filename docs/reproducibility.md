# Reproducibility and provenance

The articles describe the experiment. The checked-in source, exact per-image
run recipes and evidence records describe what actually ran. Read them together:

- [Baseline article](https://medium.com/@kiparisandrew/how-do-you-benchmark-open-source-image-models-before-fine-tuning-a9b2431bb10f)
- [Fine-tuning article](https://medium.com/@kiparisandrew/does-model-fine-tuning-beat-the-best-prompt-760c8c1dde55)

## Recorded pins

The original image reference and digest are in `recipes/historical/container.json`.
The model revision, Diffusers commit and direct dependency versions are in the
historical run configurations and `archive/original/lora-v1/requirements.txt`.
Upstream trainer URLs/hashes and the local patch hash are in
`recipes/historical/lora-v1/source-provenance.json`.

Do not silently replace these pins with `latest`, a current model branch or a
newer trainer when claiming reproduction. Model access or package availability
may change. This archive does not mirror the model or dependency distributions.

The historical training selection, frozen manifest and completion/provenance
records are included without image bytes. They provide the recorded split,
target and adapter hashes. They do not give permission to redistribute the
underlying photographs or generated targets.

## Deliberate distinctions

| Article shorthand | Actual implementation / portable workflow |
| --- | --- |
| Patch the official trainer inside the job | The trainer was patched locally and packaged; the patch is preserved for audit. |
| One paired Parquet dataset | The exporter writes separate train, validation and test Parquet files with embedded image bytes. |
| Download using an S3 synchronisation command | Original controllers and the portable runner use boto3; portable inference sync verifies output hashes. |
| Example command placeholders | Recorded container/model/commit values are supplied by versioned recipes. Resource IDs and credentials remain local. |
| Every historical model has one generic launcher | Several experimental launchers and versions exist. Portable commands currently cover the accepted Qwen pipeline; other launchers remain archived. |
| Short instruction can replace a detailed prompt after LoRA | The accepted result still uses the detailed background-v4 prompt with the adapter. |
| Reported reviewer win percentage | Reviewer narrative, not an independently reconstructed metric. No benchmark score is derived from incomplete persisted verdicts. |

The portable worker is generated from the selected original worker. It changes
cache paths and input archive keys, and supplies a locally installed environment.
It keeps the generation call, model revision, prompt composition, seed policy,
LoRA loading and full-frame decode. It is **not byte-identical to the historical
worker**, and its hash is recorded separately. New training staging likewise
changes paths and package identity; a new run is not the historical run.

## What is deterministic

Input archives have stable ordering, ownership, permissions and timestamps.
Prepared configuration and source hashes identify the inference run. Each run
gets a separate cache/output namespace; original cloud prefixes are not reused.
Seed changes create a new run. The historical same-seed repeat exercise is kept
under `recipes/historical/benchmark-v2/spiderman-repeat-20261018/`.

An equal seed identifies the random-number initialisation, not a universal image
identifier. Hardware, dependency versions, numerical kernels, model revision,
inputs, adapter and sampling settings also affect results. Retraining can change
weights even when a seed is retained. No cross-hardware bitwise guarantee is made.

Top-level requirements are pinned, but transitive packages are **not fully
locked or hash-verified**. Historical training records contain a package list;
the worker also saves `environment.txt` for each run. For stricter reproduction,
archive the full resolved environment, dependency artefacts and driver/GPU details,
and build an immutable runtime image before a production release. Reusing a
container digest alone does not freeze packages installed afterwards.

## Validation performed for this repository

- Original source inventory checksums.
- Stable archives/configuration across independent preparation directories.
- Native worker task fields, composite handling and generated Python syntax.
- Path traversal and escaping-symlink rejection.
- Cross-split source group, identity and input/target hash leakage checks.
- Inference planning with cloud/CLI access forbidden by test doubles.
- Mutation detection for prepared run/runtime files.
- Synthetic real-image export, Parquet reload, shared geometry and training staging
  when the optional local dependencies are installed.
- Preserved preparation/DDP contract checks in a synthetic staged workspace.

No paid GPU run, resource provisioning or Docker build is performed by these
checks. The GitHub CPU workflow runs standard-library checks; the real paired
export check needs `requirements-local.txt` and otherwise reports a skip. Existing
GPU results are historical evidence, not a fresh validation of this portability layer.

## Contents and exclusions

`archive/source-inventory.json` covers the original project source files, including
historical UI backups. Runtime job responses, logs, archives, virtual environments,
images and temporary downloaded inspection files are excluded. Original scripts
retain historical paths and non-secret account identifiers for provenance; the
portable configuration uses placeholders and standard credential providers.

The generated repository is private. Credentials are excluded and checked before
push; keeping a repository private is not a replacement for secret handling.
