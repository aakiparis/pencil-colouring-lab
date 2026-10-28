# Nebius resources

The portable workflow targets **UK South 2**, matching the preserved workers.
It uses an existing project and subnet. GPU availability, platform names, presets,
CLI compatibility and current model access must be checked in your account.
No cloud resources are created by cloning, preparing data or running CPU tests.

## Storage

Print the commands first:

```sh
python tools/provision_storage.py \
  --project-id YOUR_PROJECT_ID --name YOUR_UNIQUE_BUCKET_NAME \
  --cache-gib 200 --state local/storage-state.json
```

Add `--apply` to create the versioned S3 bucket and shared filesystem. This creates
chargeable storage. Subsequent runs with the same state verify the recorded
resources rather than create duplicates. Keep the state file: creation is not
transactional, so if a request succeeds but saving its response fails, inspect
the cloud before retrying. Existing resources and other projects are never deleted.

The historical filesystem size is a starting configuration, **not a capacity
recommendation** for every dataset. Training checks free space before downloads
and extraction. Optimiser checkpoints need much more space than adapter weights.

## Identity, secrets and network

Use the console or your existing infrastructure management to:

1. Create/select a project and subnet in UK South 2.
2. Give the job identity permission to mount the shared filesystem and resolve
   the specified MysteryBox secrets.
3. Create a storage service account and grant it `storage.editor` on this bucket.
4. Create its S3 access key. Store its access key ID and secret in a MysteryBox
   secret with keys `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.
5. Configure equivalent local S3 access through the standard AWS credential chain
   (environment variables or an AWS profile). These are S3 credentials, separate
   from your `nebius` CLI login.
6. If the model is gated, accept its terms and create a separate secret containing
   `HF_TOKEN`; set `hf_secret_id` for inference. The historical training launcher
   does not inject this optional token: add that secret reference before using a
   gated training model.

Copy `examples/cloud.example.json` to `local/cloud.json` and fill in the IDs.
`storage_secret_id` refers to the secret, **not its value**. Do not place tokens
or S3 keys in the configuration. The local client uses boto3's credential chain;
GPU containers receive the same keys through secret references.

The original IAM provisioning sequence is preserved in
`archive/original/provision.py`, and the original secret creation logic in
`archive/original/storage_ops.py`. They contain historical account settings and
are reference material, not portable entry points.

## Execution and ownership

An AI Job starts the pinned PyTorch container and executes `/bin/bash`, which
runs the injected bootstrap. The bootstrap installs the pinned requirements and
starts the Python worker. PyTorch does not discover or launch scripts itself.
Diffusers loads the model and performs generation; PyTorch executes tensor
operations through CUDA. boto3 transfers objects; the shared filesystem caches
model weights. Neither S3 nor the filesystem is a source-code repository.

Inference uses independent workers on the available GPUs. Training uses
Accelerate DDP: the GPU processes update **one shared adapter**. Jobs are finite
and preemptible; interrupted inference can skip matching completed outputs.
Training recovery accepts only fully committed checkpoints with matching run
identity and world size. Inspect failures before retrying.

Local job records prevent accidental duplicate submission in the same workspace.
They are not a distributed lock: use one controller per workspace. Check the
cloud after an ambiguous submission response before retrying. There is no
background poller or recurring automation started by this repository.
