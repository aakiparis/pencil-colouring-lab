Pencil colouring LoRA pilot — eight-GPU DDP, authorised for calibration and pilot
=================================================

Purpose
-------
Teach a compact adapter to reproduce the selected pencil colouring style from a photograph using this instruction:

Create a pencil colouring page.

The original Qwen-Image-2.1 remains frozen. Only attention LoRA matrices are trained. This is paired image editing, not a text-only DreamBooth portrait trainer. The official editing trainer conditions on the original twice (vision tokens and clean image latents), and computes the loss only on the target. It does not guarantee likeness preservation: that must be measured on held-out people.

Prepared files
--------------
- selection.json: every historical pair, current input hashes, candidate targets and provisional identity groups. No target is silently approved for training.
- freeze_dataset.py: validates approvals and leakage, preserves full images as opaque RGBA PNGs, exports self-contained Parquet image columns plus a manifest.
- vendor/train_dreambooth_lora_qwenimage21_img2img.py: unmodified official trainer at the pinned Diffusers commit.
- pilot.json: frozen settings for the authorised calibration and pilot.
- train_worker.py: verifies the package and dataset; records package versions, model revision and GPU; runs training; commits complete recoverable checkpoints to private S3.
- bootstrap.sh and requirements.txt: a separate training environment, leaving the running inference environment intact.
- launch.py: read-only plan by default; an explicit --launch allocates one finite, preemptible UK South 2 RTX PRO 6000 job using the existing project, S3 secret and shared cache.
- infer_adapter.py: loads the same base revision and the saved adapter and renders held-out photos. Omitting --adapter gives the short-prompt baseline. --prompt-run supplies the complete per-image prompts of a frozen baseline run.
- MODEL-LICENCE.txt and source-provenance.json: exact source and licence evidence.

What still needs selecting
--------------------------
The user has now approved target choices and authorised training. Root prepared 42 training, 4 validation and 4 test pairs, with full-image letterboxing to a source-aspect canvas and explicit provenance. See the current selection and frozen manifest for exact choices/exclusions; historical comments below explain the guardrails rather than a remaining approval request.

For each included row, choose selected_target as a candidate object with its exact path and SHA-256; set target_approval=approved, geometry_review=approved and grouping_confirmed=true. Confirm group_id and split. At the top, set target_selection_approved and group_review_approved true only after that review. Excluded or pending rows are omitted.

Filename-inferred person identities are provisional, not face recognition. The draft groups co-occurring people, identical source hashes and source groups into connected components. Near duplicates and unnamed people still need review. In particular, all appearances of one known person must stay in one split, even when photographed with someone else. The split guard rejects identity, group and exact-input leakage. Unknown human identities are listed explicitly. The draft sets independent Raphinha/Bellingham groups to validation and Vinicius/Doue groups to test; other groups are provisionally training. Check that adequate variety remains after the final exclusions. Source and reference geometry must be sufficiently aligned: do not train on an approved drawing that moves a face, changes a pose or invents limbs. The official trainer resizes both images to the same target-derived dimensions, then applies the identical centre crop to both, with flipping disabled. A condition with a different aspect ratio would therefore be stretched; the freezer rejects more than 5% aspect-ratio change unless allow_aspect_ratio_warp=true is explicitly recorded after review. That threshold is a local conservative guard, not a model requirement. It does not detect subject movement: geometry review is still necessary.

Composite pairs are excluded from this first pilot because the official trainer accepts one condition-image column. Multi-input support at inference does not imply multi-input support in this training script. Adding a composited contact sheet would change the task, so it is not done implicitly.

Pair 045: the corrected original and ChatGPT target were visually reviewed together and show the same CR7 boot design and view. The selected ChatGPT target is recorded as visually_verified_corrected_source. The previous source-mismatch flag was an inference and has been corrected; earlier baseline model runs still retain their historical input hashes.

Keep the evaluation split fixed after selection. Do not tune prompts or choose training targets by repeatedly inspecting the final test split. Additional genuinely new photographs are valuable for the final check.

Commands — no cloud operation unless --launch is present
-------------------------------------------------------
From this directory, using the local training-preparation environment:

.venv/bin/python freeze_dataset.py
../.venv/bin/python launch.py calibrate
../.venv/bin/python launch.py pilot

The latter two print the plan only. Once targets have been frozen and a training launch is wanted:

../.venv/bin/python launch.py calibrate --launch

After reading the calibration completion, sample loss and peak VRAM, record {"approved": true, "run_key": "<verified calibration run key>", "completion_sha256": "<SHA-256 of the exact private-S3 completion.json bytes>"} in calibration-reviewed.json. The pilot launcher verifies those values against the current configuration, dataset and trainer and the completed calibration timing/VRAM evidence in private S3. This review records the observed calibration result; it is not a new model choice or a request to approve duplicate work. Then:

../.venv/bin/python launch.py pilot --launch

If preemption causes a terminal failure, diagnose it before retrying:

../.venv/bin/python launch.py pilot --launch --retry

The launcher refuses duplicate active or completed jobs. A retry resumes only a complete checkpoint. No automatic endless retries and no unbounded GPU allocation are configured. The six-hour job timeout is a safety cap, not a measured training duration.

Proposed pilot and timing
-------------------------
Eight DDP replicas training one shared adapter; BF16 frozen base; rank 8, alpha 8; attention matrices only; learning rate 0.00005; batch 1 per GPU, gradient accumulation 1, effective batch 8; 1024-pixel budget with aspect-ratio buckets; centre crop; no random flips; no caption dropout; AdamW; gradient checkpointing; cached latents and image-specific prompt embeddings; text encoder/VAE offloaded outside precomputation. AdamW states are only for the adapter, so no quantised optimiser is required for this first configuration.

The pilot has 200 optimiser updates, or approximately 1,600 sample presentations, subject to the trainer's partial final batches at epoch boundaries. Effective dataset passes depend on the final training row count. This is a pilot setting to test, not an evidence-based optimum. Save every 25 updates and compare earlier checkpoints; do not assume the last checkpoint is best.

No honest wall-clock estimate is available before calibration on this exact trainer, image geometry and GPU. Inference throughput is not training throughput. The separate 20-update calibration writes preparation time, recent update timings and peak allocated VRAM. Estimate the pilot from measured update time multiplied by remaining updates, then add observed precomputation, checkpoint transfer and validation time. The cached latent and embedding precomputation currently reruns after preemption; model downloads and complete checkpoints are retained. Report a range after this measurement rather than inventing a duration now.

Eight independent inference GPUs do not automatically make one training run eight times faster. The user explicitly chose eight RTX PRO 6000 GPUs. One adapter is synchronised across eight DDP replicas, with effective batch 8. This is not eight independent training shards. The same architecture is used for calibration and pilot, so measured update throughput is comparable.

Checkpoints and storage
-----------------------
Base model weights are reused from the existing pinned Hugging Face snapshot in /cache/huggingface. The training code never uploads secrets. The dataset package, dependency/source provenance, per-update loss/learning-rate/VRAM records with epoch indices, full resume checkpoints and final adapter are saved under private lora-v1/pilot/<run-key>/<calibrate-or-pilot> prefixes in the existing Nebius bucket.

The run key includes the frozen dataset manifest, configuration and official trainer hash. All eight ranks invoke Accelerator.save_state and write their rank-specific RNG states. Barriers run before/after state saving and after the rank-zero S3 upload. A checkpoint manifest is uploaded only after every checkpoint file and all eight RNG files are verified. The latest-complete pointer is uploaded last. A recovered run verifies every restored file. Incomplete local checkpoints are quarantined. A preempted job may lose work since the previous complete checkpoint; it does not need to start again from the first update. The shared 200 GiB cache is also checked before package download, extraction, missing model files and each checkpoint. Capacity, used and free bytes are logged and recorded in provenance. Missing model download bytes come from Hugging Face dry-run metadata. An 8 GiB output reserve is a conservative local guard, not a measured checkpoint size; insufficient free space stops the job with a capacity report. No existing model caches are deleted and no extra volume is allocated. The shared filesystem is retained independently of the VM; compute is a finite Nebius job, not an indefinitely running training server.

The official trainer logs to TensorBoard files rather than an external experiment service. MLflow remains deferred. No Hub publication or public serving is configured.

Applying and evaluating the adapter
----------------------------------
The output is an additional pytorch_lora_weights.safetensors file, not a replacement for the whole Qwen model. Deploy the exact base revision plus this adapter and the short instruction. Keep weights separate initially, so the adapter can be disabled or its scale changed without modifying the base. The provided loader calls load_lora_weights and set_adapters. It retains full-frame VAE decoding; the tiled decoding setting that caused border smudges is not re-enabled.

Example on the prepared GPU environment, after downloading the chosen complete checkpoint:

python infer_adapter.py --dataset /path/to/frozen-dataset --adapter /path/to/checkpoint-100 --output /path/to/validation-scale-1 --split validation --scale 1.0

Omit --adapter for the same short instruction without LoRA. Use --prompt-run /path/to/frozen-background-run.json without --adapter for the full-prompt baseline. Use separate output folders for each comparison. The script saves exact prompts, seeds, adapter hashes, scale, input/output hashes and actual dimensions with every image. Native validation also generates one held-out preview (prefer pair 006), using the short prompt and seed 42 every five epochs, starting at epoch zero. It reuses the live transformer and cached prompt, reloads only the VAE, keeps full-frame decoding and publishes previews plus actual update/epoch metadata in preview-index.json. Other ranks wait at symmetric barriers with a twenty-minute process-group timeout. Full adapter comparison may additionally be run as a separate finite inference job. This preparation includes the inference script; integration into the shared HTML is a later step once a real adapter exists.

Compare at fixed settings and seeds:
1. Base model plus the accepted full prompt.
2. Base model plus only the short instruction.
3. Base model plus the LoRA and the short instruction, trying adapter scales 0.5, 0.75 and 1.0 on validation only.

Use validation to select the checkpoint and scale, then evaluate that one choice on untouched test photographs. Look at likeness, pose/proportion, outlines, tonal range in clothing, light meaningful background, colour contamination, border smudges and colouring space. The training loss does not replace these visual judgements. A good result is comparable style and identity on new people with the short instruction, not merely reproducing training pictures or achieving a lower loss. If the adapter weakens likeness, try an earlier checkpoint or lower scale before increasing rank or training longer.

Licences and deployment scope
----------------------------
The pinned Qwen model is released under the Qwen Research Licence, which permits research/evaluation and requires a separate licence for commercial use. This is a concrete property of the chosen model, not an additional approval flow for the already authorised research preparation. Keep the saved licence and attribution with any adapter distribution; before a commercial product, obtain the required commercial licence or choose a base model whose terms fit that use. Photo/source rights are not established by this technical preparation.

Verification limits
-------------------
Local Python syntax, source hashes, plan-only launch behaviour and synthetic paired-dataset export/load are checked. No GPU training, CUDA dependency import, optimisation step or adapter inference has been executed by this preparation. Thus GPU memory fit, speed and trained adapter quality remain unverified until calibration.

Primary sources
---------------
Official paired trainer and parameter semantics:
https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/examples/dreambooth/README_qwenimage21.md#image-to-image-editing
https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/examples/dreambooth/train_dreambooth_lora_qwenimage21_img2img.py
Model architecture, inference recipe and licence:
https://huggingface.co/Qwen/Qwen-Image-2.1/tree/d26bb61231c349cf6b7896fa83353113880e1ba3
https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/LICENSE
Adapter loading:
https://huggingface.co/docs/diffusers/using-diffusers/loading_adapters

Current distributed implementation
----------------------------------
trainer_ddp.py is an attributed local derivative of the unchanged upstream vendor file; trainer-ddp.patch records changes. distributed_train.py owns all-rank checkpoint barriers, global mean loss, native preview publication, rank-specific completion evidence and short-prompt validation arguments. train_worker.py is the CUDA-free parent orchestrator and starts accelerate launch --multi_gpu --num_processes 8. run_identity.compute_run_key is shared by the launcher, parent and children; any runtime/trainer/config/dataset change invalidates calibration reuse. Native preview keys are previews/step-<six digits>-epoch-<four digits>-0.png and .json, indexed in preview-index.json at the run prefix. distributed-verification.json records local checks; actual DDP, VRAM and preview execution still require GPU calibration.

Live monitoring
---------------
Open http://127.0.0.1:8093/lora-v1/review.html for cloud state, actual logged updates, loss and held-out previews. orchestrate.py performs evidence checks after calibration and automatically launches the already authorised pilot. Twenty and two hundred count optimiser updates; each combines eight examples across eight GPUs. Calibration and pilot are separate runs from the frozen base. Pilot checkpoints are saved every 25 updates. The first installation attempt was cancelled before any optimisation; the retry uses /tmp for its Python environment, with models and recoverable checkpoints on persistent storage.
