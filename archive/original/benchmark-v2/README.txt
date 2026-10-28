Pencil model benchmark v2

Models:
- Qwen/Qwen-Image-2.1 (full checkpoint, 40 inference steps, no guidance).
- black-forest-labs/flux-3-image-klein-base (full checkpoint, 50 steps, default guidance 3.5).
- The official FLUX prompt upsampler uses the source images; expanded captions are saved with every result.

All model revisions and the Diffusers source commit are pinned in each run.json. BF16 weights are used without added LoRA or quantisation. Both models receive the same corrected English edit instructions and original inputs. Approved ChatGPT drawings never condition inference. FLUX additionally expands the instructions through its official caption model. Single-input outputs retain the source aspect ratio at approximately one megapixel. Composite outputs use a landscape canvas, with the first subject on the left and the second on the right.

The previous Qwen-Image-Edit-2511 run is kept unchanged. Its dictated feedback, explicit preferences and attributed status interpretations are embedded in baseline-v1/review.html, stored in JSON and backed up to the existing private Object Storage bucket. The exact model used by ChatGPT to create the approved drawings is unknown.

prepare.py freezes the corrected prompt policy and model configurations.
launch.py launches finite CPU preparation or preemptible RTX PRO 6000 GPU jobs in Nebius UK South 2. Job records and creation operations are saved before monitoring. --retry is allowed only after a job has ended unsuccessfully.
control.py smoke prepares weights and runs portrait, product and two-input smoke cases. A successful smoke job is required before control.py batch starts eight independent GPU workers for the full dataset. Every finished output is published immediately; retries skip only results from the identical run configuration.
compare.py downloads results, verifies run, input and output hashes, and builds the labelled HTML comparison in the Downloads dataset folder. review.js updates displayed results without reloading user edits. New model reviews start pending and use separate browser storage from the old Qwen review. Export new reviews to save browser edits to a portable JSON file.

Credentials are read from local Hugging Face authorisation and the existing Nebius storage credential file, then supplied to cloud jobs through MysteryBox secrets. Credentials are never embedded in HTML, run.json or result metadata. Files containing secret responses are private and excluded from source control.

Commands from this directory use ../.venv/bin/python. No custom Docker image is built. GPU jobs terminate after completion or the configured finite timeout. A failed stage stops the controller for diagnosis; it does not launch the full batch automatically.

Runtime preparation note: the public container lacks Git, so the pinned Diffusers source is installed from its official GitHub archive. Its older system Brotli package is incompatible with the new HTTP client; repair_http.sh installs Brotli 1.2.0 into the isolated benchmark environment before resuming the FLUX download. Failed preparation jobs are retained as job records, and their compute is released by Nebius.

Qwen-Image-2.1 reprompt experiment (17 October 2026)
The user selected Qwen-Image-2.1 for likeness and retained the other models as history.
qwen21-feedback.json records shared comments explicitly applied across all Qwen 2.1 outputs. Needs improvement is an attributed interpretation, not a verbatim verdict enum.
Winner is hidden in the UI; historical preference data remains in JSON exports. Historical models can be shown with the checkbox. Existing local reviews keep their original browser-storage key.
qwen21_decodecheck is a diagnostic control with unchanged prompts, source images, sampling settings and seed, and VAE tiling disabled. Its outputs are retained separately.
qwen21_reprompt uses feedback-v3 prompts and disables VAE tiling; the model revision, seed and sampling settings are unchanged. There is no greyscale conversion, sharpening or image cleanup after generation. This is not a prompt-only comparison against the original run; the diagnostic controls isolate the tiling setting on selected examples.
Pair 045 uses the corrected source in the reprompt run. Historical outputs retain the previous source and must not be treated as a prompt-only comparison for this pair.
The new configuration declares every source checksum and downloads the corrected source from the private bucket if needed. Reuse of a cached model requires an exact model-revision match.
