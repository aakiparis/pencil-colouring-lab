# Pencil colouring baseline on Nebius

The reviewed dataset is uploaded to private Nebius Object Storage in UK South 2. Project: pencil-colouring-baseline. Infrastructure details are recorded in the JSON resource files.

The user supplied a PDF print of the original conversation, accepted the shared instructions, and delegated the final prompt selection. run.json now contains the selected reconstructions; prompt-confirmed.json pins the configuration hash. The CPU warm-up completed successfully; GPU jobs use preemptible allocation. Full prompt evidence and rejected alternatives remain in the dataset prompt-review folder.

The intended model is Qwen/Qwen-Image-Edit-2511 at the revision in model.json, BF16, without any additional LoRA or speed adapter. The approved output images are comparison references only and must never condition the model. Model weights will be cached on a shared filesystem; inputs and results live in the private S3-compatible bucket. One independent pipeline per GPU processes a separate shard of images. Completed image outputs and metadata are published to S3 individually, enabling retries after preemption.

The run package contains approved single-input pairs and separately labelled multi-input compositions. Unpaired style references cannot be evaluated as image-edit pairs without the original text prompts. No training is performed.

Scripts: provision.py creates isolated resources, prepare.py packages reviewed data, storage_ops.py transfers files and configures secret references, launch.py starts finite jobs, worker.py downloads and verifies data or runs inference, bootstrap.sh prepares pinned Python dependencies inside an existing public PyTorch container. Credentials are private local files and MysteryBox secrets; never include them in a dataset or review page.

## Completed baseline

All 56 selected image-edit tasks completed and were downloaded with verified checksums. The review is available at http://127.0.0.1:8093/baseline-v1/review.html and in Downloads/Pencil colouring dataset/baseline-v1/review.html. The eight-GPU batch used preemptible RTX PRO 6000 instances. No compute instances remain in this project. The private bucket and shared model cache remain for subsequent experiments. The final compatible runtime uses diffusers 0.37.0; setup attempts excluded from the final experiment are recorded in execution-notes.txt.
