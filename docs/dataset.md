# Dataset format

`examples/dataset.example.json` is the small, custom input manifest used by
`lab.py`. It is **not** a built-in Diffusers format. Paths are relative to
`--data-root`; absolute paths, traversal and symlinks escaping that root are rejected.

Each pair needs an ID and an `inputs` array. A reference `target` is optional for
inference. Two inputs mean a side-by-side composite instruction, not a training
pair. Set `train_eligible: false` for diagnostic-only cases. Use simple IDs made
of letters, numbers, hyphens and underscores.

For training, provide:

- One source image and one approved target image.
- An explicit `split`: `train`, `validation` or `test`.
- A `group_id` connecting images derived from the same source/session.
- `identities` linking the same recognisable person across different sources.
  Use opaque labels if preferred. These are human-reviewed, not face recognition.

All splits must be non-empty. The portable validator rejects groups, identities,
input hashes **and target hashes** appearing across splits. It cannot infer missing
identity labels: those remain your responsibility. It does not assign random
splits or select targets automatically.

`--freeze` confirms the targets, groups and geometry have been reviewed. It calls
the preserved exporter to apply EXIF orientation, preserve complete images and
fit source and target to a shared white-letterboxed canvas. This avoids stretching
or cropping one image differently from its pair. Inspect alignment first: a
shared canvas cannot repair different poses, framing or mismatched targets.

The exporter writes one Parquet file per split with Hugging Face Datasets
features:

| Column | Type | Meaning |
| --- | --- | --- |
| `id` | string | Pair identifier |
| `caption` | string | Training instruction |
| `cond_image` | Image, embedded bytes | Source photograph |
| `image` | Image, embedded bytes | Selected drawing target |

The trainer loads these using `--dataset_name`, `--cond_image_column`,
`--image_column` and `--caption_column`. No model learns the custom manifest:
our preparation code translates it to this conventional paired-image dataset.
The frozen manifest also records image geometry, source hashes and selected targets.

The historical manifest and selection decisions are under `recipes/historical/`.
Their absolute dataset paths identify the original experiment, not a portable
location. The images are deliberately absent. Supply your own permitted images
or restore the original assets privately and verify their hashes.

The historical full-dataset diagnostic included previously held-out examples.
Those examples are no longer a blind test after reviewing and tuning on them.
Use fresh, untouched examples for a new final assessment.
