# LaKun

[English](README.md) · [简体中文](README.zh-CN.md)

LaKun is a multimodal, option-scoring model for three decision types: **Choice** (select an option), **Score** (select an ordered rating bin), and **Noul** (a `false`/`true` decision). It accepts a text state or one image with one or more questions and returns a probability for each supplied option. It is **not** a free-form text generator.

The implementation jointly fine-tunes an [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) text encoder and a [SigLIP SO400M](https://huggingface.co/google/siglip-so400m-patch14-384) vision encoder. A visual-token bridge connects the vision output to a typed decision head inspired by Laya. No Laya fine-tuned checkpoint is loaded; the relevant Laya-derived source and its license are bundled in `lakun/vendor/laya/`.

> **Status:** Initial training is complete. The selected checkpoint is from optimizer step 16,882. Held-out test accuracy against **teacher pseudo-labels** was 83.11% (`choice`), 83.37% (`score`), and 80.92% (`noul`); these are not human-verified or general-purpose accuracy claims. An exploratory external spam-classification test showed poor cross-domain transfer, so evaluate LaKun on your own task before deployment.

## Dataset at a glance

The current local dataset has **126,663 groups** and **380,424 questions**: 59,996 image groups / 180,424 questions and 66,667 text groups / 200,000 questions. Each group is kept intact across the approximately 8:1:1 train/validation/test split. The three task types are close to equally represented. Image groups draw from seven source subsets; synthetic text groups cover 50 topics.

The answers are model-generated **pseudo-labels**, not human-verified ground truth. The [dataset profile and charts (Chinese)](analysis/DATASET_PROFILE.md) contain the counts, source breakdown, and known answer-position imbalance. Training/evaluation scores should be interpreted as agreement with these labels until independently checked. Dataset files and images are not included in the Git repository by default; review upstream dataset terms before distributing them.

## Layout

| Path | Purpose |
|---|---|
| `train_lakun.py` | One-command single-GPU or local multi-GPU training |
| `predict_lakun.py` | Score an input JSON or inspect one held-out dataset group |
| `lakun/` | Model, data loading, training, and inference code |
| `analysis/` | Dataset-only statistics and charts |
| `dataset/`, `images/` | Local train/validation/test data and referenced images |
| `weights/` | Original mmBERT and SigLIP weights used to start training |
| `runs/lakun_full/best/` | Best complete LaKun inference checkpoint after training |

## Install and train

Use Python 3.10+ with a CUDA-enabled PyTorch installation. On a server image where PyTorch is already available:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); assert torch.cuda.is_available()"
python -m pip install -r requirements-train.txt
python train_lakun.py --train-groups 12 --val-groups 6 --test-groups 6 --epochs 1 --out runs/lakun_pilot
python train_lakun.py
```

Place the original models at `weights/mmbert-base/` and `weights/siglip-so400m-patch14-384/`, and keep the six `dataset/{image,text}_{train,val,test}.jsonl` files plus referenced images in their relative locations. `train_lakun.py` uses all visible local GPUs by default and launches PyTorch DDP automatically when more than one is visible. Pass `--gpus N` to use a specific count or set `CUDA_VISIBLE_DEVICES` to choose devices. It does not combine GPUs from different servers.

Full training defaults to at most three epochs, validation every 3,000 optimizer steps, and early stopping after three validation checks without a sufficient loss improvement. The held-out test split is reserved for evaluation after model selection. The default output is `runs/lakun_full/`; the best **complete** checkpoint is `runs/lakun_full/best/`. Download that entire `best/` directory for inference—not only the `.safetensors` file. The checkpoint includes both encoders and the decision components, so inference does not separately load the two original base weights. Current checkpoints do not contain optimizer state and cannot resume an interrupted training run from the exact step.

## Inference

From this project directory, install the local code (no PyPI release is required):

```bash
python -m pip install -e .
```

With a complete checkpoint already present at `runs/lakun_full/best/`, run the included text-only example:

```bash
python predict_lakun.py --checkpoint runs/lakun_full/best --input examples/text_input.json
```

Edit the JSON to supply your own state and questions. For an image request, add an `image` path to that JSON; relative paths are resolved beside the JSON file. The checkpoint is not stored in this Git repository.

To inspect a held-out group when the local dataset is available:

```bash
python predict_lakun.py --checkpoint /path/to/your/checkpoint --modality image --group-index 0
```

Or call the library directly with your own image and questions (replace the image path):

```python
from lakun import LaKunPredictor

model = LaKunPredictor.from_pretrained("/path/to/your/checkpoint")
result = model.predict(
    [{"type": "choice", "question": "What is in the image?", "criteria": ["cat", "dog", "car"]}],
    image="/path/to/your/image.jpg",
)
print(result["answers"])
```

For text-only decisions, omit `image` and pass a `state` string. A group currently supports **one image** and 1–20 questions; multi-image input is not implemented. `noul` questions use `criteria=["false", "true"]`.

## Release and licensing

The [GitHub source repository](https://github.com/AI-Discussion-Room/LaKun), a future PyPI package, and a future ModelScope weight repository are separate releases. **As of this release, the GitHub repository is private; neither the PyPI package nor the ModelScope weights have been published.** The working examples above use a local code installation and a local checkpoint. Do not use `pip install lakun` or a ModelScope repo ID until those releases are verified. The source package never contains model weights or training data.

When publishing, upload the **contents** of `runs/lakun_full/best/` to the root of a ModelScope model repository, not the enclosing `best/` or `runs/` directory. Its root must include `model_config.json`, `lakun.safetensors`, `text_encoder/`, `vision_encoder/`, `tokenizer/`, and `image_processor/`. Keep the filename `lakun.safetensors`: the current loader expects it.

After a verified ModelScope release, install the optional `modelscope-hub` client, download that model to a local directory, then pass the directory to `LaKunPredictor.from_pretrained()`. The current loader does **not** accept a ModelScope repo ID directly. A concrete command with the actual repo ID will be added after publication.

The LaKun source code is Apache-2.0 licensed (see `LICENSE`); the copied Laya component keeps its own Apache-2.0 notice in `lakun/vendor/laya/LICENSE`. A future weight release needs its own model-card license and a review of upstream model/data terms. Do **not** upload the project root, original base weights, dataset, images, API keys, or an enclosing `runs/` directory to the model repository. This custom architecture is loaded through `lakun`, not directly through Transformers `AutoModel.from_pretrained()`.

For a more detailed Chinese startup guide, see [START.md](START.md).
