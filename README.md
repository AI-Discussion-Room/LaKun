# LaKun: multimodal JEV typed decisions

English · [简体中文](README.zh-CN.md) · [Apache-2.0 license](LICENSE) · [Dataset profile and charts](analysis/DATASET_PROFILE.md)

JEV inspired this project, and I also drew on optimization ideas from [Laya](https://github.com/mizorewww/laya-mlx) (RLCD). LaKun is my multimodal take on a JEV typed-decision model: given a text state or an image, it answers my specified choice, rating-bin, and binary questions. It returns a softmax score for every option and the highest-scoring option, rather than generating free-form text. One call can mix all three question types, with up to 20 questions and one image.

## At a glance

| Item | Current implementation / measurement |
|---|---|
| Input | A text state or one image, with 1–20 typed questions |
| Questions | `choice`: 2–20 options; `score`: 3–20 bins; `noul`: `false/true` |
| Output | Per-question `criteria`, `probabilities`, and `predicted_index`; probability calibration is **not validated** |
| Architecture | mmBERT-base + SigLIP SO400M + visual-token bridge + typed decision head |
| Parameter count | **756,409,158 (~0.756B)** across 627 checkpoint tensors; `0.7B` is an approximate repository name |
| Context | At most 512 tokens, reserving 64 visual tokens for images; overlength inputs are truncated, prioritizing question/options and the start of the state |

I jointly fine-tuned the text and vision encoders. I did **not** load a Laya fine-tuned checkpoint; the decision head uses Laya-derived code with its notice preserved. LaKun's results are not Laya's results.

## Quick start

Install a PyTorch build appropriate for your machine, then install my code package from PyPI. Pass my ModelScope model ID to automatically download and cache the complete checkpoint on first use; access follows the model repository's current settings. If access is restricted, log in with `ms-hub login` first. The CLI then prompts for your text state or image path and question:

```bash
python -m pip install lakun
lakun --checkpoint hh108801/LaKun-0.7B
```

If you have access to my source checkout, the equivalent entry point is:

```bash
python -m pip install -e .
python main.py
```

The command prints the top option and all softmax scores. You may also pass a complete local checkpoint directory to `--checkpoint`; it must contain `model_config.json`, weight files, both encoder configurations, the tokenizer, and the image processor. For non-interactive use, specify `--state` or `--image` along with one typed question.

Here is the text-state API with all three question types in one request:

```python
from lakun import LaKunPredictor

model = LaKunPredictor.from_pretrained("hh108801/LaKun-0.7B")
result = model.predict(
    [
        {"type": "choice", "question": "What kind of request is this?", "criteria": ["question", "complaint", "refund"]},
        {"type": "score", "question": "How urgent is it?", "criteria": ["low", "medium", "high"]},
        {"type": "noul", "question": "Does the user request a refund?", "criteria": ["false", "true"]},
    ],
    state="The customer says they were charged twice and asks for a prompt refund.",
)
for answer in result["answers"]:
    print(answer["type"], answer["predicted_index"], answer["probabilities"])
```

### Use your own image

Replace `image_path` with a path to an image that **exists on your machine**. This call asks a choice, rating-bin and binary question about the same image; the example does not assume what the image contains:

```python
import json
from lakun import LaKunPredictor

model = LaKunPredictor.from_pretrained("hh108801/LaKun-0.7B")
image_path = "path/to/your_image.jpg"  # Replace with your image path
result = model.predict(
    [
        {"type": "choice", "question": "What is the main subject?", "criteria": ["cat", "dog", "car"]},
        {"type": "score", "question": "How clear is the image?", "criteria": ["blurry", "average", "clear"]},
        {"type": "noul", "question": "Is there an animal in the image?", "criteria": ["false", "true"]},
    ],
    image=image_path,
)
print(json.dumps(result, ensure_ascii=False, indent=2))
```

`predict()` returns a structured Python dictionary that can be serialized as JSON. The CLI's human-readable “答案：…” lines are not part of that API result.

`predicted_index` is zero-based and refers to the supplied option order. For `score`, it is the winning **bin index**, not a continuous regression output. The probabilities are softmax values, not calibrated real-world correctness probabilities.

For overlength inputs, the encoder keeps the question and options first (shortening very long options if necessary), then as much of the **beginning of the state** as fits. Discarded trailing text is not considered. Summarize or split long documents if the important evidence appears near the end.

## What I measured

I split groups—not individual questions—between training, validation and test. I selected the checkpoint at optimizer step **16,882** using validation loss. On **12,703 held-out groups / 38,141 questions**, agreement with the teacher's pseudo-labels was:

| Type | Agreement |
|---|---:|
| Choice | 83.11% |
| Score | 83.37% |
| Noul | 80.92% |

The labels were generated by `qwen3.8-flash`, not verified by people. These are **not** general-purpose accuracy, reasoning, or probability-calibration scores. Best validation loss was `0.014464`. During the second epoch, training loss kept falling while validation loss rose, so I retained the earlier checkpoint and stopped training.

### Local inference speed

I measured the complete `LaKunPredictor.predict()` call on **Windows 11**, an **NVIDIA GeForce RTX 3060 12GB**, **PyTorch 2.7.1+cu126**, using FP32 inference. For each scenario I ran 5 warm-ups and 30 sequential timed calls, synchronizing CUDA before and after timing. The measurement includes tokenization, image decoding/preprocessing where applicable, forward pass, and result construction. It **excludes** checkpoint loading, download, and network overhead. This is a fixed-input latency test, not an accuracy test.

| Scenario | Mean latency | p95 latency |
|---|---:|---:|
| Text · one question | 17.33 ms | 19.23 ms |
| Text · three questions | 18.19 ms | 18.65 ms |
| One image · one question | 163.06 ms | 168.69 ms |
| One image · three questions | 168.88 ms | 171.59 ms |

## My dataset

I unified images and text into **one state + multiple typed questions** per group. All question labels were generated by `qwen3.8-flash`. The current profile contains **126,663 groups / 380,424 questions**: 59,996 image groups / 180,424 questions and 66,667 text groups / 200,000 questions. Choice, score and noul occur at similar frequencies. The approximately 8:1:1 split keeps every group's questions together.

![Dataset size, question types and splits](analysis/figures/01-overview.png)

The profile also covers option counts, rating bins, seven image source subsets and 50 synthetic text topics. Among four-option questions, option B is the teacher's answer in **53.15%** of image questions and **46.48%** of text questions; D accounts for **1.74%** and **2.98%**, respectively.

![Option and pseudo-label distributions](analysis/figures/02-options-and-labels.png)

[Source and coverage chart](analysis/figures/03-coverage.png) · [All 50 text topics and answer positions](analysis/figures/04-domains-and-choice-labels.png) · [Full dataset profile](analysis/DATASET_PROFILE.md). The chart labels are currently Chinese.

## Training and reproducibility

`train_lakun.py` needs Python 3.10+, a CUDA-enabled PyTorch build, the dependencies declared in `pyproject.toml`, the original base weights under `weights/mmbert-base/` and `weights/siglip-so400m-patch14-384/`, plus six `dataset/{image,text}_{train,val,test}.jsonl` files and their referenced images. I trained with `transformers==4.57.3`. I recommend a small pipeline run before full training:

```bash
python -m pip install -e .
python train_lakun.py --train-groups 12 --val-groups 6 --test-groups 6 --epochs 1 --out runs/lakun_pilot
python train_lakun.py
```

The script uses visible local GPUs (DDP with multiple GPUs), checks validation every 3,000 optimizer steps, runs at most three epochs, and stops after three checks without sufficient loss improvement. The complete `best/` directory contains both encoders and the custom decision components; do not download just the `.safetensors` file. Current checkpoints omit optimizer state and cannot resume the precise interrupted training step. See [`START.md`](START.md).

## Limitations I am seeing

Within a distribution close to my training data, LaKun is already useful for these typed decisions. In informal use, I find questions with explicit numerical cues or simple inference between supplied options easier for it, **but I have not run a separate controlled benchmark for that observation**. An external spam-classification experiment exposed weak cross-domain transfer. Falling training loss alongside worse validation loss also makes me concerned about over-specialization or a form of “fine-tuning collapse.” I cannot yet prove that the pretrained backbone itself has lost capability; that needs comparisons with the unfine-tuned base, other checkpoints, and human-labeled out-of-domain tests.

I therefore do not claim this is a general classifier, a strongly generalizing vision model, or a verified numerical-reasoning model. One-image input, answer-position bias, pseudo-label error, uncalibrated scores, cross-domain behavior, and data provenance remain open limitations.
