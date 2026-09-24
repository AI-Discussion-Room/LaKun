---
tags:
- multimodal
- typed-decisions
- safetensors
base_model:
- jhu-clsp/mmBERT-base
- google/siglip-so400m-patch14-384
---

# LaKun-0.7B: a multimodal JEV-style typed-decision model

JEV inspired this project, and I also drew on optimization ideas from [Laya](https://github.com/mizorewww/laya-mlx). LaKun is my multimodal take on a JEV typed-decision model: given a text state or an image, it answers my specified choice, rating-bin, and binary questions. It returns a softmax score for every option and the highest-scoring option, rather than generating free-form text. One call can mix all three question types, with up to 20 questions and one image.

English · [简体中文](README.md) · [Source and dataset profile](https://github.com/AI-Discussion-Room/LaKun) · [Apache-2.0 source license](https://github.com/AI-Discussion-Room/LaKun/blob/main/LICENSE)

## Model facts

| Item | This checkpoint |
|---|---|
| Input | A text state or one image, with 1–20 typed questions |
| Questions | `choice`: 2–20 options; `score`: 3–20 bins; `noul`: `false/true` |
| Output | Per-question `criteria`, `probabilities`, and zero-based `predicted_index` |
| Encoders | [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) and [SigLIP SO400M](https://huggingface.co/google/siglip-so400m-patch14-384) |
| Architecture | 64 visual-query tokens and a typed decision head; I jointly fine-tuned both encoders |
| Parameters | **756,409,158 (~0.756B)**; `0.7B` is an approximate repository name |
| Weight size | `lakun.safetensors`: **3,025,715,432 bytes (2.82 GiB)**; complete inference directory: about **2.85 GiB** |
| Context | At most 512 tokens, reserving 64 for image queries; overlength input is truncated, keeping the question/options and the start of the state first |
| Selected optimizer step | **16,882**, selected by validation loss |

I did not use Laya's trained weights. I reused some decision-head code and preserved its license notice. This is a complete checkpoint, including the text and vision encoders, bridge and decision head. Keep the model configuration, tokenizer, image processor and both encoder configurations with the weight file. Transformers `AutoModel.from_pretrained()` cannot load this custom architecture directly.

## Use it

Install a PyTorch build appropriate for your machine, then install my code package from PyPI. The weights live separately in this model repository. Passing the repository ID downloads and caches the complete checkpoint on first use:

```bash
python -m pip install lakun
lakun --checkpoint hh108801/LaKun-0.7B
```

If repository access is restricted, sign in with `ms-hub login` first. The same repository ID works in Python:

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

I currently support one image per request. `predict()` returns a structured Python dictionary; CLI prose such as “answer” is not part of the API result. A `score` answer is a selected bin index, not a continuous regression score. I have not validated probability calibration, so a softmax value is **not** a verified real-world probability of correctness.

For long inputs, I keep the question and options first, then as much of the **beginning of the state** as fits. Discarded trailing text cannot affect the decision; summarize or split a long document if its important evidence appears at the end.

## Evaluation

I split by group, not by individual question, and selected the checkpoint solely by validation loss. Best validation loss was `0.014464`. On **12,703 held-out groups / 38,141 questions**, the top option agreed with the `qwen3.8-flash` teacher pseudo-label at these rates:

| Type | Held-out agreement |
|---|---:|
| Choice | 83.11% |
| Score | 83.37% |
| Noul | 80.92% |

These are **not** accuracies against human-verified ground truth, generalization scores or evidence of calibrated probabilities. During epoch two, training loss continued to fall while validation loss rose, so I retained the earlier checkpoint.

### Inference speed

I measured complete `predict()` calls on **Windows 11, RTX 3060 12GB, PyTorch 2.7.1+cu126, FP32**. Each scenario had 5 warm-ups followed by 30 sequential timed runs with CUDA synchronization. Timing includes tokenization, image decoding/preprocessing where applicable, forward computation and output construction; it excludes checkpoint loading, download and network time. This fixed-input latency test is not an accuracy test, and other hardware or input lengths may differ.

| Scenario | Mean latency | p95 latency |
|---|---:|---:|
| Text · one question | 17.33 ms | 19.23 ms |
| Text · three questions | 18.19 ms | 18.65 ms |
| One image · one question | 163.06 ms | 168.69 ms |
| One image · three questions | 168.88 ms | 171.59 ms |

## My dataset: aggregate statistics only

I formatted both images and text as **one state plus multiple typed questions per group**. All question labels were generated by `qwen3.8-flash`; they are pseudo-labels, not verified human answers. The current dataset profile contains **126,663 groups / 380,424 questions**: 59,996 image groups / 180,424 questions and 66,667 text groups / 200,000 questions. My approximately 8:1:1 split keeps a group's questions together.

| Type | Image questions | Text questions | Total |
|---|---:|---:|---:|
| Choice | 60,401 | 66,667 | 127,068 |
| Score | 60,015 | 66,667 | 126,682 |
| Noul | 60,008 | 66,666 | 126,674 |

![Dataset size, question types and splits](assets/01-overview.png)

I also analyzed option counts, rating bins, seven image source subsets and 50 synthetic text topics. In four-option questions, answer B accounts for **53.15%** of image labels and **46.48%** of text labels, while D accounts for only **1.74%** and **2.98%**. The model may exploit this positional bias rather than the input content.

![Option counts and pseudo-label distributions](assets/02-options-and-labels.png)

[Source and coverage chart](assets/03-coverage.png) · [Text-topic and answer-position chart](assets/04-domains-and-choice-labels.png) · [Full dataset profile](https://github.com/AI-Discussion-Room/LaKun/blob/main/analysis/DATASET_PROFILE.md). The chart labels are currently Chinese.

I have not uploaded the raw training JSONL, images or the 1,000 inspected test examples. These figures contain aggregates only. Upstream image and data redistribution terms need further review.

## Limitations I am seeing

Within a distribution close to my training data, LaKun is useful for these typed decisions. In informal use, I find questions with explicit numerical cues or simple inference between supplied options easier for it, **but I have not run a separate controlled benchmark for that observation**.

An external spam-classification experiment exposed weak cross-domain transfer. Falling training loss alongside worsening validation loss makes me concerned about over-specialization or a form of “fine-tuning collapse”; I cannot yet prove that the pretrained backbone itself lost its capabilities. I need comparisons with the unfine-tuned base, other checkpoints and human-labeled out-of-domain tests. One-image input, answer-position bias, teacher-label errors and uncalibrated softmax scores also limit the model. I do not present it as a general classifier, a verified numerical-reasoning model or a strongly generalizing vision model.
