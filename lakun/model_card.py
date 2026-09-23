"""Write a conservative, self-contained Hugging Face card with each checkpoint."""
from __future__ import annotations

from pathlib import Path


def write_model_card(output: Path, report: dict) -> None:
    mode = report["mode"]
    metrics = report["test"]
    architecture = ("Tiny randomly initialized BERT/SigLIP-compatible encoders"
                    if mode == "tiny_random_smoke" else
                    "mmBERT-base text encoder + SigLIP SO400M vision encoder")
    base_models = ("" if mode == "tiny_random_smoke" else
                   "base_model:\n- jhu-clsp/mmBERT-base\n- google/siglip-so400m-patch14-384\n")
    card = f"""---
tags:
- multimodal
- image-classification
- text-classification
- typed-decisions
{base_models}---

# LaKun multimodal typed decisions

Checkpoint mode: **{mode}**. This model scores caller-supplied options for
`choice`, ordered `score`, and binary `noul` questions. It does not generate text.

Architecture: {architecture} + 64
trainable visual query tokens + copied Laya typed-decision head. No Laya
fine-tuned checkpoint was used. The Laya-derived source is Apache-2.0;
see the LaKun source repository's bundled Laya license.

Available LaKun data: teacher-generated pseudo-labels from 59,996 image
groups / 180,424 image questions and 200,000 synthetic text questions.
This checkpoint actually saw {report['history'][-1]['train']['groups']} training
groups and {report['history'][-1]['train']['questions']} training questions in
its last epoch. Model selection used validation loss; the test set was evaluated
once after selecting the best checkpoint. The recorded test metrics are against **teacher
labels**, not human gold.

Best validation loss: {report['best_val_loss']} at optimizer step {report['best_step']}.
Final test loss: {metrics['loss']}; test accuracy by type: {metrics['accuracy']}.
These numbers are not a human-verified quality or probability-calibration claim.
If the mode is `tiny_random_smoke`, the weights are a randomly initialized
pipeline fixture and must **not** be presented as a trained model.

Files in this directory are self-contained weights and processors; the custom
fusion/head implementation still requires the `lakun` Python package.
After installing that code, use:

```python
from lakun import LaKunPredictor
model = LaKunPredictor.from_pretrained("/path/to/this-directory")
result = model.predict(
    [{{"type": "choice", "question": "What is shown?", "criteria": ["cat", "dog", "car"]}}],
    image="example.jpg",
)
print(result["answers"])
```

The project owner must choose and document the LaKun derivative-weight license
and data provenance before public release. Do not include API keys, images,
or training data in the model repository by default.
"""
    (output / "README.md").write_text(card, encoding="utf-8")
