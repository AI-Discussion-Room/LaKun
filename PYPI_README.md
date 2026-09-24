# LaKun

I built LaKun to answer caller-defined **choice**, ordered **score**, and binary **noul** questions about a text state or one local image. It returns a softmax score for each supplied option; it is not a free-form text generator. This package contains inference and training code, **not** the 2.82 GiB checkpoint or any dataset.

The selected checkpoint has 756,409,158 parameters. On the creator's Windows 11 / RTX 3060 12GB machine, warm `predict()` calls had median latency of 17.01 ms for one text question and 162.01 ms for one image question (30 sequential calls after five warm-ups, FP32, PyTorch 2.7.1+cu126). These are device-specific measurements, not throughput or accuracy claims.

## Install and run

Install a PyTorch build suited to your system, then install this package **after its PyPI release**:

```bash
python -m pip install lakun
```

You also need a **complete local LaKun checkpoint directory**. `lakun` does not download weights by itself, and `LaKunPredictor.from_pretrained()` expects a local directory rather than a ModelScope repository ID. The checkpoint is currently held in a private ModelScope repository; public weight access and its license have not yet been finalized. Installing this package alone does not make the model usable without a checkpoint.

With a checkpoint available locally, ask a single question from the command line:

```bash
lakun --checkpoint /path/to/LaKun-0.7B --state "The order has been refunded." --type noul --question "Has the order been refunded?"
# Image example: replace photo.jpg with your own image path
lakun --checkpoint /path/to/LaKun-0.7B --image photo.jpg --type choice --question "What is shown?" --criteria cat dog car
```

Or use the Python API:

```python
from lakun import LaKunPredictor

model = LaKunPredictor.from_pretrained("/path/to/LaKun-0.7B")
result = model.predict(
    [{"type": "choice", "question": "What kind of request is this?", "criteria": ["question", "complaint", "refund"]}],
    state="I was charged twice. Please refund the duplicate payment.",
)
print(result["answers"][0])
```

`choice` accepts 2–20 options, `score` accepts 3–20 bins, and `noul` uses fixed `false/true` options. One call can ask up to 20 questions, but currently accepts only one image. Long inputs are truncated to the 512-token context: questions/options take priority, followed by the beginning of the state. The softmax scores have not been validated as calibrated real-world probabilities.

## Evaluation and limits

Against held-out **teacher-generated pseudo-labels**, the selected checkpoint agreed on 83.11% of choice, 83.37% of score, and 80.92% of noul questions (12,703 groups / 38,141 questions). These are not human-verified or general-purpose accuracy figures. The data were generated with `qwen3.8-flash`; answer-position bias and weak cross-domain transfer remain important limitations.

I observe good behavior on questions close to my training distribution, particularly some with explicit numerical cues or simple inference between given options, but I have not independently benchmarked that observation. An external spam-classification experiment was much weaker, and worsening validation loss in the second training epoch suggests over-specialization or possible fine-tuning collapse. That is **not yet proof** that the base model lost its general capability.

Source code is Apache-2.0 licensed. Model weights require a separate license decision. The GitHub source and ModelScope weight repositories are currently private; do not interpret the code license as permission to redistribute the weights.
