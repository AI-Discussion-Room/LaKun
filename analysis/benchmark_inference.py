"""Measure end-to-end LaKunPredictor.predict latency on one local device.

Includes tokenization, image decoding/preprocessing (when used), model forward,
softmax and Python result construction. Excludes checkpoint loading.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from lakun import LaKunPredictor


QUESTIONS = [
    {"type": "choice", "question": "这是什么类型的请求？", "criteria": ["咨询", "投诉", "退款"]},
    {"type": "score", "question": "这件事有多紧急？", "criteria": ["不紧急", "一般", "紧急"]},
    {"type": "noul", "question": "用户是否要求退款？", "criteria": ["false", "true"]},
]
IMAGE_QUESTIONS = [
    {"type": "choice", "question": "图中领带是什么颜色？", "criteria": ["红色", "蓝色", "黑色", "黄色"]},
    {"type": "score", "question": "图片有多清晰？", "criteria": ["模糊", "一般", "清晰"]},
    {"type": "noul", "question": "图中人物是否佩戴领带？", "criteria": ["false", "true"]},
]
STATE = "用户说：我被重复扣费了，请尽快退款。"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, help="one local image for the image scenarios")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be >= 0 and repeats must be >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    begin = time.perf_counter()
    predictor = LaKunPredictor.from_pretrained(args.checkpoint, device=args.device)
    if args.device == "cuda":
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - begin
    cases = [("text_1q", QUESTIONS[:1], None), ("text_3q", QUESTIONS, None)]
    if args.image:
        cases.extend([("image_1q", IMAGE_QUESTIONS[:1], args.image),
                      ("image_3q", IMAGE_QUESTIONS, args.image)])

    output = {
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU",
        "torch": torch.__version__,
        "checkpoint": str(args.checkpoint.resolve()),
        "load_seconds": round(load_seconds, 3),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "method": "predict() wall time; CUDA synchronized before/after; sequential requests",
        "cases": {},
    }
    for name, questions, image in cases:
        kwargs = {"state": "" if image else STATE, "image": image}
        for _ in range(args.warmup):
            predictor.predict(questions, **kwargs)
        if args.device == "cuda":
            torch.cuda.synchronize()
        timings = []
        for _ in range(args.repeats):
            if args.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = predictor.predict(questions, **kwargs)
            if args.device == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
        if len(result["answers"]) != len(questions):
            raise RuntimeError("unexpected answer count")
        output["cases"][name] = {
            "median_ms": round(statistics.median(timings), 2),
            "p95_ms": round(percentile(timings, 0.95), 2),
            "mean_ms": round(statistics.mean(timings), 2),
        }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
