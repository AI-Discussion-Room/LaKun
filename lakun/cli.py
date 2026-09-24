"""Ask LaKun one typed question about text or one local image."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="向 LaKun 输入一道选择、评分或二元问题。省略参数时进入交互输入。",
        epilog="--checkpoint 可输入魔塔模型 ID 或完整本地权重目录；权重不随安装包提供。",
    )
    parser.add_argument("--checkpoint", help="魔塔模型 ID 或完整本地权重目录；不提供时交互输入")
    parser.add_argument("--image", type=Path, help="可选：单张本地图片路径（jpg/png 等）")
    parser.add_argument("--state", help="文本状态；可与 --image 同时提供")
    parser.add_argument("--type", dest="kind", choices=("choice", "score", "noul"), help="问题类型")
    parser.add_argument("--question", help="要提问的单个问题")
    parser.add_argument("--criteria", nargs="+", help="choice/score 候选项，按顺序列出")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="默认自动选择 GPU 或 CPU")
    return parser


def resolve_checkpoint(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    input_fn: Callable[[str], str] = input,
) -> str | Path:
    """Require a user-selected checkpoint; never guess the author's output path."""
    raw = args.checkpoint
    if raw is None:
        raw = input_fn("完整权重目录或魔塔模型 ID：")
    raw = str(raw).strip().strip('"')
    if not raw:
        parser.error("请提供完整权重目录或魔塔模型 ID；PyPI 包不包含权重")
    from .pretrained import modelscope_repo_id

    if modelscope_repo_id(raw) is not None:
        return raw
    checkpoint = Path(raw).expanduser().resolve()
    if not checkpoint.is_dir():
        parser.error(f"完整权重目录不存在：{checkpoint}")
    if not (checkpoint / "model_config.json").is_file():
        parser.error(f"这不是完整的 LaKun 检查点目录（缺少 model_config.json）：{checkpoint}")
    return checkpoint


def resolve_request(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    input_fn: Callable[[str], str] = input,
) -> tuple[list[dict], str, Path | None]:
    """Collect a single question, checking it before loading the large model."""
    interactive = all(value is None for value in
                      (args.image, args.state, args.kind, args.question, args.criteria))
    image = args.image
    if interactive:
        raw_image = input_fn("图片路径（直接回车使用纯文本）：").strip().strip('"')
        image = Path(raw_image) if raw_image else None
    if image is not None:
        image = image.expanduser().resolve()
        if not image.is_file():
            parser.error(f"图片不存在：{image}")

    state = args.state
    if state is None:
        state = input_fn("文本状态：").strip() if image is None else ""
    if not state and image is None:
        parser.error("纯文本任务需要 --state 或交互输入文本状态")

    kind = args.kind or input_fn("题型（choice / score / noul）：").strip().lower()
    if kind not in ("choice", "score", "noul"):
        parser.error("题型必须是 choice、score 或 noul")
    question = args.question or input_fn("问题：").strip()
    if not question:
        parser.error("问题不能为空")

    if kind == "noul":
        if args.criteria is not None:
            parser.error("noul 的候选项固定为 false/true，不需要 --criteria")
        criteria = ["false", "true"]
    elif args.criteria is not None:
        criteria = [item.strip() for item in args.criteria]
    else:
        criteria = [item.strip() for item in input_fn("候选项（用 | 分隔）：").replace("｜", "|").split("|")]
    minimum = 2 if kind == "choice" else 3 if kind == "score" else 2
    if not minimum <= len(criteria) <= 20 or any(not item for item in criteria):
        parser.error(f"{kind} 需要 {minimum}–20 个非空候选项")
    if len(set(criteria)) != len(criteria):
        parser.error("候选项不能重复")
    return [{"type": kind, "question": question, "criteria": criteria}], state, image


def main(argv: list[str] | None = None) -> None:
    # Windows pipe consumers (including some IDE terminals) commonly expect UTF-8.
    if sys.platform == "win32":
        for stream, original in ((sys.stdout, sys.__stdout__), (sys.stderr, sys.__stderr__)):
            if stream is original and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    checkpoint = resolve_checkpoint(args, parser)
    questions, state, image = resolve_request(args, parser)

    # 单图示例：运行 lakun，输入魔塔模型 ID，再输入本地图片路径和问题。
    from .inference import LaKunPredictor

    model = LaKunPredictor.from_pretrained(checkpoint, device=args.device)
    answer = model.predict(questions, state=state, image=image)["answers"][0]
    winner = answer["predicted_index"]
    print(f"答案：{answer['criteria'][winner]}（索引 {winner}）")
    print("候选项 softmax 分数（未经概率校准）：")
    for criterion, probability in zip(answer["criteria"], answer["probabilities"]):
        print(f"  {criterion}: {probability:.4f}")


if __name__ == "__main__":
    main()
