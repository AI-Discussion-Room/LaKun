"""Score caller-supplied questions or inspect one held-out LaKun group."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lakun.data import DecisionGroups
from lakun.inference import LaKunPredictor


def load_input(path: Path) -> tuple[list[dict], str, Path | None]:
    """Load a standalone inference request; resolve its image beside the JSON file."""
    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("questions"), list):
        raise ValueError("input JSON must be an object with a questions array")
    state = payload.get("state", "")
    if not isinstance(state, str):
        raise ValueError("input state must be a string")
    image = payload.get("image")
    if image is not None and (not isinstance(image, str) or not image):
        raise ValueError("input image must be a nonempty path string")
    image_path = (source.parent / image).resolve() if image is not None else None
    return payload["questions"], state, image_path


def main():
    parser = argparse.ArgumentParser(description="Predict from JSON or inspect one test group")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="trained LaKun checkpoint directory")
    source.add_argument("--base-weights", action="store_true",
                        help="load original towers with a RANDOM head; wiring test only")
    parser.add_argument("--text-model", type=Path)
    parser.add_argument("--vision-model", type=Path)
    parser.add_argument("--modality", choices=("image", "text"), default="image")
    parser.add_argument("--group-index", type=int, default=0)
    parser.add_argument("--input", type=Path,
                        help="JSON with questions, optional state, and optional image path")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.group_index < 0:
        raise ValueError("group-index must be nonnegative")
    root = Path(__file__).resolve().parent
    if args.base_weights:
        text_model = args.text_model or root / "weights" / "mmbert-base"
        vision_model = args.vision_model or root / "weights" / "siglip-so400m-patch14-384"
        predictor = LaKunPredictor.from_base_models(text_model, vision_model, device=args.device)
    else:
        predictor = LaKunPredictor.from_pretrained(args.checkpoint, device=args.device)
    if args.input is not None:
        questions, state, image = load_input(args.input)
        result = predictor.predict(questions, state=state, image=image)
    else:
        dataset = DecisionGroups(root, "test", max_groups=2 * (args.group_index + 1))
        group = dataset[args.group_index if args.modality == "image"
                        else args.group_index + (args.group_index + 1)]
        group["from_dataset"] = True
        result = predictor.predict_group(group)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
