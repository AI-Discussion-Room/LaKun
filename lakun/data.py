"""Read the existing flat JSONL exports as image or text decision groups."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset


@dataclass(frozen=True)
class GroupSpan:
    path: Path
    start: int
    end: int
    group_id: str
    split: str
    modality: str


def index_file(path: Path, split: str, modality: str, limit: int | None) -> list[GroupSpan]:
    if not path.is_file():
        raise FileNotFoundError(path)
    spans: list[GroupSpan] = []
    seen: set[str] = set()
    current_id: str | None = None
    group_start = 0
    with path.open("rb") as stream:
        while True:
            line_start = stream.tell()
            raw = stream.readline()
            if not raw:
                if current_id is not None:
                    spans.append(GroupSpan(path, group_start, line_start, current_id, split, modality))
                break
            row = json.loads(raw)
            if row["split"] != split:
                raise ValueError(f"wrong split in {path}: {row['split']}")
            group_id = row["group_id"]
            if group_id != current_id:
                if current_id is not None:
                    spans.append(GroupSpan(path, group_start, line_start, current_id, split, modality))
                    if limit is not None and len(spans) >= limit:
                        break
                if group_id in seen:
                    raise ValueError(f"noncontiguous group {group_id} in {path}")
                seen.add(group_id)
                current_id = group_id
                group_start = line_start
    if limit is not None and len(spans) < limit:
        raise ValueError(f"{path} has {len(spans)} groups, needs {limit}")
    return spans


class DecisionGroups(Dataset):
    """One dataset item is one image plus all its questions, or one text state."""

    def __init__(self, root: Path, split: str, max_groups: int | None = None):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        if max_groups is not None and max_groups < 2:
            raise ValueError("max_groups must be at least 2 to include both modalities")
        self.root = root.resolve()
        out = self.root / "dataset"
        image_limit = (max_groups + 1) // 2 if max_groups is not None else None
        text_limit = max_groups // 2 if max_groups is not None else None
        images = index_file(out / f"image_{split}.jsonl", split, "image", image_limit)
        texts = index_file(out / f"text_{split}.jsonl", split, "text", text_limit)
        self.spans = images + texts

    def __len__(self) -> int:
        return len(self.spans)

    def __getitem__(self, index: int) -> dict:
        span = self.spans[index]
        with span.path.open("rb") as stream:
            stream.seek(span.start)
            rows = [json.loads(line) for line in stream.read(span.end - span.start).splitlines()]
        if not rows or any(row["group_id"] != span.group_id for row in rows):
            raise ValueError(f"invalid group span: {span.group_id}")
        images = rows[0]["images"]
        if any(row["images"] != images or row["state"] != rows[0]["state"] for row in rows):
            raise ValueError(f"inconsistent rows in {span.group_id}")
        if (span.modality == "image" and len(images) != 1) or (span.modality == "text" and images):
            raise ValueError(f"wrong image count in {span.group_id}")
        for row in rows:
            criteria = row["criteria"]
            target = row["target"]
            if (row["type"] not in {"choice", "score", "noul"}
                    or len(criteria) != len(target)
                    or not 0 <= row["selected_index"] < len(criteria)
                    or abs(sum(target) - 1.0) > 1e-4):
                raise ValueError(f"invalid decision {row['id']}")
        image_path = None
        if images:
            image_path = (self.root / images[0]).resolve()
            if not image_path.is_relative_to(self.root) or not image_path.is_file():
                raise FileNotFoundError(image_path)
        return {"group_id": span.group_id, "modality": span.modality,
                "state": rows[0]["state"], "image_path": image_path, "rows": rows}
