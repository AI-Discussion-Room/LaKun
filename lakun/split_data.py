"""One-time, deterministic 80/10/10 group split of the existing 90/10 exports."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from .data import index_file


def split_dataset(root: Path, backup_dir: Path) -> dict:
    root = root.resolve()
    dataset = root / "dataset"
    backup_dir = backup_dir.resolve()
    if backup_dir == dataset or backup_dir.is_relative_to(dataset):
        raise ValueError("backup directory must be outside the active dataset directory")
    if backup_dir.exists():
        raise FileExistsError(f"backup already exists: {backup_dir}")
    if (dataset / "split_report.json").exists():
        raise FileExistsError("dataset has already been split")

    prepared = []
    report = {"strategy": "group_id_sha256_v1", "seed": "lakun-validation-20260923",
              "fractions": {"train": 0.8, "val": 0.1, "test": 0.1}, "modalities": {}}
    for modality in ("image", "text"):
        train_path = dataset / f"{modality}_train.jsonl"
        test_path = dataset / f"{modality}_test.jsonl"
        train_spans = index_file(train_path, "train", modality, None)
        test_spans = index_file(test_path, "test", modality, None)
        train_ids = {span.group_id for span in train_spans}
        test_ids = {span.group_id for span in test_spans}
        if len(train_ids) != len(train_spans) or len(test_ids) != len(test_spans) or train_ids & test_ids:
            raise ValueError(f"duplicate or overlapping {modality} group ids")
        count = round((len(train_spans) + len(test_spans)) * 0.1)
        if count < 1 or count >= len(train_spans):
            raise ValueError(f"not enough {modality} groups to form validation split")
        def order(group_id: str) -> bytes:
            return hashlib.sha256(f"lakun-validation-20260923:{modality}:{group_id}".encode()).digest()
        val_ids = set(sorted(train_ids, key=order)[:count])
        train_temp = dataset / f".{modality}_train.new.jsonl"
        val_temp = dataset / f".{modality}_val.new.jsonl"
        if train_temp.exists() or val_temp.exists() or (dataset / f"{modality}_val.jsonl").exists():
            raise FileExistsError(f"temporary or validation {modality} file already exists")
        question_counts = {"train": 0, "val": 0, "test": 0}
        with train_path.open("rb") as source, train_temp.open("wb") as train_out, val_temp.open("wb") as val_out:
            for span in train_spans:
                source.seek(span.start)
                raw = source.read(span.end - span.start)
                if span.group_id in val_ids:
                    for line in raw.splitlines():
                        row = json.loads(line)
                        row["split"] = "val"
                        val_out.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
                        question_counts["val"] += 1
                else:
                    train_out.write(raw)
                    question_counts["train"] += len(raw.splitlines())
        with test_path.open("rb") as stream:
            question_counts["test"] = sum(1 for _ in stream)
        if len(index_file(train_temp, "train", modality, None)) != len(train_spans) - count:
            raise RuntimeError(f"new {modality} train index mismatch")
        if len(index_file(val_temp, "val", modality, None)) != count:
            raise RuntimeError(f"new {modality} validation index mismatch")
        report["modalities"][modality] = {
            "groups": {"train": len(train_spans) - count, "val": count, "test": len(test_spans)},
            "questions": question_counts}
        prepared.append((train_path, train_temp, dataset / f"{modality}_val.jsonl", val_temp))

    backup_dir.mkdir(parents=True)
    for train_path, _, _, _ in prepared:
        shutil.copy2(train_path, backup_dir / train_path.name)
    for train_path, train_temp, val_path, val_temp in prepared:
        os.replace(train_temp, train_path)
        os.replace(val_temp, val_path)
    (dataset / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--backup-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(split_dataset(args.project_root, args.backup_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
