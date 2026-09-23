"""The validation cut must preserve groups and never edit the original test split."""
import json
from pathlib import Path

import pytest

from lakun.data import DecisionGroups
from lakun.split_data import split_dataset


def test_group_split_is_disjoint_and_preserves_test(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    before = {}
    for modality in ("image", "text"):
        for split, size in (("train", 18), ("test", 2)):
            path = dataset / f"{modality}_{split}.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for group_number in range(size):
                    for question_number in range(2):
                        row = {"id": f"{modality}:{split}:{group_number}:q{question_number}",
                               "group_id": f"{modality}:{split}:{group_number}",
                               "split": split, "images": [], "state": "s",
                               "type": "choice", "criteria": ["a", "b"],
                               "target": [1.0, 0.0], "selected_index": 0}
                        stream.write(json.dumps(row) + "\n")
            if split == "test":
                before[modality] = path.read_bytes()
    report = split_dataset(tmp_path, tmp_path / "backup")
    for modality in ("image", "text"):
        assert report["modalities"][modality]["groups"] == {"train": 16, "val": 2, "test": 2}
        assert (dataset / f"{modality}_test.jsonl").read_bytes() == before[modality]
        ids = {}
        for split in ("train", "val", "test"):
            rows = [json.loads(line) for line in (dataset / f"{modality}_{split}.jsonl").read_text().splitlines()]
            assert all(row["split"] == split for row in rows)
            assert len(rows) == report["modalities"][modality]["questions"][split]
            ids[split] = {row["group_id"] for row in rows}
            assert len(rows) == len(ids[split]) * 2
        assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])
    assert len(DecisionGroups(tmp_path, "val")) == 4
    with pytest.raises(FileExistsError):
        split_dataset(tmp_path, tmp_path / "another_backup")
