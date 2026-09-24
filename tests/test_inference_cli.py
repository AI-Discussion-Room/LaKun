"""Check the standalone inference JSON contract without model weights."""

import json

import pytest
import torch

from lakun.inference import LaKunPredictor
from predict_lakun import load_input


def test_predict_returns_json_serializable_data_without_cli_text(monkeypatch, capsys):
    import lakun.inference as inference

    monkeypatch.setattr(inference, "prepare_group", lambda *args: (None, None))
    predictor = LaKunPredictor.__new__(LaKunPredictor)
    predictor.model = lambda batch, pixels: torch.tensor([[1.0, 3.0]])
    predictor.device = torch.device("cpu")
    predictor.model_status = "trained_checkpoint"
    predictor.tokenizer = None
    predictor.processor = None

    result = predictor.predict(
        [{"type": "choice", "question": "哪一个？", "criteria": ["1", "2"]}],
        state="测试",
    )
    assert result["answers"][0]["predicted_index"] == 1
    assert isinstance(json.dumps(result, ensure_ascii=False), str)
    assert capsys.readouterr().out == ""


def test_input_resolves_image_relative_to_json(tmp_path):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({
        "state": "A short scene.",
        "image": "scene.png",
        "questions": [{"type": "choice", "question": "What is shown?",
                       "criteria": ["person", "animal"]}],
    }), encoding="utf-8")
    questions, state, image = load_input(request)
    assert len(questions) == 1
    assert state == "A short scene."
    assert image == tmp_path / "scene.png"


def test_input_rejects_missing_questions(tmp_path):
    request = tmp_path / "request.json"
    request.write_text('{"state": "text"}', encoding="utf-8")
    with pytest.raises(ValueError, match="questions array"):
        load_input(request)
