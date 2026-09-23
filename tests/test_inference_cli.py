"""Check the standalone inference JSON contract without model weights."""

import json

import pytest

from predict_lakun import load_input


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
