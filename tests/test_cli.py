"""Single-question CLI input and output contract."""

from pathlib import Path

import pytest

from lakun.cli import build_parser, main, resolve_checkpoint, resolve_request


def test_checkpoint_has_no_author_specific_default(tmp_path: Path):
    parser = build_parser()
    assert parser.parse_args([]).checkpoint is None
    checkpoint = tmp_path / "downloaded-model"
    checkpoint.mkdir()
    (checkpoint / "model_config.json").write_text("{}", encoding="utf-8")
    assert resolve_checkpoint(parser.parse_args([]), parser,
                              input_fn=lambda _: str(checkpoint)) == checkpoint.resolve()


def test_checkpoint_must_be_complete(tmp_path: Path):
    parser = build_parser()
    with pytest.raises(SystemExit, match="2"):
        resolve_checkpoint(parser.parse_args(["--checkpoint", str(tmp_path)]), parser)


def test_checkpoint_accepts_modelscope_id():
    parser = build_parser()
    args = parser.parse_args(["--checkpoint", "hh108801/LaKun-0.7B"])
    assert resolve_checkpoint(args, parser) == "hh108801/LaKun-0.7B"


def test_text_noul_does_not_need_criteria():
    parser = build_parser()
    args = parser.parse_args(["--state", "订单已经退款", "--type", "noul",
                              "--question", "订单是否已退款？"])
    questions, state, image = resolve_request(args, parser)
    assert state == "订单已经退款" and image is None
    assert questions == [{"type": "noul", "question": "订单是否已退款？",
                          "criteria": ["false", "true"]}]


def test_image_choice_accepts_one_local_image(tmp_path: Path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"fixture path only")
    parser = build_parser()
    args = parser.parse_args(["--image", str(image), "--type", "choice",
                              "--question", "图中是什么？", "--criteria", "猫", "狗", "汽车"])
    questions, state, selected_image = resolve_request(args, parser)
    assert state == ""
    assert selected_image == image.resolve()
    assert questions[0]["criteria"] == ["猫", "狗", "汽车"]


def test_interactive_single_question():
    parser = build_parser()
    answers = iter(["", "这是一条很紧急的工单", "score", "紧急程度？", "低 | 中 | 高"])
    questions, state, image = resolve_request(parser.parse_args([]), parser,
                                               input_fn=lambda _: next(answers))
    assert image is None and state == "这是一条很紧急的工单"
    assert questions[0]["criteria"] == ["低", "中", "高"]


def test_rejects_duplicate_options_before_model_load():
    parser = build_parser()
    args = parser.parse_args(["--state", "text", "--type", "choice",
                              "--question", "Which?", "--criteria", "A", "A"])
    with pytest.raises(SystemExit, match="2"):
        resolve_request(args, parser)


def test_cli_renders_predicted_option(monkeypatch, capsys, tmp_path: Path):
    (tmp_path / "model_config.json").write_text("{}", encoding="utf-8")
    class FakePredictor:
        def predict(self, questions, *, state, image):
            assert questions[0]["criteria"] == ["false", "true"]
            assert state == "订单已经退款" and image is None
            return {"answers": [{"criteria": ["false", "true"],
                                  "probabilities": [0.2, 0.8], "predicted_index": 1}]}

    from lakun.inference import LaKunPredictor
    monkeypatch.setattr(LaKunPredictor, "from_pretrained", lambda *a, **kw: FakePredictor())
    main(["--checkpoint", str(tmp_path), "--state", "订单已经退款",
          "--type", "noul", "--question", "是否退款？"])
    output = capsys.readouterr().out
    assert "答案：true" in output and "0.8000" in output
