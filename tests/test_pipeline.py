"""Exercise the real base-loading route with tiny local weights of the same architectures."""
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (ModernBertConfig, ModernBertForMaskedLM,
                          PreTrainedTokenizerFast, SiglipConfig, SiglipImageProcessor,
                          SiglipModel, SiglipTextConfig, SiglipVisionConfig)

from lakun.encoding import prepare_group
from lakun.inference import LaKunPredictor
from lakun.pretrained import initialize_base_components
from lakun.train import decision_loss, save_checkpoint


def test_base_weights_train_save_and_predict(tmp_path: Path, monkeypatch) -> None:
    text_dir, vision_dir = tmp_path / "mmbert", tmp_path / "siglip"
    text_dir.mkdir()
    vision_dir.mkdir()
    vocabulary = {word: index for index, word in enumerate(
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "What", "color", "The",
         "sky", "is", "blue", "red", "?", "."])}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocabulary["[CLS]"]), ("[SEP]", vocabulary["[SEP]"])])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, cls_token="[CLS]", sep_token="[SEP]",
        mask_token="[MASK]", pad_token="[PAD]", unk_token="[UNK]")
    tokenizer.save_pretrained(text_dir)
    text_config = ModernBertConfig(vocab_size=len(tokenizer), hidden_size=32,
                                   num_hidden_layers=1, num_attention_heads=4,
                                   intermediate_size=64, max_position_embeddings=512,
                                   pad_token_id=tokenizer.pad_token_id)
    ModernBertForMaskedLM(text_config).save_pretrained(text_dir, safe_serialization=True)
    vision_config = SiglipVisionConfig(hidden_size=32, intermediate_size=64,
                                       num_hidden_layers=1, num_attention_heads=4,
                                       image_size=32, patch_size=16)
    vision_text = SiglipTextConfig(hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=1, num_attention_heads=4, vocab_size=100)
    SiglipModel(SiglipConfig(text_config=vision_text.to_dict(),
                             vision_config=vision_config.to_dict())).save_pretrained(
                                 vision_dir, safe_serialization=True)
    SiglipImageProcessor(size={"height": 32, "width": 32}).save_pretrained(vision_dir)
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), (80, 120, 160)).save(image_path)

    loaded_tokenizer, processor, model = initialize_base_components(text_dir, vision_dir)
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
    assert not [name for name, parameter in model.named_parameters() if parameter.is_meta]
    assert not [name for name, buffer in model.named_buffers() if buffer.is_meta]
    row = {"id": "toy:0", "type": "choice", "question": "What color?",
           "criteria": ["blue", "red"], "selected_index": 0, "target": [1.0, 0.0]}
    image_group = {"group_id": "toy-image", "modality": "image", "state": "A square.",
                   "image_path": image_path, "rows": [row]}
    text_group = {"group_id": "toy-text", "modality": "text", "state": "The sky is blue.",
                  "image_path": None, "rows": [row]}
    for group in (image_group, text_group):
        batch, pixels = prepare_group(group, loaded_tokenizer, processor, torch.device("cpu"))
        loss = decision_loss(model(batch, pixels), batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert model.decision.encoder.get_input_embeddings().weight.grad is not None
        if pixels is not None:
            assert any(parameter.grad is not None for parameter in model.vision.parameters())
        model.zero_grad(set_to_none=True)

    checkpoint = tmp_path / "lakun-checkpoint"
    save_checkpoint(model, loaded_tokenizer, processor, SimpleNamespace(smoke=False), checkpoint)
    predictor = LaKunPredictor.from_pretrained(checkpoint, device="cpu")
    assert predictor.predict([{"type": "choice", "question": "What color?",
                               "criteria": ["blue", "red"]}], image=image_path)["model_status"] == "trained_checkpoint"
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo_id: str(checkpoint))
    hub_predictor = LaKunPredictor.from_pretrained("example/LaKun", device="cpu")
    assert hub_predictor.model_status == "trained_checkpoint"
    raw = LaKunPredictor.from_base_models(text_dir, vision_dir, device="cpu")
    assert raw.predict([{"type": "choice", "question": "What color?",
                         "criteria": ["blue", "red"]}], state="The sky is blue.")["model_status"] == "random_head_untrained"
