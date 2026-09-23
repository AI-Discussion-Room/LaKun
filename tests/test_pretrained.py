from pathlib import Path

import torch
from transformers import (AutoModel, ModernBertConfig, ModernBertForMaskedLM,
                          SiglipConfig, SiglipImageProcessor, SiglipModel,
                          SiglipTextConfig, SiglipVisionConfig)

from lakun.pretrained import load_siglip_vision, require_siglip_checkpoint


def test_load_only_vision_weights(tmp_path: Path) -> None:
    vision = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                num_attention_heads=4, image_size=32, patch_size=16)
    text = SiglipTextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                            num_attention_heads=4, vocab_size=100)
    original = SiglipModel(SiglipConfig(text_config=text.to_dict(), vision_config=vision.to_dict()))
    original.save_pretrained(tmp_path, safe_serialization=True)
    SiglipImageProcessor(size={"height": 32, "width": 32}).save_pretrained(tmp_path)
    require_siglip_checkpoint(tmp_path)
    loaded = load_siglip_vision(tmp_path)
    expected = {"vision_model." + key: value for key, value in original.vision_model.state_dict().items()}
    assert set(loaded.state_dict()) == set(expected)
    assert all(torch.equal(value, expected[key]) for key, value in loaded.state_dict().items())


def test_text_encoder_can_initialize_from_masked_lm_checkpoint(tmp_path: Path) -> None:
    config = ModernBertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1,
                              num_attention_heads=4, intermediate_size=64,
                              max_position_embeddings=128, pad_token_id=0,
                              bos_token_id=1, eos_token_id=2)
    source = ModernBertForMaskedLM(config)
    source.save_pretrained(tmp_path, safe_serialization=True)
    encoder = AutoModel.from_pretrained(tmp_path, local_files_only=True, attn_implementation="sdpa")
    assert encoder.config.model_type == "modernbert"
    assert torch.equal(encoder.embeddings.tok_embeddings.weight,
                       source.model.embeddings.tok_embeddings.weight)
