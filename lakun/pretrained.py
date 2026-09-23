"""Strict local base-weight loading and optional published LaKun checkpoint resolution."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import (AutoImageProcessor, AutoModel, AutoTokenizer,
                          SiglipVisionConfig, SiglipVisionModel)


def require_text_tokenizer(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    required = (root / "config.json", root / "tokenizer.json", root / "tokenizer_config.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("mmBERT-base tokenizer is incomplete: " + ", ".join(missing))
    return root


def require_text_backbone(root: str | Path) -> Path:
    root = require_text_tokenizer(root)
    missing = []
    if not (root / "model.safetensors").is_file() and not (root / "pytorch_model.bin").is_file():
        missing.append(str(root / "model.safetensors or pytorch_model.bin"))
    if missing:
        raise FileNotFoundError("mmBERT-base backbone is incomplete: " + ", ".join(missing))
    return root


def require_siglip_checkpoint(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    required = (root / "config.json", root / "model.safetensors", root / "preprocessor_config.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("SigLIP checkpoint is incomplete: " + ", ".join(missing))
    return root


def load_siglip_vision(root: str | Path) -> SiglipVisionModel:
    """Load only the vision tower from Google's full SigLIP safetensors file."""
    root = require_siglip_checkpoint(root)
    raw_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if raw_config.get("model_type") != "siglip" or "vision_config" not in raw_config:
        raise ValueError(f"Expected a full SigLIP config with vision_config: {root / 'config.json'}")
    with torch.device("meta"):
        model = SiglipVisionModel(SiglipVisionConfig.from_dict(raw_config["vision_config"]))
    expected = set(model.state_dict())
    with safe_open(str(root / "model.safetensors"), framework="pt", device="cpu") as weights:
        names = set(weights.keys())
        missing = expected - names
        if missing:
            raise ValueError(f"SigLIP vision weights missing {len(missing)} keys; first: {sorted(missing)[:3]}")
        state = {name: weights.get_tensor(name) for name in expected}
    model.load_state_dict(state, strict=True, assign=True)
    # This non-persistent buffer is not in safetensors, so meta construction
    # must materialize it explicitly before the first forward/backward pass.
    embeddings = model.vision_model.embeddings
    embeddings.position_ids = torch.arange(embeddings.num_positions).expand(1, -1)
    return model


def initialize_base_components(text_root: str | Path, vision_root: str | Path):
    """Load both original towers; initialize LaKun fusion and decision head afresh."""
    from .encoding import VISUAL_TOKENS
    from .model import LaKunModel
    from .vendor.laya.common import DecisionModel

    text_root = require_text_backbone(text_root)
    vision_root = require_siglip_checkpoint(vision_root)
    tokenizer = AutoTokenizer.from_pretrained(text_root, local_files_only=True)
    encoder = AutoModel.from_pretrained(text_root, local_files_only=True,
                                        low_cpu_mem_usage=True, attn_implementation="sdpa")
    encoder.config.reference_compile = False
    decision = DecisionModel(encoder, head_layers=2, n_act=2)
    vision = load_siglip_vision(vision_root)
    processor = AutoImageProcessor.from_pretrained(vision_root, use_fast=False,
                                                    local_files_only=True)
    return tokenizer, processor, LaKunModel(decision, vision, VISUAL_TOKENS)


def resolve_lakun_checkpoint(path_or_repo: str) -> Path:
    """Load a local training output or a future Hugging Face LaKun model repository."""
    path = Path(path_or_repo).expanduser()
    if path.is_dir():
        return path.resolve()
    if path.is_absolute() or path_or_repo.startswith((".", "~")):
        raise FileNotFoundError(path)
    if "/" not in path_or_repo:
        raise FileNotFoundError(f"Checkpoint directory not found: {path_or_repo}")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=path_or_repo))
