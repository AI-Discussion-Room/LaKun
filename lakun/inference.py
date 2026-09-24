"""Load a local or Hub LaKun checkpoint and score caller-supplied decisions."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import (AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer,
                          SiglipVisionConfig, SiglipVisionModel)

from .encoding import prepare_group
from .model import LaKunModel
from .pretrained import initialize_base_components, resolve_lakun_checkpoint
from .vendor.laya.common import DecisionModel


def load_checkpoint_into(model: LaKunModel, root: Path) -> None:
    """Copy a saved checkpoint into an existing model without allocating a second model."""
    index_path = root / "lakun.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = ["lakun.safetensors"]
    expected = set(model.state_dict())
    loaded: set[str] = set()
    for shard in shards:
        state = load_file(str(root / shard), device="cpu")
        overlap = loaded.intersection(state)
        if overlap:
            raise ValueError(f"Duplicate checkpoint tensors: {sorted(overlap)[:3]}")
        loaded.update(state)
        model.load_state_dict(state, strict=False)
        del state
    if loaded != expected:
        raise ValueError(f"Checkpoint tensor mismatch: missing={len(expected - loaded)}, extra={len(loaded - expected)}")


class LaKunPredictor:
    def __init__(self, model, tokenizer, processor, device: torch.device,
                 model_status: str = "trained_checkpoint"):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device
        self.model_status = model_status

    @classmethod
    def from_base_models(cls, text_model: str | Path, vision_model: str | Path,
                         device: str | None = None) -> "LaKunPredictor":
        """Check raw-weight inference wiring; the new head is random and scores are not useful."""
        tokenizer, processor, model = initialize_base_components(text_model, vision_model)
        chosen_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        return cls(model, tokenizer, processor, chosen_device, "random_head_untrained")

    @classmethod
    def from_pretrained(cls, checkpoint: str | Path, device: str | None = None) -> "LaKunPredictor":
        root = resolve_lakun_checkpoint(str(checkpoint))
        config = json.loads((root / "model_config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "lakun":
            raise ValueError(f"Not a LaKun checkpoint: {root}")
        with torch.device("meta"):
            text_encoder = AutoModel.from_config(AutoConfig.from_pretrained(root / "text_encoder"))
            decision = DecisionModel(text_encoder, head_layers=config["decision_head_layers"],
                                     n_act=config["decision_n_act"])
            vision = SiglipVisionModel(SiglipVisionConfig.from_pretrained(root / "vision_encoder"))
            model = LaKunModel(decision, vision, config["visual_tokens"])
        index_path = root / "lakun.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shards = sorted(set(index["weight_map"].values()))
        else:
            shards = ["lakun.safetensors"]
        expected = set(model.state_dict())
        loaded = set()
        for shard in shards:
            state = load_file(str(root / shard), device="cpu")
            overlap = loaded.intersection(state)
            if overlap:
                raise ValueError(f"Duplicate checkpoint tensors: {sorted(overlap)[:3]}")
            loaded.update(state)
            model.load_state_dict(state, strict=False, assign=True)
            del state
        if loaded != expected:
            raise ValueError(f"Checkpoint tensor mismatch: missing={len(expected - loaded)}, extra={len(loaded - expected)}")
        embeddings = model.vision.vision_model.embeddings
        embeddings.position_ids = torch.arange(embeddings.num_positions).expand(1, -1)
        text_embeddings = getattr(model.decision.encoder, "embeddings", None)
        if text_embeddings is not None:
            position_ids = getattr(text_embeddings, "position_ids", None)
            if position_ids is not None and position_ids.is_meta:
                text_embeddings.position_ids = torch.arange(position_ids.shape[-1]).expand(position_ids.shape)
            token_type_ids = getattr(text_embeddings, "token_type_ids", None)
            if token_type_ids is not None and token_type_ids.is_meta:
                text_embeddings.token_type_ids = torch.zeros(token_type_ids.shape, dtype=token_type_ids.dtype)
        for module in model.modules():
            if hasattr(module, "rope_init_fn") and hasattr(module, "inv_freq") and module.inv_freq.is_meta:
                inv_freq, module.attention_scaling = module.rope_init_fn(module.config, device="cpu")
                module.inv_freq = inv_freq
                module.original_inv_freq = inv_freq
        remaining_meta = [(name, tuple(value.shape)) for name, value in model.named_buffers() if value.is_meta]
        if remaining_meta:
            raise RuntimeError(f"Unmaterialized checkpoint buffers: {remaining_meta}")
        tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
        processor = AutoImageProcessor.from_pretrained(root / "image_processor", use_fast=False,
                                                      local_files_only=True)
        chosen_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        fixture = config.get("base_models", {}).get("text") == "tiny_random_BERT_fixture"
        return cls(model, tokenizer, processor, chosen_device,
                   "tiny_random_smoke" if fixture else "trained_checkpoint")

    def predict_group(self, group: dict) -> dict:
        batch, pixels = prepare_group(group, self.tokenizer, self.processor, self.device,
                                      getattr(self.model, "visual_tokens", 64))
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(batch, pixels), dim=-1)
        answers = []
        for i, row in enumerate(group["rows"]):
            values = probabilities[i, :len(row["criteria"])].tolist()
            answer = {"type": row["type"], "question": row["question"],
                      "criteria": row["criteria"], "probabilities": values,
                      "predicted_index": max(range(len(values)), key=values.__getitem__)}
            if group.get("from_dataset"):
                answer["label_index"] = row["selected_index"]
            answers.append(answer)
        return {"group_id": group["group_id"], "modality": group["modality"],
                "model_status": self.model_status, "answers": answers}

    def predict(self, questions: list[dict], *, state: str = "", image: str | Path | None = None) -> dict:
        """Each question has `type`, `question`, and ordered `criteria` strings."""
        if not isinstance(state, str) or not 1 <= len(questions) <= 20:
            raise ValueError("state must be text and questions must contain 1..20 items")
        image_path = Path(image).expanduser().resolve() if image is not None else None
        if image_path is not None and not image_path.is_file():
            raise FileNotFoundError(image_path)
        rows = []
        for i, item in enumerate(questions):
            kind, criteria = item["type"], item["criteria"]
            if not isinstance(criteria, list) or not all(isinstance(x, str) and x for x in criteria):
                raise ValueError(f"invalid criteria for question {i}")
            if (kind == "choice" and not 2 <= len(criteria) <= 20
                    or kind == "score" and not 3 <= len(criteria) <= 20
                    or kind == "noul" and criteria != ["false", "true"]
                    or kind not in {"choice", "score", "noul"}):
                raise ValueError(f"invalid type or option count for question {i}")
            rows.append({"id": f"inference:q{i}", "type": kind, "question": item["question"],
                         "criteria": criteria, "selected_index": 0,
                         "target": [1.0] + [0.0] * (len(criteria) - 1)})
        group = {"group_id": "inference", "state": state,
                 "modality": "image" if image_path else "text", "image_path": image_path,
                 "rows": rows}
        return self.predict_group(group)
