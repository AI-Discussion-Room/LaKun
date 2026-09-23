"""One strict 512-token decision encoding shared by training and inference."""
from __future__ import annotations

import torch
from PIL import Image

from .vendor.laya.common import QTYPES, render_options


MAX_CONTEXT = 512
MAX_HEAD = 192
VISUAL_TOKENS = 64


def encode_question(tokenizer, state: str, row: dict, image: bool) -> tuple[list[int], list[int]]:
    kind = row["type"]
    criteria = row["criteria"]
    q = {"t": kind, "ins": row["question"],
         "crit": dict.fromkeys(criteria) if kind == "choice" else criteria if kind == "score" else {}}
    options = render_options(q)
    if len(options) != len(criteria):
        raise ValueError(f"option count changed for {row['id']}")
    mask_token = tokenizer.mask_token
    head = tokenizer(f"{kind} question: {row['question'].replace(mask_token, ' ')}",
                     add_special_tokens=False)["input_ids"]
    option_ids = [tokenizer(" " + option.replace(mask_token, " "), add_special_tokens=False)["input_ids"]
                  for option in options]
    if any(len(ids) > 48 for ids in option_ids):
        raise ValueError(f"option exceeds 48 tokens: {row['id']}")
    if len(head) + sum(1 + len(ids) for ids in option_ids) > MAX_HEAD:
        raise ValueError(f"head exceeds 192 tokens: {row['id']}")
    ids = [tokenizer.cls_token_id] + head + [tokenizer.sep_token_id]
    markers = []
    for option in option_ids:
        markers.append(len(ids))
        ids.extend([tokenizer.mask_token_id] + option)
    ids.append(tokenizer.sep_token_id)
    ids.extend(tokenizer(state.replace(mask_token, " "), add_special_tokens=False)["input_ids"])
    ids.append(tokenizer.sep_token_id)
    if len(ids) + (VISUAL_TOKENS if image else 0) > MAX_CONTEXT:
        raise ValueError(f"sequence exceeds 512 tokens without truncation: {row['id']}")
    return ids, markers


def prepare_group(group: dict, tokenizer, processor, device: torch.device) -> tuple[dict, torch.Tensor | None]:
    rows = group["rows"]
    encoded = [encode_question(tokenizer, group["state"], row, group["modality"] == "image")
               for row in rows]
    count = len(rows)
    length = max(len(ids) for ids, _ in encoded)
    choices = max(len(row["criteria"]) for row in rows)
    if length + (VISUAL_TOKENS if group["modality"] == "image" else 0) > MAX_CONTEXT:
        raise ValueError(f"padded group exceeds 512 tokens: {group['group_id']}")
    input_ids = torch.full((count, length), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((count, length), dtype=torch.long)
    marker_pos = torch.zeros((count, choices), dtype=torch.long)
    marker_mask = torch.zeros((count, choices), dtype=torch.bool)
    target = torch.zeros((count, choices), dtype=torch.float32)
    for i, ((ids, markers), row) in enumerate(zip(encoded, rows)):
        n, k = len(ids), len(markers)
        input_ids[i, :n] = torch.tensor(ids)
        attention_mask[i, :n] = 1
        marker_pos[i, :k] = torch.tensor(markers)
        marker_mask[i, :k] = True
        target[i, :k] = torch.tensor(row["target"], dtype=torch.float32)
    batch = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device),
             "marker_pos": marker_pos.to(device), "marker_mask": marker_mask.to(device),
             "target": target.to(device),
             "qtype": torch.tensor([QTYPES[row["type"]] for row in rows], device=device),
             "selected": torch.tensor([row["selected_index"] for row in rows], device=device)}
    pixels = None
    if group["modality"] == "image":
        with Image.open(group["image_path"]) as image:
            pixels = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].to(device)
    return batch, pixels
