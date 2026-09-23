"""LaKun: early visual-token fusion with a typed-decision encoder."""
from __future__ import annotations

import torch
from torch import nn


class LaKunModel(nn.Module):
    def __init__(self, decision_model: nn.Module, vision_model: nn.Module, visual_tokens: int = 64):
        super().__init__()
        self.decision = decision_model
        self.vision = vision_model
        self.visual_tokens = visual_tokens
        vision_width = vision_model.config.hidden_size
        text_width = decision_model.encoder.config.hidden_size
        self.visual_queries = nn.Parameter(torch.randn(visual_tokens, vision_width) * 0.02)
        self.visual_attention = nn.MultiheadAttention(vision_width, 8, batch_first=True)
        self.visual_projection = nn.Sequential(nn.LayerNorm(vision_width), nn.Linear(vision_width, text_width))
        for parameter in self.decision.act_head.parameters():
            parameter.requires_grad_(False)  # Laya's action head has no label in this dataset.

    def forward(self, batch: dict, pixel_values: torch.Tensor | None = None) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        hidden = self.decision.encoder.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            if pixel_values.shape[0] != 1:
                raise ValueError("one group must contain exactly one image")
            patches = self.vision(pixel_values=pixel_values).last_hidden_state
            queries = self.visual_queries.unsqueeze(0).expand(patches.shape[0], -1, -1)
            visual, _ = self.visual_attention(queries, patches, patches, need_weights=False)
            visual = self.visual_projection(visual)[0].to(hidden.dtype)
            lengths = attention_mask.sum(-1).tolist()
            sequences = [torch.cat([hidden[i, :length - 1], visual, hidden[i, length - 1:length]], dim=0)
                         for i, length in enumerate(lengths)]
            hidden = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
            attention_mask = torch.zeros(hidden.shape[:2], device=hidden.device, dtype=attention_mask.dtype)
            for i, sequence in enumerate(sequences):
                attention_mask[i, :sequence.shape[0]] = 1
        hidden = self.decision.encoder(inputs_embeds=hidden, attention_mask=attention_mask).last_hidden_state
        hidden = hidden + self.decision.type_emb(batch["qtype"])[:, None, :]
        if self.decision.head is not None:
            padding = ~attention_mask.bool()
            for layer in self.decision.head.layers:
                hidden = layer(hidden, src_key_padding_mask=padding)
        marker_pos = batch["marker_pos"]
        index = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, hidden.shape[-1])
        markers = torch.gather(hidden, 1, index)
        logits = self.decision.scorer(markers).squeeze(-1).float()
        return logits.masked_fill(~batch["marker_mask"], -1e4)
