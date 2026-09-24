"""LaKun multimodal typed decisions: mmBERT-base + SigLIP + Laya-style head."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .inference import LaKunPredictor

__all__ = ["LaKunPredictor"]


def __getattr__(name: str):
    if name == "LaKunPredictor":
        from .inference import LaKunPredictor
        return LaKunPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
