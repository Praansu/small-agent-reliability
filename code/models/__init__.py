"""Model registry for small language model evaluation."""

from .model_registry import (
    ModelRegistry,
    get_recommended_models,
    AVAILABLE_MODELS,
)

__all__ = [
    "ModelRegistry",
    "get_recommended_models",
    "AVAILABLE_MODELS",
]
