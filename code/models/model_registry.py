"""Model registry for small language models compatible with 6GB VRAM."""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ModelInfo:
    """Information about a small language model for evaluation."""
    name: str
    display_name: str
    ollama_id: str
    parameters_b: float  # Billions
    quantization: str
    vram_estimate_gb: float
    context_length: int
    family: str
    release_date: str
    notes: str = ""


# Models that fit in 6GB VRAM with 4-bit quantization
AVAILABLE_MODELS = [
    ModelInfo(
        name="llama3.2-3b",
        display_name="Llama 3.2 3B",
        ollama_id="llama3.2:3b",
        parameters_b=3.0,
        quantization="Q4_K_M",
        vram_estimate_gb=2.5,
        context_length=8192,
        family="Llama",
        release_date="2025-09",
        notes="Smallest capable model, good baseline",
    ),
    ModelInfo(
        name="phi-3.5-mini",
        display_name="Phi-3.5-mini 3.8B",
        ollama_id="phi3.5:3.8b",
        parameters_b=3.8,
        quantization="Q4_K_M",
        vram_estimate_gb=2.8,
        context_length=4096,
        family="Phi",
        release_date="2025-08",
        notes="Microsoft, strong on structured tasks",
    ),
    ModelInfo(
        name="qwen2.5-7b",
        display_name="Qwen 2.5 7B",
        ollama_id="qwen2.5:7b",
        parameters_b=7.0,
        quantization="Q4_K_M",
        vram_estimate_gb=4.5,
        context_length=32768,
        family="Qwen",
        release_date="2025-06",
        notes="Strong open-weight, good tool-use",
    ),
    ModelInfo(
        name="mistral-7b",
        display_name="Mistral 7B v0.3",
        ollama_id="mistral:7b",
        parameters_b=7.0,
        quantization="Q4_K_M",
        vram_estimate_gb=4.5,
        context_length=32768,
        family="Mistral",
        release_date="2025-05",
        notes="Well-known baseline, good benchmark coverage",
    ),
    ModelInfo(
        name="gemma2-9b",
        display_name="Gemma 2 9B",
        ollama_id="gemma2:9b",
        parameters_b=9.0,
        quantization="Q4_K_M",
        vram_estimate_gb=5.5,
        context_length=8192,
        family="Gemma",
        release_date="2025-07",
        notes="Google, largest model that fits, strong performer",
    ),
    ModelInfo(
        name="llama3.2-1b",
        display_name="Llama 3.2 1B",
        ollama_id="llama3.2:1b",
        parameters_b=1.0,
        quantization="Q4_K_M",
        vram_estimate_gb=1.0,
        context_length=8192,
        family="Llama",
        release_date="2025-09",
        notes="Ultra-small, lower bound comparison",
    ),
    ModelInfo(
        name="deepseek-r1-7b",
        display_name="DeepSeek-R1 7B",
        ollama_id="deepseek-r1:7b",
        parameters_b=7.0,
        quantization="Q4_K_M",
        vram_estimate_gb=4.5,
        context_length=16384,
        family="DeepSeek",
        release_date="2026-01",
        notes="Reasoning-distilled, RL from chain-of-thought traces",
    ),
    ModelInfo(
        name="qwen2.5-coder-7b",
        display_name="Qwen 2.5 Coder 7B",
        ollama_id="qwen2.5-coder:7b",
        parameters_b=7.0,
        quantization="Q4_K_M",
        vram_estimate_gb=4.5,
        context_length=32768,
        family="Qwen",
        release_date="2025-12",
        notes="Code-specialized, strong HumanEval, tests tool-use transfer",
    ),
    ModelInfo(
        name="llama3.1-8b",
        display_name="Llama 3.1 8B",
        ollama_id="llama3.1:8b",
        parameters_b=8.0,
        quantization="Q4_K_M",
        vram_estimate_gb=5.5,
        context_length=131072,
        family="Llama",
        release_date="2025-07",
        notes="Larger Llama, within-family size comparison",
    ),
]


def get_recommended_models() -> List[ModelInfo]:
    """Return the recommended set of models for the main evaluation."""
    # Expanded set: 9 models spanning 1B-9B, 7 families, 3 architectures
    recommended_names = [
        "llama3.2-1b",      # Ultra-small baseline
        "llama3.2-3b",      # Small capable Llama
        "phi-3.5-mini",     # Microsoft, structured tasks
        "deepseek-r1-7b",   # Reasoning-distilled (RL from CoT)
        "qwen2.5-coder-7b", # Code-specialized Qwen
        "qwen2.5-7b",       # Strong open-weight general
        "mistral-7b",       # Well-known baseline
        "llama3.1-8b",      # Larger Llama, within-family comparison
        "gemma2-9b",        # Largest we can fit
    ]
    return [m for m in AVAILABLE_MODELS if m.name in recommended_names]


def get_all_compatible_models() -> List[ModelInfo]:
    """Return all models compatible with 6GB VRAM."""
    return [m for m in AVAILABLE_MODELS if m.vram_estimate_gb <= 6.0]


class ModelRegistry:
    """Registry for managing available models."""

    def __init__(self):
        self._models = {m.name: m for m in AVAILABLE_MODELS}

    def get_model(self, name: str) -> Optional[ModelInfo]:
        return self._models.get(name)

    def list_models(self) -> List[ModelInfo]:
        return list(self._models.values())

    def check_ollama_installed(self, model_name: str) -> bool:
        """Check if a model is installed in Ollama."""
        import subprocess
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return model_name in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def pull_model(self, model_name: str) -> bool:
        """Pull a model using Ollama."""
        import subprocess
        try:
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
