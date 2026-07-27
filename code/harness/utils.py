"""Utility functions for the agent harness."""

import json
import re
import random
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a language model."""
    name: str
    display_name: str
    provider: str  # "ollama", "llama_cpp", "huggingface"
    model_id: str
    quantization: str = "Q4_K_M"
    max_tokens: int = 4096
    temperature: float = 0.0  # Use 0 for reproducibility
    context_length: int = 8192
    parameters_b: float = 0.0  # Parameter count in billions


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Parse a tool call from model output.
    Supports both JSON and XML-style tool call formats.
    Returns (tool_name, arguments) or None if no tool call found.
    """
    # Try JSON format: {"name": "tool_name", "arguments": {...}}
    json_match = re.search(
        r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\})\s*\}',
        text,
        re.DOTALL,
    )
    if json_match:
        tool_name = json_match.group(1)
        try:
            arguments = json.loads(json_match.group(2))
            return tool_name, arguments
        except json.JSONDecodeError:
            pass

    # Try function_call format: {"function": "name", "params": {...}}
    func_match = re.search(
        r'\{\s*"function"\s*:\s*"([^"]+)"\s*,\s*"params"\s*:\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\})\s*\}',
        text,
        re.DOTALL,
    )
    if func_match:
        tool_name = func_match.group(1)
        try:
            arguments = json.loads(func_match.group(2))
            return tool_name, arguments
        except json.JSONDecodeError:
            pass

    # Try XML format: <tool_call><tool_name>name</tool_name><arguments>...</arguments></tool_call>
    xml_match = re.search(
        r'<tool_call>\s*<tool_name>([^<]+)</tool_name>\s*<arguments>(.*?)</arguments>\s*</tool_call>',
        text,
        re.DOTALL,
    )
    if xml_match:
        tool_name = xml_match.group(1).strip()
        try:
            arguments = json.loads(xml_match.group(2).strip())
            return tool_name, arguments
        except json.JSONDecodeError:
            pass

    # Try simpler: tool_name(arg1=val1, arg2=val2)
    simple_match = re.search(
        r'(\w+)\s*\(\s*(.*?)\s*\)\s*$',
        text.strip(),
        re.DOTALL,
    )
    if simple_match:
        tool_name = simple_match.group(1)
        args_str = simple_match.group(2)
        arguments = {}
        for param_match in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', args_str):
            arguments[param_match.group(1)] = param_match.group(2)
        if arguments:
            return tool_name, arguments

    return None


def format_messages(
    system_prompt: str,
    conversation_history: List[Dict[str, str]],
    tool_schemas: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    Format messages for the model with system prompt and tool schemas.
    Returns a list of message dictionaries.
    """
    messages = [
        {"role": "system", "content": system_prompt + "\n\nAvailable tools:\n" + json.dumps(tool_schemas, indent=2)}
    ]
    messages.extend(conversation_history)
    return messages


def call_llm_ollama(
    model_name: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Call an LLM via Ollama API.
    Returns (response_text, metadata).
    """
    import requests

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        }
    }

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        return result["message"]["content"], {
            "model": model_name,
            "tokens": result.get("eval_count", 0),
            "duration_ms": result.get("eval_duration", 0) / 1_000_000 if result.get("eval_duration") else 0,
        }
    except requests.exceptions.ConnectionError:
        return "", {"error": "Ollama not running. Please start Ollama first."}
    except Exception as e:
        return "", {"error": str(e)}


def load_model(config: ModelConfig):
    """
    Load a model based on configuration.
    Returns a function that takes messages and returns text.
    """
    if config.provider == "ollama":
        def call_fn(messages, max_tokens=None, temperature=None):
            return call_llm_ollama(
                config.model_id,
                messages,
                max_tokens=max_tokens or config.max_tokens,
                temperature=temperature if temperature is not None else config.temperature,
            )
        return call_fn
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")
