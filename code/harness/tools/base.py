"""Base tool classes for the agent harness."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import time


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: str
    data: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """Abstract base class for all tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._failure_mode: Optional[str] = None  # For fault injection experiments

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Return JSON schema for the tool's parameters."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters_schema(),
        }

    @abstractmethod
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Return JSON schema of parameters."""
        pass

    def set_failure_mode(self, mode: Optional[str] = None):
        """
        Set a failure mode for fault injection experiments.
        Modes: None (normal), "timeout", "rate_limit", "error", "schema_drift"
        """
        self._failure_mode = mode

    def _maybe_inject_fault(self) -> Optional[ToolResult]:
        """Inject a fault if failure mode is set. Returns result if faulted, None otherwise."""
        if self._failure_mode is None:
            return None

        if self._failure_mode == "timeout":
            time.sleep(2.0)  # Simulate timeout
            return ToolResult(
                success=False,
                output="",
                error="timeout: tool execution exceeded maximum allowed time",
            )
        elif self._failure_mode == "rate_limit":
            return ToolResult(
                success=False,
                output="",
                error="rate_limit: too many requests. Please retry after 30 seconds.",
            )
        elif self._failure_mode == "error":
            return ToolResult(
                success=False,
                output="",
                error="internal_error: unexpected error during tool execution",
            )
        elif self._failure_mode == "schema_drift":
            # Return output in wrong format
            return ToolResult(
                success=True,
                output=json.dumps({"wrong_key": "value", "format": "changed"}),
                data=None,
                metadata={"schema_drift": True},
            )
        return None


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_schemas(self) -> List[Dict[str, Any]]:
        return [t.get_schema() for t in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools
