"""Code execution and file operation tools."""

from .base import Tool, ToolResult
from typing import Any, Dict, Optional
import json
import subprocess
import tempfile
import os


class CodeInterpreterTool(Tool):
    """Execute Python code in a sandboxed environment."""

    def __init__(self):
        super().__init__(
            name="code_interpreter",
            description="Execute Python code and return the output. Use for calculations, data processing, and scripting.",
        )
        self._namespace = {}

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds",
                    "default": 10,
                },
            },
            "required": ["code"],
        }

    def execute(self, code: str, timeout: int = 10) -> ToolResult:
        fault = self._maybe_inject_fault()
        if fault:
            return fault

        # Safety: block dangerous operations
        dangerous = ["__import__", "exec(", "eval(", "os.system", "subprocess", "open("]
        for item in dangerous:
            if item in code:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Security: use of '{item}' is not allowed in code_interpreter",
                )

        # Safe execution using a restricted environment
        try:
            # Capture stdout
            import io
            import sys
            from contextlib import redirect_stdout

            f = io.StringIO()
            restricted_globals = {
                "__builtins__": {
                    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
                    "enumerate": enumerate, "float": float, "int": int, "len": len,
                    "list": list, "max": max, "min": min, "print": print,
                    "range": range, "round": round, "sorted": sorted, "str": str,
                    "sum": sum, "tuple": tuple, "type": type, "zip": zip,
                    "True": True, "False": False, "None": None,
                    "map": map, "filter": filter, "reversed": reversed, "set": set,
                    "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr,
                }
            }

            with redirect_stdout(f):
                exec(code, restricted_globals)

            output = f.getvalue()
            return ToolResult(
                success=True,
                output=output if output else "Code executed successfully (no output)",
                data={"output": output},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}",
            )


class FileOperator(Tool):
    """Simple file read/write operations in a sandboxed workspace."""

    def __init__(self):
        super().__init__(
            name="file_operator",
            description="Read, write, and list files in the workspace directory.",
        )
        self._workspace = tempfile.mkdtemp(prefix="agent_workspace_")

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write", "list", "delete"],
                    "description": "File operation to perform",
                },
                "filename": {
                    "type": "string",
                    "description": "Name of the file to operate on",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (for write action)",
                },
            },
            "required": ["action"],
        }

    def execute(
        self,
        action: str,
        filename: Optional[str] = None,
        content: Optional[str] = None,
    ) -> ToolResult:
        fault = self._maybe_inject_fault()
        if fault:
            return fault

        # Safety: prevent directory traversal
        if filename and ".." in filename:
            return ToolResult(success=False, output="", error="Security: path traversal not allowed")

        if action == "list":
            files = os.listdir(self._workspace)
            return ToolResult(
                success=True,
                output=json.dumps({"files": files}, indent=2),
                data={"files": files},
            )

        elif action == "read":
            if not filename:
                return ToolResult(success=False, output="", error="filename required")
            filepath = os.path.join(self._workspace, filename)
            if not os.path.exists(filepath):
                return ToolResult(success=False, output="", error=f"File '{filename}' not found")
            with open(filepath, "r") as f:
                content = f.read()
            return ToolResult(success=True, output=content, data={"content": content})

        elif action == "write":
            if not filename or content is None:
                return ToolResult(success=False, output="", error="filename and content required")
            filepath = os.path.join(self._workspace, filename)
            with open(filepath, "w") as f:
                f.write(content)
            return ToolResult(success=True, output=f"File '{filename}' written successfully")

        elif action == "delete":
            if not filename:
                return ToolResult(success=False, output="", error="filename required")
            filepath = os.path.join(self._workspace, filename)
            if not os.path.exists(filepath):
                return ToolResult(success=False, output="", error=f"File '{filename}' not found")
            os.remove(filepath)
            return ToolResult(success=True, output=f"File '{filename}' deleted")

        return ToolResult(success=False, output="", error=f"Unknown action: {action}")
