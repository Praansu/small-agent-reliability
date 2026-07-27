"""ReAct agent implementation for tool-use evaluation."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import time
import uuid

from .tools.base import Tool, ToolRegistry, ToolResult
from .utils import parse_tool_call


@dataclass
class AgentConfig:
    """Configuration for the ReAct agent."""
    max_steps: int = 15
    max_tool_retries: int = 2
    verbose: bool = False
    seed: int = 42


@dataclass
class AgentStep:
    """Record of a single agent step."""
    step_number: int
    thought: str
    tool_call: Optional[Tuple[str, Dict[str, Any]]] = None
    tool_result: Optional[ToolResult] = None
    response: str = ""
    duration_ms: float = 0.0


@dataclass
class AgentResult:
    """Complete result of an agent run."""
    task_id: str
    task_description: str
    success: bool
    final_answer: str
    steps: List[AgentStep]
    total_duration_ms: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_correct(self) -> bool:
        return self.success


class ReActAgent:
    """
    ReAct (Reasoning + Acting) agent that uses tools to complete tasks.
    Follows the pattern: Thought → Action → Observation → Thought → ...
    """

    def __init__(
        self,
        model_call_fn: Callable,
        tool_registry: ToolRegistry,
        config: AgentConfig = None,
    ):
        self.model = model_call_fn
        self.tools = tool_registry
        self.config = config or AgentConfig()

    def _build_system_prompt(self) -> str:
        """Build the system prompt with tool descriptions."""
        tools_desc = []
        for tool_name in self.tools.list_tools():
            tool = self.tools.get(tool_name)
            schema = tool.get_schema()
            params = schema.get("parameters", {}).get("properties", {})
            param_str = ", ".join(f"{p}: {info.get('description', p)}" for p, info in params.items())
            tools_desc.append(f"- {schema['name']}: {schema['description']} Parameters: {param_str}")

        example = """Example:
User: What is the population of Paris?
THOUGHT: I need to search for the population of Paris.
ACTION: {"name": "web_search", "arguments": {"query": "population of Paris"}}
OBSERVATION: [{"title": "population of paris", "snippet": "The population of Paris is approximately 2.1 million..."}]
THOUGHT: I found the answer.
FINAL_ANSWER: The population of Paris is approximately 2.1 million people."""

        return f"""You are an autonomous AI agent that uses tools to accomplish tasks.

You have the following tools available:
{chr(10).join(tools_desc)}

CRITICAL: You MUST use the exact JSON format for tool calls.

{example}

Your format must be:
THOUGHT: Your reasoning about what to do next
ACTION: {{"name": "tool_name", "arguments": {{"param1": "value1"}}}}
OBSERVATION: (you will receive this after each ACTION)
... (repeat as needed)
FINAL_ANSWER: Your final answer to the user

Rules:
1. Always start with THOUGHT
2. ACTION must be valid JSON with "name" and "arguments" keys
3. Only use the tools listed above with their exact names
4. After receiving OBSERVATION, either take another ACTION or give FINAL_ANSWER
5. If a tool fails, try a different approach
6. Never make up data - always use tools first"""

    def run(self, task_description: str, task_id: str = None) -> AgentResult:
        """Execute a task using the ReAct loop."""
        if task_id is None:
            task_id = str(uuid.uuid4())[:8]

        system_prompt = self._build_system_prompt()
        # For small models, embed instructions in the first user message
        messages = [{"role": "user", "content": f"{system_prompt}\n\nYour task: {task_description}"}]
        
        steps = []
        start_time = time.time()
        final_answer = ""
        success = False
        error = None

        for step_num in range(1, self.config.max_steps + 1):
            step_start = time.time()

            # Get model response
            response_text, metadata = self.model(messages)

            step = AgentStep(
                step_number=step_num,
                thought="",
                duration_ms=(time.time() - step_start) * 1000,
            )

            if "error" in metadata:
                step.response = f"Model error: {metadata['error']}"
                error = metadata["error"]
                break

            step.response = response_text

            # Check for final answer
            if "FINAL_ANSWER:" in response_text:
                final_answer = response_text.split("FINAL_ANSWER:")[-1].strip()
                success = True
                steps.append(step)
                break

            # Try to parse tool call
            tool_call = parse_tool_call(response_text)
            
            if tool_call:
                tool_name, arguments = tool_call
                step.tool_call = tool_call

                if tool_name not in self.tools:
                    step.tool_result = ToolResult(
                        success=False,
                        output="",
                        error=f"Unknown tool: '{tool_name}'. Available: {self.tools.list_tools()}",
                    )
                else:
                    tool = self.tools.get(tool_name)
                    # Execute with retries
                    for attempt in range(self.config.max_tool_retries + 1):
                        tool_result = tool.execute(**arguments)
                        if tool_result.success or attempt == self.config.max_tool_retries:
                            break
                    
                    step.tool_result = tool_result

                # Add to conversation
                obs = f"OBSERVATION: {step.tool_result.output if step.tool_result.success else f'ERROR: {step.tool_result.error}'}"
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "system", "content": obs})
            else:
                # No tool call found, just add response to conversation
                messages.append({"role": "assistant", "content": response_text})

            steps.append(step)

            # Safety: check for obvious loops (same response 3+ times)
            if len(steps) >= 3:
                last_responses = [s.response for s in steps[-3:]]
                if len(set(last_responses)) == 1:
                    error = "Loop detected: identical responses 3 consecutive times"
                    break

        total_duration = (time.time() - start_time) * 1000

        return AgentResult(
            task_id=task_id,
            task_description=task_description,
            success=success,
            final_answer=final_answer,
            steps=steps,
            total_duration_ms=total_duration,
            error=error,
            metadata={"step_count": len(steps), "model_info": metadata},
        )
