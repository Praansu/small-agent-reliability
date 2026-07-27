"""Search and knowledge base tools for the agent environment."""

from .base import Tool, ToolResult
from typing import Any, Dict, Optional
import json
import random


class SearchTool(Tool):
    """Simulated web search tool."""

    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for information. Returns relevant results for a given query.",
        )
        # Simulated knowledge base for deterministic evaluation
        self._knowledge_base = {
            "population of paris": "The population of Paris is approximately 2.1 million within the city limits.",
            "capital of france": "The capital of France is Paris.",
            "python list append": "The append() method in Python adds an element to the end of a list.",
            "einstein relativity": "Albert Einstein published the theory of special relativity in 1905.",
            "boiling point of water": "Water boils at 100°C (212°F) at sea level.",
            "height of eiffel tower": "The Eiffel Tower is 330 meters (1,083 feet) tall including antennas.",
            "sql select syntax": "SELECT column1, column2 FROM table_name WHERE condition;",
            "meaning of life": "This is a philosophical question with many perspectives across cultures.",
            "current year": "The current year is 2026.",
            "weather in tokyo": "Tokyo has a humid subtropical climate. Current conditions vary by season.",
        }

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 3,
                },
            },
            "required": ["query"],
        }

    def execute(self, query: str, max_results: int = 3) -> ToolResult:
        fault = self._maybe_inject_fault()
        if fault:
            return fault

        query_lower = query.lower().strip()
        results = []

        for key, value in self._knowledge_base.items():
            if any(term in query_lower for term in key.split()):
                results.append({"title": key, "snippet": value})

        if not results:
            results.append({
                "title": "No results found",
                "snippet": f"No relevant information found for '{query}'. Please try a different query.",
            })

        results = results[:max_results]

        return ToolResult(
            success=True,
            output=json.dumps(results, indent=2),
            data=results,
        )


class KnowledgeBaseTool(Tool):
    """Structured knowledge base for factual lookups."""

    def __init__(self):
        super().__init__(
            name="knowledge_lookup",
            description="Look up structured facts from a knowledge base. Use for checking facts, dates, and definitions.",
        )
        self._facts = {
            "python_created": 1991,
            "python_creator": "Guido van Rossum",
            "transformers_introduced": 2017,
            "attention_paper": "Attention Is All You Need",
            "first_llm": "GPT-1 (2018)",
            "gpt3_params": "175 billion",
            "bert_params": "340 million",
        }

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "fact_id": {
                    "type": "string",
                    "description": "The identifier for the fact to look up",
                },
            },
            "required": ["fact_id"],
        }

    def execute(self, fact_id: str) -> ToolResult:
        fault = self._maybe_inject_fault()
        if fault:
            return fault

        if fact_id in self._facts:
            return ToolResult(
                success=True,
                output=json.dumps({"fact_id": fact_id, "value": self._facts[fact_id]}),
                data={"fact_id": fact_id, "value": self._facts[fact_id]},
            )
        else:
            return ToolResult(
                success=False,
                output="",
                error=f"Fact '{fact_id}' not found in knowledge base",
            )


class WebSearchTool(Tool):
    """More realistic web search with simulated results."""

    def __init__(self):
        super().__init__(
            name="web_search_extended",
            description="Extended web search with snippets and source URLs. Returns richer information.",
        )
        self._database = {
            "climate change causes": [
                {"source": "IPCC Report 2025", "snippet": "Greenhouse gas emissions from human activities are the primary driver of climate change, with CO2 concentrations reaching 427 ppm in 2025."},
                {"source": "NASA Climate", "snippet": "The five warmest years on record have all occurred since 2020, with global temperatures rising 1.3°C above pre-industrial levels."},
            ],
            "machine learning definition": [
                {"source": "Mitchell (1997)", "snippet": "Machine learning is the study of computer algorithms that improve automatically through experience."},
                {"source": "Stanford CS229", "snippet": "Machine learning explores the study and construction of algorithms that can learn from and make predictions on data."},
            ],
        }

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        }

    def execute(self, query: str) -> ToolResult:
        fault = self._maybe_inject_fault()
        if fault:
            return fault

        query_lower = query.lower().strip()
        for key, results in self._database.items():
            if any(term in query_lower for term in key.split()):
                return ToolResult(
                    success=True,
                    output=json.dumps(results, indent=2),
                    data=results,
                )

        return ToolResult(
            success=True,
            output=json.dumps([{"source": "General", "snippet": f"Information about '{query}' was not found in the local database."}]),
            data=[],
        )
