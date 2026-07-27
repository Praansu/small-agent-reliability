"""Data processing and query tools."""

from .base import Tool, ToolResult
from typing import Any, Dict, List, Optional
import json
import random


class DataQueryTool(Tool):
    """Query structured data from a database."""

    def __init__(self):
        super().__init__(
            name="data_query",
            description="Query structured data from the database. Supports SELECT queries with filtering.",
        )
        # Simulated database tables
        self._tables = {
            "employees": [
                {"id": 1, "name": "Alice", "department": "Engineering", "salary": 95000},
                {"id": 2, "name": "Bob", "department": "Marketing", "salary": 72000},
                {"id": 3, "name": "Charlie", "department": "Engineering", "salary": 110000},
                {"id": 4, "name": "Diana", "department": "Sales", "salary": 85000},
                {"id": 5, "name": "Eve", "department": "Engineering", "salary": 105000},
            ],
            "products": [
                {"id": 1, "name": "Widget A", "price": 29.99, "stock": 150},
                {"id": 2, "name": "Widget B", "price": 49.99, "stock": 75},
                {"id": 3, "name": "Gadget X", "price": 99.99, "stock": 200},
                {"id": 4, "name": "Gadget Y", "price": 149.99, "stock": 30},
                {"id": 5, "name": "Tool Pro", "price": 199.99, "stock": 0},
            ],
            "sales": [
                {"id": 1, "product_id": 1, "amount": 5000, "month": "2026-01"},
                {"id": 2, "product_id": 2, "amount": 7500, "month": "2026-01"},
                {"id": 3, "product_id": 1, "amount": 6200, "month": "2026-02"},
                {"id": 4, "product_id": 3, "amount": 12000, "month": "2026-02"},
                {"id": 5, "product_id": 2, "amount": 8100, "month": "2026-03"},
            ],
        }

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "Table name to query",
                },
                "filters": {
                    "type": "object",
                    "description": "Key-value pairs to filter by (optional)",
                },
            },
            "required": ["table"],
        }

    def execute(self, table: str, filters: Optional[Dict[str, Any]] = None) -> ToolResult:
        fault = self._maybe_inject_fault()
        if fault:
            return fault

        if table not in self._tables:
            return ToolResult(
                success=False,
                output="",
                error=f"Table '{table}' not found. Available tables: {list(self._tables.keys())}",
            )

        results = self._tables[table]
        if filters:
            filtered = []
            for row in results:
                match = True
                for key, value in filters.items():
                    if key in row and str(row[key]) != str(value):
                        match = False
                        break
                if match:
                    filtered.append(row)
            results = filtered

        return ToolResult(
            success=True,
            output=json.dumps(results, indent=2),
            data=results,
            metadata={"row_count": len(results)},
        )


class CSVProcessor(Tool):
    """Process CSV data with various operations."""

    def __init__(self):
        super().__init__(
            name="csv_processor",
            description="Process CSV data: filter, aggregate, sort, and transform tabular data.",
        )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["filter", "aggregate", "sort", "summary"],
                    "description": "Operation to perform",
                },
                "data": {
                    "type": "array",
                    "description": "Array of row objects to process",
                },
                "column": {
                    "type": "string",
                    "description": "Column name to operate on",
                },
                "value": {
                    "type": "string",
                    "description": "Value to filter by or aggregate key",
                },
            },
            "required": ["operation", "data"],
        }

    def execute(
        self,
        operation: str,
        data: List[Dict[str, Any]],
        column: Optional[str] = None,
        value: Optional[str] = None,
    ) -> ToolResult:
        fault = self._maybe_inject_fault()
        if fault:
            return fault

        if operation == "filter":
            if not column or value is None:
                return ToolResult(success=False, output="", error="column and value required for filter")
            filtered = [r for r in data if str(r.get(column, "")) == str(value)]
            return ToolResult(success=True, output=json.dumps(filtered, indent=2), data=filtered)

        elif operation == "aggregate":
            if not column:
                return ToolResult(success=False, output="", error="column required for aggregate")
            values = [r.get(column, 0) for r in data if isinstance(r.get(column), (int, float))]
            if not values:
                return ToolResult(success=True, output=json.dumps({"message": "No numeric data to aggregate"}))
            result = {
                "mean": sum(values) / len(values),
                "max": max(values),
                "min": min(values),
                "count": len(values),
                "sum": sum(values),
            }
            return ToolResult(success=True, output=json.dumps(result, indent=2), data=result)

        elif operation == "sort":
            if not column:
                return ToolResult(success=False, output="", error="column required for sort")
            sorted_data = sorted(data, key=lambda r: str(r.get(column, "")))
            return ToolResult(success=True, output=json.dumps(sorted_data, indent=2), data=sorted_data)

        elif operation == "summary":
            return ToolResult(
                success=True,
                output=json.dumps({
                    "row_count": len(data),
                    "columns": list(data[0].keys()) if data else [],
                }, indent=2),
                data={"row_count": len(data), "columns": list(data[0].keys()) if data else []},
            )

        return ToolResult(success=False, output="", error=f"Unknown operation: {operation}")
