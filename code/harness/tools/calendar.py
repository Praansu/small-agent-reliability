"""Calendar and scheduling tools."""

from .base import Tool, ToolResult
from typing import Any, Dict, List, Optional
import json
from datetime import datetime, timedelta


class CalendarTool(Tool):
    """Calendar management tool for scheduling events."""

    def __init__(self):
        super().__init__(
            name="calendar",
            description="Manage calendar events. Can create, check availability, list, and cancel events.",
        )
        self._events: List[Dict[str, Any]] = [
            {"id": "1", "title": "Team Standup", "date": "2026-07-27", "time": "09:00", "duration_minutes": 30},
            {"id": "2", "title": "Lunch Meeting", "date": "2026-07-27", "time": "12:00", "duration_minutes": 60},
            {"id": "3", "title": "Project Review", "date": "2026-07-28", "time": "14:00", "duration_minutes": 45},
            {"id": "4", "title": "Client Call", "date": "2026-07-29", "time": "10:00", "duration_minutes": 30},
            {"id": "5", "title": "Workshop Planning", "date": "2026-07-30", "time": "15:00", "duration_minutes": 90},
        ]
        self._next_id = 6

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "cancel", "check_availability"],
                    "description": "Action to perform on the calendar",
                },
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format",
                },
                "title": {
                    "type": "string",
                    "description": "Event title (for create action)",
                },
                "time": {
                    "type": "string",
                    "description": "Time in HH:MM format (for create action)",
                },
                "event_id": {
                    "type": "string",
                    "description": "Event ID (for cancel action)",
                },
            },
            "required": ["action"],
        }

    def execute(
        self,
        action: str,
        date: Optional[str] = None,
        title: Optional[str] = None,
        time: Optional[str] = None,
        event_id: Optional[str] = None,
    ) -> ToolResult:
        fault = self._maybe_inject_fault()
        if fault:
            return fault

        if action == "list":
            if date:
                events = [e for e in self._events if e["date"] == date]
            else:
                events = self._events
            return ToolResult(
                success=True,
                output=json.dumps(events, indent=2),
                data=events,
            )

        elif action == "create":
            if not title or not date or not time:
                return ToolResult(
                    success=False,
                    output="",
                    error="Missing required parameters: title, date, and time are required",
                )
            event = {
                "id": str(self._next_id),
                "title": title,
                "date": date,
                "time": time,
                "duration_minutes": 60,
            }
            self._events.append(event)
            self._next_id += 1
            return ToolResult(
                success=True,
                output=json.dumps({"message": f"Event '{title}' created", "event_id": event["id"]}),
                data=event,
            )

        elif action == "cancel":
            if not event_id:
                return ToolResult(
                    success=False,
                    output="",
                    error="event_id is required for cancel action",
                )
            for i, e in enumerate(self._events):
                if e["id"] == event_id:
                    removed = self._events.pop(i)
                    return ToolResult(
                        success=True,
                        output=json.dumps({"message": f"Event '{removed['title']}' cancelled"}),
                        data=removed,
                    )
            return ToolResult(
                success=False,
                output="",
                error=f"Event with id '{event_id}' not found",
            )

        elif action == "check_availability":
            if not date:
                return ToolResult(
                    success=False,
                    output="",
                    error="date is required for check_availability",
                )
            day_events = [e for e in self._events if e["date"] == date]
            occupied = [(e["time"], e["time"]) for e in day_events]
            return ToolResult(
                success=True,
                output=json.dumps({"date": date, "events_count": len(day_events), "occupied_slots": occupied}),
                data={"date": date, "events": day_events},
            )

        return ToolResult(
            success=False,
            output="",
            error=f"Unknown action: {action}",
        )


class ScheduleManager(Tool):
    """Advanced scheduling with conflict detection."""

    def __init__(self):
        super().__init__(
            name="schedule_manager",
            description="Smart scheduling tool that finds optimal meeting times across participants.",
        )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of participant names",
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "Meeting duration in minutes",
                },
                "date": {
                    "type": "string",
                    "description": "Date for the meeting",
                },
            },
            "required": ["participants", "duration_minutes", "date"],
        }

    def execute(self, participants: List[str], duration_minutes: int, date: str) -> ToolResult:
        fault = self._maybe_inject_fault()
        if fault:
            return fault

        # Simulate finding a slot
        suggested_times = ["09:00", "10:00", "11:00", "14:00", "15:00", "16:00"]
        return ToolResult(
            success=True,
            output=json.dumps({
                "date": date,
                "suggested_slots": suggested_times[:3],
                "participants": participants,
                "duration_minutes": duration_minutes,
            }, indent=2),
            data={"date": date, "slots": suggested_times[:3]},
        )
