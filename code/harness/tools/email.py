"""Email and notification tools."""

from .base import Tool, ToolResult
from typing import Any, Dict, List, Optional
import json


class EmailTool(Tool):
    """Simulated email tool for sending and managing emails."""

    def __init__(self):
        super().__init__(
            name="email",
            description="Send and manage emails. Can send, list inbox, and search messages.",
        )
        self._inbox = [
            {"id": "1", "from": "boss@company.com", "subject": "Q3 Review", "body": "Please prepare the quarterly review slides by Friday.", "date": "2026-07-26"},
            {"id": "2", "from": "alice@company.com", "subject": "Code Review Request", "body": "Can you review my PR #342 for the new API endpoint?", "date": "2026-07-25"},
            {"id": "3", "from": "meeting-robot@company.com", "subject": "Meeting Reminder", "body": "Reminder: Team standup tomorrow at 9:00 AM.", "date": "2026-07-26"},
            {"id": "4", "from": "hr@company.com", "subject": "Benefits Update", "body": "Open enrollment for health insurance ends August 15.", "date": "2026-07-24"},
            {"id": "5", "from": "it-support@company.com", "subject": "Password Expiry", "body": "Your password will expire in 7 days. Please update.", "date": "2026-07-23"},
        ]
        self._next_id = 6
        self._sent = []

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["send", "list_inbox", "search", "read"],
                    "description": "Email action to perform",
                },
                "to": {
                    "type": "string",
                    "description": "Recipient email (for send action)",
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject (for send action)",
                },
                "body": {
                    "type": "string",
                    "description": "Email body (for send action)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for search action)",
                },
                "email_id": {
                    "type": "string",
                    "description": "Email ID (for read action)",
                },
            },
            "required": ["action"],
        }

    def execute(
        self,
        action: str,
        to: Optional[str] = None,
        subject: Optional[str] = None,
        body: Optional[str] = None,
        query: Optional[str] = None,
        email_id: Optional[str] = None,
    ) -> ToolResult:
        fault = self._maybe_inject_fault()
        if fault:
            return fault

        if action == "list_inbox":
            return ToolResult(
                success=True,
                output=json.dumps([{"id": e["id"], "from": e["from"], "subject": e["subject"], "date": e["date"]} for e in self._inbox], indent=2),
                data=self._inbox,
            )

        elif action == "read":
            if not email_id:
                return ToolResult(success=False, output="", error="email_id required")
            for email in self._inbox:
                if email["id"] == email_id:
                    return ToolResult(success=True, output=json.dumps(email, indent=2), data=email)
            return ToolResult(success=False, output="", error=f"Email {email_id} not found")

        elif action == "search":
            if not query:
                return ToolResult(success=False, output="", error="query required")
            query_lower = query.lower()
            results = [e for e in self._inbox if query_lower in e["subject"].lower() or query_lower in e["body"].lower()]
            return ToolResult(
                success=True,
                output=json.dumps(results, indent=2),
                data=results,
            )

        elif action == "send":
            if not to or not subject or not body:
                return ToolResult(success=False, output="", error="to, subject, and body are required")
            sent = {"id": str(self._next_id), "to": to, "subject": subject, "body": body, "date": "2026-07-26"}
            self._sent.append(sent)
            self._next_id += 1
            return ToolResult(
                success=True,
                output=json.dumps({"message": f"Email sent to {to}", "id": sent["id"]}, indent=2),
                data=sent,
            )

        return ToolResult(success=False, output="", error=f"Unknown action: {action}")


class NotificationTool(Tool):
    """Send notifications through various channels."""

    def __init__(self):
        super().__init__(
            name="notification",
            description="Send notifications through Slack, Teams, or other channels.",
        )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Notification channel (slack, teams, email)",
                },
                "message": {
                    "type": "string",
                    "description": "Notification message content",
                },
                "urgency": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Urgency level",
                    "default": "medium",
                },
            },
            "required": ["channel", "message"],
        }

    def execute(self, channel: str, message: str, urgency: str = "medium") -> ToolResult:
        fault = self._maybe_inject_fault()
        if fault:
            return fault

        if channel not in ["slack", "teams", "email"]:
            return ToolResult(success=False, output="", error=f"Unknown channel: {channel}")

        return ToolResult(
            success=True,
            output=json.dumps({
                "status": f"Notification sent via {channel}",
                "urgency": urgency,
                "message_preview": message[:100],
            }, indent=2),
            data={"channel": channel, "sent": True},
        )
