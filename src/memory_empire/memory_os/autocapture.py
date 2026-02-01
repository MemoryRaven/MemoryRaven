"""
Auto-capture middleware for Clawdbot
Automatically captures all events without explicit calls
"""

import logging
import os
import subprocess
from typing import Any

from .core import Event, MemoryBridge

logger = logging.getLogger(__name__)


class AutoCapture:
    """
    Middleware that automatically captures:
    - All messages (in/out)
    - Tool calls and results
    - File operations
    - Web fetches
    - Decisions
    - Git commits
    """

    def __init__(self, memory_bridge: MemoryBridge):
        self.memory = memory_bridge
        self.session_id = os.environ.get("CLAUDE_SESSION_ID", "main")
        self.actor_id = os.environ.get("CLAUDE_ACTOR_ID", "raven")

    def capture_message(
        self,
        direction: str,  # 'inbound' or 'outbound'
        channel: str,
        text: str,
        metadata: dict = None,
    ) -> str | None:
        """Capture a message event"""
        return self.memory.capture_message(
            channel=channel,
            text=text,
            thread_id=self.session_id,
            actor_id=self.actor_id
            if direction == "outbound"
            else (metadata.get("from") if metadata else None),
            direction=direction,
            metadata=metadata or {},
        )

    def capture_tool_call(
        self, tool_name: str, arguments: dict, result: Any, success: bool = True
    ) -> str | None:
        """Capture a tool call event"""
        return self.memory.capture(
            Event(
                source="tool",
                event_type="tool_call",
                content_text=f"Called {tool_name}: {str(arguments)[:200]}",
                content_json={
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result if success else None,
                    "error": result if not success else None,
                    "success": success,
                },
                thread_id=self.session_id,
                actor_id=self.actor_id,
            )
        )

    def capture_file_operation(
        self,
        operation: str,  # read, write, edit, delete
        file_path: str,
        content: str = None,
        diff: str = None,
    ) -> str | None:
        """Capture file operations"""
        # Auto-detect if this is a memory file
        is_memory_file = "memory/" in file_path or "MEMORY.md" in file_path

        event = Event(
            source="file",
            event_type=f"file_{operation}",
            content_text=f"{operation} {file_path}",
            content_json={
                "operation": operation,
                "file_path": file_path,
                "content": content[:1000] if content else None,
                "diff": diff,
                "is_memory_file": is_memory_file,
            },
            thread_id=self.session_id,
            actor_id=self.actor_id,
            tags=["memory"] if is_memory_file else [],
        )

        # For memory files, also extract structured updates
        if is_memory_file and content:
            self._extract_memory_updates(file_path, content)

        return self.memory.capture(event)

    def capture_web_fetch(self, url: str, content: str, metadata: dict = None) -> str | None:
        """Capture web content fetches"""
        return self.memory.capture_web_content(
            url=url,
            content=content,
            thread_id=self.session_id,
            actor_id=self.actor_id,
            metadata=metadata or {},
        )

    def capture_decision(
        self, decision: str, rationale: str = None, alternatives: list[str] = None
    ) -> str | None:
        """Capture decisions made"""
        return self.memory.capture_decision(
            decision=decision,
            rationale=rationale,
            thread_id=self.session_id,
            actor_id=self.actor_id,
            alternatives=alternatives,
        )

    def capture_git_commit(self, repo_path: str = ".") -> str | None:
        """Capture git commits automatically"""
        try:
            # Get latest commit info
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=format:%H|%s|%an|%ae|%ai"],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return None

            commit_hash, subject, author, email, date = result.stdout.strip().split("|")

            # Get diff stat
            diff_result = subprocess.run(
                ["git", "diff", "--stat", f"{commit_hash}~1", commit_hash],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )

            diff_stat = diff_result.stdout if diff_result.returncode == 0 else ""

            # Get changed files
            files_result = subprocess.run(
                ["git", "diff", "--name-only", f"{commit_hash}~1", commit_hash],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )

            changed_files = (
                files_result.stdout.strip().split("\n") if files_result.returncode == 0 else []
            )

            return self.memory.capture(
                Event(
                    source="git",
                    event_type="commit",
                    content_text=f"Git commit: {subject}",
                    content_json={
                        "commit_hash": commit_hash,
                        "subject": subject,
                        "author": author,
                        "email": email,
                        "date": date,
                        "diff_stat": diff_stat,
                        "changed_files": changed_files,
                        "repo_path": os.path.abspath(repo_path),
                    },
                    thread_id=self.session_id,
                    actor_id=self.actor_id,
                    observed_at=date,
                )
            )

        except Exception as e:
            logger.error(f"Failed to capture git commit: {e}")
            return None

    def _extract_memory_updates(self, file_path: str, content: str):
        """Extract structured information from memory file updates"""
        # Look for patterns like decisions, todos, important facts
        lines = content.split("\n")

        for line in lines:
            line = line.strip()

            # Decision pattern
            if line.startswith("DECISION:") or line.startswith("**DECISION"):
                decision_text = line.replace("DECISION:", "").replace("**", "").strip()
                self.capture_decision(decision_text)

            # TODO pattern
            elif line.startswith("TODO:") or line.startswith("- [ ]"):
                todo_text = line.replace("TODO:", "").replace("- [ ]", "").strip()
                self.memory.capture(
                    Event(
                        source="memory",
                        event_type="todo",
                        content_text=f"TODO: {todo_text}",
                        content_json={"todo": todo_text, "file": file_path},
                        thread_id=self.session_id,
                        actor_id=self.actor_id,
                    )
                )

            # Important pattern
            elif line.startswith("IMPORTANT:") or line.startswith("⚠️"):
                important_text = line.replace("IMPORTANT:", "").replace("⚠️", "").strip()
                self.memory.capture(
                    Event(
                        source="memory",
                        event_type="important",
                        content_text=f"Important: {important_text}",
                        content_json={"note": important_text, "file": file_path},
                        thread_id=self.session_id,
                        actor_id=self.actor_id,
                        tags=["important"],
                    )
                )


# Clawdbot integration hooks
class ClawdbotMemoryHooks:
    """
    Integration hooks for Clawdbot to automatically capture everything
    """

    def __init__(self, pinecone_api_key: str | None = None):
        # Initialize memory bridge
        self.memory = MemoryBridge(
            pinecone_api_key=pinecone_api_key,
            pinecone_environment=os.environ.get("PINECONE_ENVIRONMENT"),
            pinecone_index=os.environ.get("PINECONE_INDEX", "claude-memory"),
        )
        self.autocapture = AutoCapture(self.memory)

        # Hook into Clawdbot events
        self._setup_hooks()

    def _setup_hooks(self):
        """Set up event hooks (this would integrate with Clawdbot's event system)"""
        # In practice, these would hook into Clawdbot's middleware system
        # For now, these are the methods that would be called
        pass

    def on_message_received(self, message: dict):
        """Hook for incoming messages"""
        self.autocapture.capture_message(
            direction="inbound",
            channel=message.get("channel", "unknown"),
            text=message.get("text", ""),
            metadata=message,
        )

    def on_message_sent(self, message: dict):
        """Hook for outgoing messages"""
        self.autocapture.capture_message(
            direction="outbound",
            channel=message.get("channel", "unknown"),
            text=message.get("text", ""),
            metadata=message,
        )

    def on_tool_called(self, tool_name: str, arguments: dict, result: Any, success: bool = True):
        """Hook for tool calls"""
        self.autocapture.capture_tool_call(tool_name, arguments, result, success)

    def on_file_operation(self, operation: str, file_path: str, **kwargs):
        """Hook for file operations"""
        self.autocapture.capture_file_operation(operation, file_path, **kwargs)


# Convenience function to get memory instance
_memory_instance = None


def get_memory() -> MemoryBridge:
    """Get or create the global memory instance"""
    global _memory_instance
    if not _memory_instance:
        _memory_instance = MemoryBridge(
            pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
            pinecone_environment=os.environ.get("PINECONE_ENVIRONMENT"),
            pinecone_index=os.environ.get("PINECONE_INDEX", "claude-memory"),
        )
    return _memory_instance
