"""
Claude Memory Bridge - Never forget anything
"""

from .autocapture import AutoCapture, ClawdbotMemoryHooks, get_memory
from .core import Event, MemoryBridge
from .memory_os import MemoryOS, MemoryPolicy
from .retrieval import MemoryRetrieval, SearchResult

__version__ = "0.2.0"

__all__ = [
    "MemoryBridge",
    "Event",
    "MemoryRetrieval",
    "SearchResult",
    "AutoCapture",
    "ClawdbotMemoryHooks",
    "get_memory",
    "MemoryOS",
    "MemoryPolicy",
]
