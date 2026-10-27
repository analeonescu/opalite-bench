"""Append-only event log. Everything the scorers see comes from here."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Event:
    t: float
    kind: str  # "tool_call" | "sim" (hidden ground truth) | "real" (your backend)
    name: str
    data: dict[str, Any] = field(default_factory=dict)


class EventLog:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def add(self, kind: str, name: str, **data: Any) -> Event:
        e = Event(time.time(), kind, name, data)
        self.events.append(e)
        return e

    def tool_calls(self, name: str | None = None) -> list[Event]:
        return [e for e in self.events if e.kind == "tool_call" and (name is None or e.name == name)]

    def to_dicts(self) -> list[dict]:
        return [asdict(e) for e in self.events]


def events_from_dicts(rows: list[dict]) -> list[Event]:
    return [Event(**r) for r in rows]
