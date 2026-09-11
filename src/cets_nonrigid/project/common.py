"""Gate objects shared by checks, project writers and the report."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

GateStatus = Literal["pass", "fail", "not_evaluated"]


class Gate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: GateStatus
    value: Any = None
    expected: Any = None
    note: str = ""

    def __str__(self) -> str:
        s = f"{self.name}: {self.status}"
        if self.value is not None or self.expected is not None:
            s += f" (value {self.value!r}, expected {self.expected!r})"
        if self.note:
            s += f" - {self.note}"
        return s


def gate(name: str, ok: bool | None, *, value: Any = None, expected: Any = None, note: str = "") -> Gate:
    status: GateStatus = "not_evaluated" if ok is None else ("pass" if ok else "fail")
    return Gate(name=name, status=status, value=value, expected=expected, note=note)


def failed(gates: list[Gate]) -> list[Gate]:
    return [g for g in gates if g.status == "fail"]
