from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    for row in rows[1:]:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = [_format(row.get(field)) for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return "" if value is None else str(value)

