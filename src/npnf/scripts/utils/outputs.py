"""CSV and JSON output serialization for scripts."""

from __future__ import annotations

import csv
import html
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


def write_csv(
    path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def markdown_table(
    header: Sequence[str], rows: Iterable[Sequence[str]], *, label_columns: int = 1
) -> str:
    """A Markdown table, ``label_columns`` left-aligned columns, then right-aligned."""
    alignments = ["---"] * label_columns + ["---:"] * (len(header) - label_columns)
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(alignments) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    return "\n".join(lines)


def html_table(rows: Sequence[Mapping[str, object]]) -> str:
    """An HTML table of rows with the keys of the first row as columns."""
    columns = list(rows[0])
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            text = f"{value:.6g}" if isinstance(value, float) else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )
