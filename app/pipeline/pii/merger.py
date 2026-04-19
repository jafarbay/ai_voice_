"""Merge PII spans from multiple detectors.

Rules:
- Spans are sorted by ``start_char``.
- Overlapping spans: the one whose source has higher priority
  (regex > word_phone > llm > natasha) wins. If both sources are equal,
  the longer span wins; ties broken by earlier start.
- Exact duplicates are collapsed.
- Output is sorted by ``start_char``.
"""

from __future__ import annotations

from app.pipeline.pii.types import SOURCE_PRIORITY, PIISpan


def _overlaps(a: PIISpan, b: PIISpan) -> bool:
    return a.start_char < b.end_char and b.start_char < a.end_char


def _priority(span: PIISpan) -> int:
    return SOURCE_PRIORITY.get(str(span.source), 0)


def _prefer(a: PIISpan, b: PIISpan) -> PIISpan:
    """Pick the winning span among two overlapping candidates."""
    pa, pb = _priority(a), _priority(b)
    if pa != pb:
        return a if pa > pb else b
    # Same source -> longer wins, tie -> earlier start.
    la = a.end_char - a.start_char
    lb = b.end_char - b.start_char
    if la != lb:
        return a if la > lb else b
    return a if a.start_char <= b.start_char else b


def merge_spans(spans: list[PIISpan]) -> list[PIISpan]:
    if not spans:
        return []

    # Dedup exact (type, start, end, text, source) tuples.
    seen: set[tuple] = set()
    unique: list[PIISpan] = []
    for s in spans:
        key = (s.type, s.start_char, s.end_char, s.text, s.source)
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)

    # Sort by start, then by descending priority so higher-priority spans
    # get considered first when scanning.
    unique.sort(key=lambda s: (s.start_char, -_priority(s), -(s.end_char - s.start_char)))

    # Greedy merge: iterate, keep `result`, and whenever a new span overlaps
    # the last accepted one, decide who wins and possibly replace.
    result: list[PIISpan] = []
    for span in unique:
        replaced = False
        # Walk from end since overlaps are likely with the last one(s).
        i = len(result) - 1
        while i >= 0 and result[i].end_char > span.start_char:
            if _overlaps(result[i], span):
                winner = _prefer(result[i], span)
                if winner is span:
                    # Current span beats result[i]; remove it and keep scanning.
                    result.pop(i)
                    i -= 1
                    continue
                else:
                    # Existing span wins; drop the incoming one.
                    replaced = True
                    break
            i -= 1
        if not replaced:
            result.append(span)

    result.sort(key=lambda s: s.start_char)
    return result
