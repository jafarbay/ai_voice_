"""Natasha-based NER detector for Russian PERSON / LOCATION (ADDRESS).

Natasha ships its own embeddings / tagger. First call downloads ~70 MB of
navec/slovnet weights to the user cache; subsequent calls are fast.
"""

from __future__ import annotations

from threading import Lock
from typing import Any

from app.pipeline.pii.types import PIISpan, PIIType

SOURCE = "natasha"
_PERSON_CONF = 0.85
_ADDRESS_CONF = 0.75

_nat_state: dict[str, Any] = {}
_nat_lock = Lock()


def _get_natasha() -> dict[str, Any]:
    """Lazily initialize Natasha components and cache them at module level."""
    global _nat_state
    if _nat_state:
        return _nat_state
    with _nat_lock:
        if _nat_state:
            return _nat_state
        from natasha import (  # imported lazily
            Doc,
            MorphVocab,
            NewsEmbedding,
            NewsNERTagger,
            Segmenter,
        )

        segmenter = Segmenter()
        morph_vocab = MorphVocab()
        emb = NewsEmbedding()
        ner_tagger = NewsNERTagger(emb)
        _nat_state = {
            "Doc": Doc,
            "segmenter": segmenter,
            "morph_vocab": morph_vocab,
            "ner_tagger": ner_tagger,
        }
        return _nat_state


def detect_natasha(text: str) -> list[PIISpan]:
    """Run Natasha NER and map PER→PERSON, LOC→ADDRESS."""
    if not text.strip():
        return []

    state = _get_natasha()
    Doc = state["Doc"]
    doc = Doc(text)
    doc.segment(state["segmenter"])
    doc.tag_ner(state["ner_tagger"])

    spans: list[PIISpan] = []
    for span in doc.spans:
        span_type = getattr(span, "type", None)
        if span_type == "PER":
            pii_type = PIIType.PERSON
            confidence = _PERSON_CONF
        elif span_type == "LOC":
            pii_type = PIIType.ADDRESS
            confidence = _ADDRESS_CONF
        else:
            # ORG etc. are not treated as PII.
            continue

        start = span.start
        end = span.stop
        matched_text = getattr(span, "text", None) or text[start:end]
        spans.append(
            PIISpan(
                type=pii_type,
                text=matched_text,
                start_char=start,
                end_char=end,
                source=SOURCE,
                confidence=confidence,
            )
        )
    return spans
