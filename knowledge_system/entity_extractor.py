"""Entity extraction: rule-based (no dependencies), spaCy (optional), composite."""
from __future__ import annotations

import bisect
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .knowledge_core import Entity, Provenance, make_entity_id, normalize_name

logger = logging.getLogger(__name__)

__all__ = [
    "ExtractedMention", "EntityExtractor", "RuleBasedEntityExtractor",
    "SpacyEntityExtractor", "CompositeEntityExtractor",
    "split_sentences", "resolve_overlaps",
]


@dataclass
class ExtractedMention:
    """A span of text recognised as an entity."""

    text: str
    canonical: str
    type: str
    start: int
    end: int
    confidence: float
    source: str = "rule"
    sentence_index: int = -1

    @property
    def entity_id(self) -> str:
        return make_entity_id(self.canonical, self.type)


# --------------------------------------------------------------------------- #
# Text utilities
# --------------------------------------------------------------------------- #
_ABBREV = {"dr", "mr", "mrs", "ms", "prof", "inc", "ltd", "corp", "co", "st", "vs",
           "etc", "jr", "sr", "no", "fig", "eg", "ie"}
_SENT_BREAK = re.compile(r"([.!?]+[\"')\]]*)(\s+)(?=[\"'(\[]?[A-Z0-9])|\n\s*\n")


def split_sentences(text: str) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` spans of sentences (abbreviation-aware)."""
    spans: List[Tuple[int, int]] = []
    start = 0
    for m in _SENT_BREAK.finditer(text):
        if m.group(1):
            tail = re.search(r"[A-Za-z.]+$", text[start:m.start(1)])
            word = tail.group(0).rstrip(".").casefold() if tail else ""
            if word in _ABBREV or len(word) == 1:
                continue
            end = m.end(1)
        else:
            end = m.start()
        spans.append((start, end))
        start = m.end()
    spans.append((start, len(text)))
    cleaned = []
    for s, e in spans:
        seg = text[s:e]
        lead = len(seg) - len(seg.lstrip())
        trail = len(seg) - len(seg.rstrip())
        if e - s - lead - trail > 0:
            cleaned.append((s + lead, e - trail))
    return cleaned


def sentence_index_for(spans: Sequence[Tuple[int, int]], position: int) -> int:
    starts = [s for s, _ in spans]
    return max(0, bisect.bisect_right(starts, position) - 1)


def resolve_overlaps(mentions: Iterable[ExtractedMention]) -> List[ExtractedMention]:
    """Keep the longest (then most confident) mention among overlapping spans."""
    ordered = sorted(mentions, key=lambda m: (-(m.end - m.start), -m.confidence, m.start))
    kept: List[ExtractedMention] = []
    for m in ordered:
        if all(m.end <= k.start or m.start >= k.end for k in kept):
            kept.append(m)
    return sorted(kept, key=lambda m: m.start)


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class EntityExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> List[ExtractedMention]:
        """Return non-overlapping mentions sorted by position."""

    def mentions_to_entities(self, mentions: Iterable[ExtractedMention],
                             provenance: Provenance) -> List[Entity]:
        """Collapse mentions into one :class:`Entity` per canonical id."""
        grouped: Dict[str, Entity] = {}
        for m in mentions:
            ent = grouped.get(m.entity_id)
            if ent is None:
                ent = Entity(
                    id=m.entity_id, name=m.canonical, type=m.type, confidence=m.confidence,
                    provenance=[replace(provenance, confidence=m.confidence)],
                )
                grouped[ent.id] = ent
            else:
                ent.confidence = max(ent.confidence, m.confidence)
            if normalize_name(m.text) != normalize_name(m.canonical) and \
                    normalize_name(m.text) not in {normalize_name(a) for a in ent.aliases}:
                ent.aliases.append(m.text.strip())
        return list(grouped.values())


# --------------------------------------------------------------------------- #
# Rule-based extractor
# --------------------------------------------------------------------------- #
_MONTHS = ("January|February|March|April|May|June|July|August|September|October|"
           "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec")
_DATE_RE = re.compile(
    rf"\b(?:\d{{4}}-\d{{2}}-\d{{2}}"
    rf"|(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\.?,?\s+\d{{4}}"
    rf"|(?:{_MONTHS})\.?\s+\d{{4}})\b"
)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
_TITLE_RE = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Prof|Professor|Sir|Dame)\.?[ \t]+([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){0,2})"
)
_CAP_RUN = re.compile(
    r"\b[A-Z][A-Za-z0-9&'’\-]*(?:[ \t]+(?:(?:of|de|del|von|van|la)[ \t]+)?[A-Z][A-Za-z0-9&'’\-]*)*"
)

_STOP_CAPS = frozenset("""
The A An In On At It This That These Those He She They We I You His Her Its Their Our My Your
However But And Or If When While After Before Also For With By From As Then There Here Where
What Who Which Why How Not No Yes So Because Since Although Though Meanwhile Furthermore
Moreover Therefore Thus Today Yesterday Tomorrow Monday Tuesday Wednesday Thursday Friday
Saturday Sunday January February March April May June July August September October November
December Jan Feb Mar Apr Jun Jul Aug Sep Sept Oct Nov Dec Some Many Most Several Each Every
All Both Another Other Such One Two Three First Second Third New Mr Mrs Ms Dr Prof Sir Dame
Let Please Note See Figure Table Section Chapter Is Are Was Were Be Been Do Does Did Can Could
Will Would Should May Might Must Has Have Had CEO CTO CFO COO CIO VP
""".split())
_CONNECTORS = frozenset({"of", "de", "del", "von", "van", "la"})
_ACRONYM_STOP = frozenset({"API", "CPU", "GPU", "URL", "HTTP", "HTTPS", "JSON", "XML", "SQL",
                           "HTML", "CSS", "PDF", "RAM", "USB", "FAQ", "TODO", "NOTE", "ID"})
_ORG_SUFFIX = frozenset({
    "inc", "corp", "corporation", "ltd", "llc", "gmbh", "co", "company", "group", "labs", "lab",
    "technologies", "systems", "foundation", "university", "institute", "association", "agency",
    "bank", "holdings", "partners", "solutions", "industries", "laboratories", "society",
    "council", "ministry", "department", "college", "school", "hospital", "airlines", "robotics",
})
_ORG_LEAD = frozenset({"university", "institute", "bank", "ministry", "department", "college",
                       "school", "hospital", "council", "society"})
_EVENT_SUFFIX = frozenset({"conference", "summit", "festival", "olympics", "championship",
                           "symposium", "expo", "hackathon", "cup", "war"})
_LOC_SUFFIX = frozenset({"city", "river", "mountain", "mountains", "lake", "island", "islands",
                         "valley", "county", "province", "street", "avenue", "bay", "ocean",
                         "sea", "park"})
_LOC_CUES = frozenset({"in", "at", "from", "near", "to", "across", "throughout", "within"})

_DEFAULT_GAZETTEER: Dict[str, str] = {
    **{n: "Technology" for n in (
        "Python", "Java", "JavaScript", "TypeScript", "Rust", "Linux", "PostgreSQL", "SQLite",
        "MySQL", "MongoDB", "Redis", "TensorFlow", "PyTorch", "Kubernetes", "Docker", "React",
        "Electron", "Node.js", "Kafka", "Spark", "Hadoop", "Neo4j")},
    **{n: "Organization" for n in (
        "Google", "Microsoft", "Apple", "Amazon", "Meta", "OpenAI", "Anthropic", "IBM", "Intel",
        "NVIDIA", "Tesla", "Oracle", "Samsung")},
    **{n: "Location" for n in (
        "United States", "United Kingdom", "USA", "UK", "India", "China", "Japan", "Germany",
        "France", "Canada", "Australia", "Brazil", "Bangladesh", "Paris", "London", "Berlin",
        "Tokyo", "New York", "San Francisco", "Dhaka", "Delhi", "Beijing", "Singapore", "Europe",
        "Asia", "Africa")},
}


class RuleBasedEntityExtractor(EntityExtractor):
    """Dependency-free extractor: structured patterns, gazetteer, and heuristics.

    Capitalised phrases that cannot be typed confidently become ``Thing``
    (untyped); :class:`RelationshipMapper` refines them from relation context.
    For production-grade NER, use :class:`SpacyEntityExtractor` or add a
    domain gazetteer.
    """

    def __init__(self, gazetteer: Optional[Mapping[str, str]] = None, *,
                 include_defaults: bool = True) -> None:
        merged: Dict[str, str] = dict(_DEFAULT_GAZETTEER) if include_defaults else {}
        merged.update(gazetteer or {})
        self._gazetteer: Dict[str, str] = {k.casefold(): v for k, v in merged.items()}
        names = sorted(merged, key=len, reverse=True)
        self._gaz_re = (
            re.compile(r"(?<!\w)(?:" + "|".join(re.escape(n) for n in names) + r")(?!\w)")
            if names else None
        )

    def add_to_gazetteer(self, name: str, entity_type: str) -> None:
        self._gazetteer[name.casefold()] = entity_type
        names = sorted({*self._gazetteer, name.casefold()}, key=len, reverse=True)
        # Rebuilt case-insensitively keyed; surface-form matching stays case-sensitive on originals.
        self._gaz_re = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(n) for n in names) + r")(?!\w)",
                                  re.IGNORECASE)

    # ------------------------------------------------------------------ #
    def extract(self, text: str) -> List[ExtractedMention]:
        found: List[ExtractedMention] = []
        found.extend(self._structured(text))
        found.extend(self._gazetteer_matches(text))
        found.extend(self._titled_people(text))
        found.extend(self._capitalised_runs(text))
        mentions = resolve_overlaps(found)
        spans = split_sentences(text)
        for m in mentions:
            m.sentence_index = sentence_index_for(spans, m.start)
        return mentions

    def _structured(self, text: str) -> List[ExtractedMention]:
        out: List[ExtractedMention] = []
        for m in _EMAIL_RE.finditer(text):
            out.append(ExtractedMention(m.group(0), m.group(0).lower(), "Email", m.start(), m.end(), 0.95))
        for m in _URL_RE.finditer(text):
            url = m.group(0).rstrip(".,;:!?")
            out.append(ExtractedMention(url, url, "URL", m.start(), m.start() + len(url), 0.95))
        for m in _DATE_RE.finditer(text):
            out.append(ExtractedMention(m.group(0), m.group(0), "Date", m.start(), m.end(), 0.9))
        return out

    def _gazetteer_matches(self, text: str) -> List[ExtractedMention]:
        if not self._gaz_re:
            return []
        out = []
        for m in self._gaz_re.finditer(text):
            etype = self._gazetteer.get(m.group(0).casefold())
            if etype:
                out.append(ExtractedMention(m.group(0), m.group(0), etype, m.start(), m.end(), 0.95, "gazetteer"))
        return out

    def _titled_people(self, text: str) -> List[ExtractedMention]:
        return [
            ExtractedMention(m.group(0), m.group(1), "Person", m.start(), m.end(), 0.9)
            for m in _TITLE_RE.finditer(text)
        ]

    def _capitalised_runs(self, text: str) -> List[ExtractedMention]:
        out: List[ExtractedMention] = []
        for m in _CAP_RUN.finditer(text):
            tokens = [(t.group(0), m.start() + t.start()) for t in re.finditer(r"\S+", m.group(0))]
            while tokens and (tokens[0][0] in _STOP_CAPS or tokens[0][0] in _CONNECTORS):
                tokens.pop(0)
            if not tokens:
                continue
            start = tokens[0][1]
            end = tokens[-1][1] + len(tokens[-1][0])
            surface = text[start:end]
            for suffix in ("'s", "’s"):
                if surface.endswith(suffix):
                    surface, end = surface[: -len(suffix)], end - len(suffix)
            words = surface.split()
            if not words:
                continue
            etype, conf = self._classify(text, words, start)
            if etype:
                out.append(ExtractedMention(surface, surface, etype, start, end, conf))
        return out

    @staticmethod
    def _classify(text: str, words: List[str], start: int) -> Tuple[Optional[str], float]:
        lowered = [w.casefold().strip(".,") for w in words]
        prefix = text[max(0, start - 24):start]
        at_sentence_start = start == 0 or not prefix.strip() or prefix.rstrip(" \t")[-1:] in ".!?:\n\"“"
        prev = re.search(r"(\w+)\W*$", prefix)
        prev_word = prev.group(1).casefold() if prev else ""

        if len(words) >= 2:
            if lowered[-1] in _ORG_SUFFIX or lowered[0] in _ORG_LEAD:
                return "Organization", 0.8
            if lowered[-1] in _EVENT_SUFFIX:
                return "Event", 0.7
            if lowered[-1] in _LOC_SUFFIX:
                return "Location", 0.7
        if prev_word in _LOC_CUES and not at_sentence_start:
            return "Location", 0.55
        if len(words) == 1:
            word = words[0]
            if word.isupper() and word.isalpha():
                if len(word) < 3 or word in _ACRONYM_STOP:
                    return None, 0.0
                return "Organization", 0.4
            if len(word) < 3 or word.casefold() in _STOP_CAPS:
                return None, 0.0
            return "Thing", 0.35 if at_sentence_start else 0.45
        return "Thing", 0.5


# --------------------------------------------------------------------------- #
# Optional spaCy backend
# --------------------------------------------------------------------------- #
class SpacyEntityExtractor(EntityExtractor):
    """NER via spaCy (``pip install spacy && python -m spacy download en_core_web_sm``)."""

    _LABELS = {
        "PERSON": "Person", "ORG": "Organization", "GPE": "Location", "LOC": "Location",
        "FAC": "Location", "EVENT": "Event", "PRODUCT": "Product", "DATE": "Date",
        "WORK_OF_ART": "Document", "NORP": "Organization", "LAW": "Document",
    }

    def __init__(self, model: str = "en_core_web_sm", confidence: float = 0.8) -> None:
        try:
            import spacy  # type: ignore
        except ImportError as exc:
            raise ImportError("SpacyEntityExtractor requires `pip install spacy`") from exc
        self._nlp = spacy.load(model)
        self._confidence = confidence

    def extract(self, text: str) -> List[ExtractedMention]:
        doc = self._nlp(text)
        spans = split_sentences(text)
        out = []
        for ent in doc.ents:
            etype = self._LABELS.get(ent.label_)
            if etype:
                m = ExtractedMention(ent.text, ent.text.strip(), etype, ent.start_char,
                                     ent.end_char, self._confidence, "spacy")
                m.sentence_index = sentence_index_for(spans, m.start)
                out.append(m)
        return resolve_overlaps(out)


class CompositeEntityExtractor(EntityExtractor):
    """Run several extractors and merge their results (longest span wins)."""

    def __init__(self, extractors: Sequence[EntityExtractor]) -> None:
        if not extractors:
            raise ValueError("At least one extractor is required")
        self.extractors = list(extractors)

    def extract(self, text: str) -> List[ExtractedMention]:
        combined: List[ExtractedMention] = []
        for extractor in self.extractors:
            try:
                combined.extend(extractor.extract(text))
            except Exception:
                logger.exception("Extractor %s failed", type(extractor).__name__)
        return resolve_overlaps(combined)
