"""Infer typed relationships between entity mentions within sentences."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern, Sequence, Tuple

from .entity_extractor import ExtractedMention, sentence_index_for, split_sentences
from .knowledge_core import Provenance, Relationship, make_relationship_id, normalize_name
from .ontology_manager import OntologyManager

__all__ = ["RelationPattern", "RelationshipMapper"]

_LEAD = (r"^[\s,;:()\-–—]*(?:(?:who|which|that|and|also|then|has|have|had|was|were|is|are|"
         r"been|being|now|later|currently|recently|once|formerly|the|a|an)\s+)*")
_TAIL = r"[\s,;:()]*$"
_ROLE = (r"(?:[\w\-]+\s+){0,2}(?:engineer|scientist|researcher|developer|manager|director|ceo|cto|"
         r"cfo|coo|president|professor|designer|analyst|architect|lead|head|chairman|officer|employee)")


def _rx(phrases: str) -> Pattern[str]:
    return re.compile(_LEAD + r"(?:" + phrases + r")" + _TAIL, re.IGNORECASE)


@dataclass(frozen=True)
class RelationPattern:
    """If the text between two mentions matches ``regex``, A <relation> B.

    ``reverse=True`` means the *second* mention is the subject.
    """

    relation: str
    regex: Pattern[str]
    reverse: bool = False
    confidence: float = 0.75


def default_patterns() -> List[RelationPattern]:
    """Ordered by specificity; the first ontology-valid match wins."""
    P = RelationPattern
    return [
        P("created_by", _rx(r"(?:created|developed|built|designed|authored|written|invented|made|released|published)\s+by")),
        P("created_by", _rx(r"created|developed|built|designed|authored|wrote|invented|released|published"), reverse=True),
        P("founded", _rx(r"(?:co-?)?(?:founded|established|started|created|launched)\s+by"), reverse=True),
        P("founded", _rx(r"(?:co-?)?found(?:ed|s)|established|started|created|launched|(?:co-?)?founder\s+of")),
        P("acquired", _rx(r"(?:acquired|bought|purchased|taken\s+over)\s+by"), reverse=True),
        P("acquired", _rx(r"acquired|bought|purchased|took\s+over|acquires")),
        P("works_for", _rx(r"works?\s+(?:for|at)|worked\s+(?:for|at)|working\s+(?:for|at)|joined|"
                           r"employed\s+(?:by|at)|hired\s+by|" + _ROLE + r"\s+(?:at|of|for)")),
        P("collaborates_with", _rx(r"collaborat(?:es|ed|ing)\s+with|partner(?:s|ed|ing)?\s+with|"
                                   r"works?\s+with|worked\s+with|teamed\s+up\s+with|allied\s+with")),
        P("part_of", _rx(r"part\s+of|belongs?\s+to|subsidiary\s+of|division\s+of|branch\s+of|"
                         r"member\s+of|component\s+of|module\s+of|included\s+in")),
        P("uses", _rx(r"(?:used|utili[sz]ed|leveraged|adopted)\s+by"), reverse=True),
        P("uses", _rx(r"uses|used|using|utili[sz]es|leverages|built\s+(?:on|with|using)|runs\s+on|"
                      r"depends\s+on|relies\s+on|powered\s+by|implemented\s+(?:in|with)|written\s+in")),
        P("occurred_on", _rx(r"on|took\s+place\s+on|taking\s+place\s+on|occurred\s+on|held\s+on|"
                             r"scheduled\s+for|happened\s+on|begins?\s+on|started\s+on"), confidence=0.7),
        P("located_in", _rx(r"located\s+in|located\s+at|based\s+in|headquartered\s+in|situated\s+in|"
                            r"lives?\s+in|lived\s+in|born\s+in|resides?\s+in|operates\s+in|held\s+in|"
                            r"takes?\s+place\s+in|took\s+place\s+in|moved\s+to|relocated\s+to")),
        P("located_in", _rx(r"in|near|,"), confidence=0.5),
    ]


class RelationshipMapper:
    """Pattern-based relation extraction with ontology validation.

    Side effects: mentions typed ``Thing`` are refined in place when a relation
    implies a single possible class (e.g. the subject of ``works_for`` is a
    ``Person``). Types are then propagated to other mentions of the same name.
    """

    def __init__(
        self,
        ontology: OntologyManager,
        patterns: Optional[Sequence[RelationPattern]] = None,
        *,
        max_gap_chars: int = 100,
        max_lookahead: int = 4,
        cooccurrence: bool = True,
        cooccurrence_confidence: float = 0.3,
    ) -> None:
        self.ontology = ontology
        self.patterns = list(patterns) if patterns is not None else default_patterns()
        self.max_gap_chars = max_gap_chars
        self.max_lookahead = max_lookahead
        self.cooccurrence = cooccurrence
        self.cooccurrence_confidence = cooccurrence_confidence

    def add_pattern(self, pattern: RelationPattern) -> None:
        self.patterns.insert(0, pattern)

    # ------------------------------------------------------------------ #
    def map_relationships(self, text: str, mentions: List[ExtractedMention],
                          provenance: Provenance) -> List[Relationship]:
        spans = split_sentences(text)
        by_sentence: Dict[int, List[ExtractedMention]] = {}
        for m in sorted(mentions, key=lambda m: m.start):
            by_sentence.setdefault(sentence_index_for(spans, m.start), []).append(m)

        found: List[Tuple[ExtractedMention, ExtractedMention, str, float, str]] = []
        for idx, group in by_sentence.items():
            s_start, s_end = spans[idx]
            evidence = text[s_start:s_end].strip()[:300]
            for i, left in enumerate(group):
                for right in group[i + 1: i + 1 + self.max_lookahead]:
                    gap = text[left.end:right.start]
                    if len(gap) > self.max_gap_chars:
                        break
                    hit = self._match(left, right, gap)
                    if hit is None and self.cooccurrence and i + 1 < len(group) and right is group[i + 1] \
                            and not re.search(r"[.!?;]", gap):
                        hit = self._cooccurrence(left, right)
                    if hit is not None:
                        subj, obj, relation, conf = hit
                        found.append((subj, obj, relation, conf, evidence))

        self._propagate_types(mentions)

        results: Dict[str, Relationship] = {}
        for subj, obj, relation, conf, evidence in found:
            if subj.entity_id == obj.entity_id:
                continue
            src, dst = subj.entity_id, obj.entity_id
            if self.ontology.is_symmetric(relation) and dst < src:
                src, dst = dst, src
            rel_id = make_relationship_id(src, relation, dst)
            existing = results.get(rel_id)
            if existing is None:
                results[rel_id] = Relationship(
                    id=rel_id, source_id=src, target_id=dst, type=relation,
                    confidence=round(conf * min(subj.confidence, obj.confidence) ** 0.5, 3),
                    provenance=[Provenance(**{**provenance.__dict__, "confidence": conf})],
                    evidence=evidence,
                )
            else:
                existing.confidence = max(existing.confidence, conf)
        return list(results.values())

    # ------------------------------------------------------------------ #
    def _match(self, left: ExtractedMention, right: ExtractedMention,
               gap: str) -> Optional[Tuple[ExtractedMention, ExtractedMention, str, float]]:
        for pat in self.patterns:
            if not pat.regex.search(gap):
                continue
            subj, obj = (right, left) if pat.reverse else (left, right)
            relation = self.ontology.resolve_relation(pat.relation)
            if relation is None:
                continue
            if self._try_refine(subj, obj, relation):
                return subj, obj, relation, pat.confidence
        return None

    def _cooccurrence(self, left: ExtractedMention,
                      right: ExtractedMention) -> Optional[Tuple[ExtractedMention, ExtractedMention, str, float]]:
        relation = self.ontology.resolve_relation("related_to")
        skip = {self.ontology.ROOT, "Date", "Email", "URL", "Identifier"}
        if relation is None or left.type in skip or right.type in skip:
            return None
        ok, _ = self.ontology.validate_relationship(relation, left.type, right.type)
        return (left, right, relation, self.cooccurrence_confidence) if ok else None

    def _try_refine(self, subj: ExtractedMention, obj: ExtractedMention, relation: str) -> bool:
        """Validate against the ontology, refining untyped endpoints first."""
        root = self.ontology.ROOT
        new_subj = subj.type
        new_obj = obj.type
        if subj.type == root:
            new_subj = self.ontology.single_class(relation, "domain") or root
        if obj.type == root:
            new_obj = self.ontology.single_class(relation, "range") or root
        ok, _ = self.ontology.validate_relationship(relation, new_subj, new_obj)
        if not ok:
            return False
        subj.type, obj.type = new_subj, new_obj
        return True

    def _propagate_types(self, mentions: Sequence[ExtractedMention]) -> None:
        root = self.ontology.ROOT
        known: Dict[str, str] = {}
        for m in mentions:
            if m.type != root:
                known.setdefault(normalize_name(m.canonical), m.type)
        for m in mentions:
            if m.type == root:
                m.type = known.get(normalize_name(m.canonical), root)
