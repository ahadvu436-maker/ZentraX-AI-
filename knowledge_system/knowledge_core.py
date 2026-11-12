"""Core data model and orchestrator of the ZentraX-AI knowledge system.

This module defines the shared types (Entity, Relationship, Chunk, Provenance),
text chunking, and :class:`KnowledgeCore`, which wires extraction, relationship
mapping, validation, the graph and retrieval into a single ingestion pipeline.

Other modules import their types from here; the orchestrator only imports the
other components under ``TYPE_CHECKING`` (and lazily in ``create_default``), so
there are no circular imports.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Union

from .sources import BaseSource, RawDocument

if TYPE_CHECKING:  # pragma: no cover
    from .entity_extractor import EntityExtractor
    from .knowledge_graph import KnowledgeGraph
    from .knowledge_retrieval import KnowledgeRetriever, SearchResult
    from .knowledge_validation import KnowledgeValidator, ValidationReport
    from .ontology_manager import OntologyManager
    from .relationship_mapper import RelationshipMapper

logger = logging.getLogger(__name__)

__all__ = [
    "KnowledgeError", "ValidationFailure", "IngestionError", "RetrievalError",
    "Provenance", "Entity", "Relationship", "Chunk", "IngestionReport",
    "KnowledgeCore", "normalize_name", "make_entity_id", "make_relationship_id",
    "chunk_text", "utcnow",
]

MAX_PROVENANCE = 100


class KnowledgeError(Exception):
    """Base error for the knowledge system."""


class ValidationFailure(KnowledgeError):
    """Data failed validation."""


class IngestionError(KnowledgeError):
    """A source or document could not be ingested."""


class RetrievalError(KnowledgeError):
    """A query could not be served."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")
_EDGE_PUNCT = re.compile(r"^[\W_]+|[\W_]+$")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_name(name: str) -> str:
    """Case-fold, collapse whitespace and trim edge punctuation."""
    return _EDGE_PUNCT.sub("", _WS.sub(" ", name.strip())).casefold()


def make_entity_id(name: str, entity_type: str) -> str:
    digest = hashlib.sha1(f"{entity_type}|{normalize_name(name)}".encode("utf-8")).hexdigest()
    return f"ent_{digest[:16]}"


def make_relationship_id(source_id: str, rel_type: str, target_id: str) -> str:
    digest = hashlib.sha1(f"{source_id}|{rel_type}|{target_id}".encode("utf-8")).hexdigest()
    return f"rel_{digest[:16]}"


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into chunks of at most ``max_chars`` on paragraph/sentence edges."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    overlap = max(0, min(overlap, max_chars // 2))
    text = text.strip()
    if not text:
        return []
    pieces: List[str] = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
        if len(para) <= max_chars:
            pieces.append(para)
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            while len(sent) > max_chars:
                pieces.append(sent[:max_chars])
                sent = sent[max_chars:]
            if sent:
                pieces.append(sent)
    chunks: List[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= max_chars:
            current = f"{current}\n{piece}"
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            if " " in tail:
                tail = tail[tail.find(" ") + 1:]
            candidate = f"{tail} {piece}".strip() if tail else piece
            current = candidate if len(candidate) <= max_chars else piece
    if current:
        chunks.append(current)
    return chunks


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Provenance:
    source_type: str = "unknown"
    source_uri: str = ""
    doc_id: str = ""
    chunk_id: str = ""
    confidence: float = 1.0
    timestamp: str = field(default_factory=utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Provenance":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def _merge_provenance(target: List[Provenance], extra: List[Provenance]) -> None:
    seen = {(p.doc_id, p.chunk_id) for p in target}
    for prov in extra:
        key = (prov.doc_id, prov.chunk_id)
        if key not in seen and len(target) < MAX_PROVENANCE:
            target.append(prov)
            seen.add(key)


@dataclass
class Entity:
    id: str
    name: str
    type: str
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    provenance: List[Provenance] = field(default_factory=list)
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)

    @classmethod
    def create(cls, name: str, entity_type: str, **kwargs: Any) -> "Entity":
        return cls(id=make_entity_id(name, entity_type), name=name.strip(), type=entity_type, **kwargs)

    def all_names(self) -> List[str]:
        return [self.name, *self.aliases]

    def merge(self, other: "Entity") -> None:
        """Fold another entity's knowledge into this one."""
        known = {normalize_name(n) for n in self.all_names()}
        for name in other.all_names():
            key = normalize_name(name)
            if key and key not in known:
                self.aliases.append(name)
                known.add(key)
        for key, value in other.properties.items():
            self.properties.setdefault(key, value)
        self.confidence = max(self.confidence, other.confidence)
        _merge_provenance(self.provenance, other.provenance)
        self.updated_at = utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Entity":
        d = dict(data)
        d["provenance"] = [Provenance.from_dict(p) for p in d.get("provenance", [])]
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in fields(cls)}})


@dataclass
class Relationship:
    id: str
    source_id: str
    target_id: str
    type: str
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    provenance: List[Provenance] = field(default_factory=list)
    evidence: str = ""
    created_at: str = field(default_factory=utcnow)

    @classmethod
    def create(cls, source_id: str, rel_type: str, target_id: str, **kwargs: Any) -> "Relationship":
        return cls(
            id=make_relationship_id(source_id, rel_type, target_id),
            source_id=source_id, target_id=target_id, type=rel_type, **kwargs,
        )

    def merge(self, other: "Relationship") -> None:
        self.confidence = max(self.confidence, other.confidence)
        for key, value in other.properties.items():
            self.properties.setdefault(key, value)
        _merge_provenance(self.provenance, other.provenance)
        if not self.evidence:
            self.evidence = other.evidence

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Relationship":
        d = dict(data)
        d["provenance"] = [Provenance.from_dict(p) for p in d.get("provenance", [])]
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in fields(cls)}})


@dataclass
class Chunk:
    id: str
    doc_id: str
    index: int
    text: str
    source_uri: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    entity_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Chunk":
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}})


@dataclass
class IngestionReport:
    documents_seen: int = 0
    documents_ingested: int = 0
    documents_skipped: int = 0
    chunks_indexed: int = 0
    entities_added: int = 0
    entities_merged: int = 0
    entities_rejected: int = 0
    relationships_added: int = 0
    relationships_merged: int = 0
    relationships_rejected: int = 0
    errors: List[str] = field(default_factory=list)

    def merge(self, other: "IngestionReport") -> None:
        for f in fields(self):
            if f.name == "errors":
                self.errors.extend(other.errors)
            else:
                setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class KnowledgeCore:
    """Ingests sources into the graph + index and serves queries."""

    def __init__(
        self,
        ontology: "OntologyManager",
        graph: "KnowledgeGraph",
        extractor: "EntityExtractor",
        mapper: "RelationshipMapper",
        retriever: "KnowledgeRetriever",
        validator: "KnowledgeValidator",
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        min_mention_confidence: float = 0.3,
        keep_untyped: bool = False,
        link_documents: bool = True,
    ) -> None:
        self.ontology = ontology
        self.graph = graph
        self.extractor = extractor
        self.mapper = mapper
        self.retriever = retriever
        self.validator = validator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_mention_confidence = min_mention_confidence
        self.keep_untyped = keep_untyped
        self.link_documents = link_documents
        self._doc_hashes: Dict[str, str] = {}
        self._lock = threading.RLock()

    @classmethod
    def create_default(
        cls,
        *,
        ontology: Optional["OntologyManager"] = None,
        gazetteer: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> "KnowledgeCore":
        """Build a fully wired system with the rule-based extractor."""
        from .entity_extractor import RuleBasedEntityExtractor
        from .knowledge_graph import KnowledgeGraph
        from .knowledge_retrieval import KnowledgeRetriever
        from .knowledge_validation import KnowledgeValidator
        from .ontology_manager import OntologyManager
        from .relationship_mapper import RelationshipMapper

        ontology = ontology or OntologyManager()
        graph = KnowledgeGraph()
        return cls(
            ontology=ontology,
            graph=graph,
            extractor=RuleBasedEntityExtractor(gazetteer=gazetteer),
            mapper=RelationshipMapper(ontology),
            retriever=KnowledgeRetriever(graph),
            validator=KnowledgeValidator(ontology, graph),
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest_source(self, source: BaseSource, *, continue_on_error: bool = True) -> IngestionReport:
        report = IngestionReport()
        try:
            for doc in source.fetch():
                try:
                    report.merge(self.ingest_document(doc))
                except Exception as exc:
                    logger.exception("Failed to ingest %s", doc.uri)
                    report.errors.append(f"{doc.uri}: {exc}")
                    if not continue_on_error:
                        raise IngestionError(str(exc)) from exc
        finally:
            source.close()
        return report

    def ingest_text(self, text: str, uri: str = "inline://text", metadata: Optional[Dict[str, Any]] = None) -> IngestionReport:
        from .sources import make_doc_id

        return self.ingest_document(
            RawDocument(id=make_doc_id(uri), uri=uri, content=text, source_type="inline",
                        title=uri, metadata=metadata or {})
        )

    def ingest_document(self, doc: RawDocument) -> IngestionReport:
        report = IngestionReport(documents_seen=1)
        with self._lock:
            previous = self._doc_hashes.get(doc.id)
            if previous == doc.content_hash:
                report.documents_skipped = 1
                return report
            if previous is not None:
                self.retriever.remove_document(doc.id)  # re-ingest changed content
            base_meta = {**doc.metadata, "source_type": doc.source_type, "title": doc.title}
            doc_entities: Dict[str, float] = {}
            for idx, text in enumerate(chunk_text(doc.content, self.chunk_size, self.chunk_overlap)):
                chunk = Chunk(id=f"{doc.id}:{idx}", doc_id=doc.id, index=idx, text=text,
                              source_uri=doc.uri, metadata=dict(base_meta))
                try:
                    self._process_chunk(chunk, doc, report, doc_entities)
                except Exception as exc:
                    logger.exception("Chunk %s failed", chunk.id)
                    report.errors.append(f"{chunk.id}: {exc}")
                self.retriever.index_chunk(chunk)
                report.chunks_indexed += 1
            if self.link_documents and doc_entities:
                self._link_document(doc, doc_entities, report)
            self._doc_hashes[doc.id] = doc.content_hash
            report.documents_ingested = 1
        return report

    def _process_chunk(self, chunk: Chunk, doc: RawDocument, report: IngestionReport,
                       doc_entities: Dict[str, float]) -> None:
        from .knowledge_validation import Severity

        mentions = []
        for m in self.extractor.extract(chunk.text):
            if m.confidence < self.min_mention_confidence:
                continue
            resolved = self.ontology.resolve_entity_type(m.type)
            if resolved is None:
                continue
            m.type = resolved
            mentions.append(m)

        prov = Provenance(source_type=doc.source_type, source_uri=doc.uri,
                          doc_id=doc.id, chunk_id=chunk.id)
        # The mapper may refine untyped ("Thing") mentions from relation context.
        relationships = self.mapper.map_relationships(chunk.text, mentions, prov)
        used = {r.source_id for r in relationships} | {r.target_id for r in relationships}
        if not self.keep_untyped:
            mentions = [m for m in mentions if m.type != self.ontology.ROOT or m.entity_id in used]

        accepted: Dict[str, float] = {}
        for entity in self.extractor.mentions_to_entities(mentions, prov):
            issues = self.validator.validate_entity(entity)
            if any(i.severity == Severity.ERROR for i in issues):
                report.entities_rejected += 1
                continue
            _, created = self.graph.add_entity(entity)
            if created:
                report.entities_added += 1
            else:
                report.entities_merged += 1
            accepted[entity.id] = entity.confidence
            doc_entities[entity.id] = max(doc_entities.get(entity.id, 0.0), entity.confidence)
        chunk.entity_ids = list(accepted)

        for rel in relationships:
            if rel.source_id not in accepted or rel.target_id not in accepted:
                report.relationships_rejected += 1
                continue
            issues = self.validator.validate_relationship(rel)
            if any(i.severity == Severity.ERROR for i in issues):
                report.relationships_rejected += 1
                continue
            _, created = self.graph.add_relationship(rel)
            if created:
                report.relationships_added += 1
            else:
                report.relationships_merged += 1

    def _link_document(self, doc: RawDocument, entity_ids: Dict[str, float], report: IngestionReport) -> None:
        doc_type = self.ontology.resolve_entity_type("Document")
        if doc_type is None:
            return
        prov = Provenance(source_type=doc.source_type, source_uri=doc.uri, doc_id=doc.id)
        doc_entity = Entity.create(
            doc.uri, doc_type,
            properties={"title": doc.title, "source_type": doc.source_type, "uri": doc.uri},
            provenance=[prov],
        )
        self.graph.add_entity(doc_entity)
        for eid in entity_ids:
            rel = Relationship.create(doc_entity.id, "mentions", eid, confidence=0.9, provenance=[prov])
            if any(i.severity.value == "error" for i in self.validator.validate_relationship(rel)):
                continue
            _, created = self.graph.add_relationship(rel)
            if created:
                report.relationships_added += 1

    def remove_document(self, doc_id: str) -> int:
        """Remove a document's text from the index (graph facts are retained)."""
        with self._lock:
            self._doc_hashes.pop(doc_id, None)
            return self.retriever.remove_document(doc_id)

    # ------------------------------------------------------------------ #
    # Queries and maintenance
    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 5, expand_graph: bool = True,
               filters: Optional[Mapping[str, Any]] = None) -> List["SearchResult"]:
        return self.retriever.search(query, top_k=top_k, expand_graph=expand_graph, filters=filters)

    def build_context(self, query: str, top_k: int = 5, max_chars: int = 4000) -> str:
        return self.retriever.build_context(query, top_k=top_k, max_chars=max_chars)

    def entity_profile(self, name_or_id: str) -> Optional[Dict[str, Any]]:
        entity = self.graph.get_entity(name_or_id)
        if entity is None:
            matches = self.graph.find_entities(name_or_id)
            entity = matches[0] if matches else None
        return self.retriever.entity_profile(entity.id) if entity else None

    def validate(self) -> "ValidationReport":
        return self.validator.validate_graph()

    def merge_duplicates(self, threshold: float = 0.93) -> int:
        """Merge same-type entities with near-identical names. Returns merge count."""
        merged = 0
        with self._lock:
            for keep_id, drop_id, _score in self.validator.find_duplicate_candidates(threshold=threshold):
                if self.graph.has_entity(keep_id) and self.graph.has_entity(drop_id):
                    self.graph.merge_entities(keep_id, drop_id)
                    self.retriever.remap_entity(drop_id, keep_id)
                    merged += 1
        return merged

    def stats(self) -> Dict[str, Any]:
        return {
            "graph": self.graph.stats(),
            "chunks_indexed": len(self.retriever),
            "documents": len(self._doc_hashes),
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, directory: Union[str, Path]) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.graph.save(path / "graph.json")
            self._atomic_write(path / "chunks.json", json.dumps(self.retriever.to_dict()))
            self._atomic_write(path / "state.json", json.dumps({"doc_hashes": self._doc_hashes}))

    def load(self, directory: Union[str, Path]) -> None:
        path = Path(directory)
        with self._lock:
            self.graph.load(path / "graph.json")
            chunks = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
            self.retriever.load_dict(chunks)
            state = json.loads((path / "state.json").read_text(encoding="utf-8"))
            self._doc_hashes = dict(state.get("doc_hashes", {}))

    @staticmethod
    def _atomic_write(path: Path, payload: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
