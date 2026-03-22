"""
08_rag_pipeline.py — RAG Pipeline for NOC Intelligence
=======================================================
Telco MLOps Reference Architecture — Part 2

Implements the TelcoRAGPipeline described in §8.2: a five-stage
retrieval-augmented generation pipeline for translating machine-readable
anomaly outputs into NOC-actionable narratives.

Five-stage pipeline:
  1. Document ingestion — chunk and embed telco documents
  2. Index construction — build a vector store (Qdrant in-memory)
  3. Query construction — build retrieval queries from alert cards
  4. Retrieval — hybrid dense vector search with metadata filtering
  5. Faithfulness evaluation — verify generated responses against sources

Governance gates:
  FaithfulnessMetric ≥ 0.85 (fraction of claims supported by context)
  ContextualPrecision ≥ 0.80 (fraction of retrieved passages that contribute)
  TeleQnA accuracy ≥ 0.70 (domain Q&A correctness)

Inputs:
  - Sample 3GPP-like documents (generated internally for demonstration)
  - Sample alert cards (generated from Script 01 anomaly patterns)

Outputs:
  - artifacts/rag/index_metadata.json
  - artifacts/rag/evaluation_report.json
  - artifacts/rag/sample_narrations.json

Usage:
  python 08_rag_pipeline.py

Prerequisites:
  pip install sentence-transformers qdrant-client numpy pandas

Optional (for LLM generation):
  pip install openai   # or use Anthropic API

Coursebook cross-reference:
  Ch. 35  — Retrieval-Augmented Generation (chunking, embeddings, retrieval)
  Ch. 33  — Transformers (attention, tokenisation, embedding models)
  Ch. 52  — System Design for ML (serving pipelines, evaluation gates)

Part 2 architecture notes (see §8.2):
  - Production deployment uses BAAI/bge-m3 embeddings + Qdrant + vLLM.
  - This script uses sentence-transformers for embeddings and Qdrant
    in-memory mode — no external server required.
  - LLM generation is stubbed with a template-based approach; operators
    should substitute their chosen LLM (Llama 3.1 8B, Nemotron 30B,
    or a cloud API) for production use.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_pipeline")

# ── Constants ──────────────────────────────────────────────────────────────

CHUNK_SIZE = 512          # Tokens per chunk (approximate by chars / 4)
CHUNK_OVERLAP = 64        # Overlap between chunks
EMBEDDING_DIM = 384       # all-MiniLM-L6-v2 dimension (demo/companion code)
# PRODUCTION NOTE: The whitepaper (§7.4, §8.2) recommends BAAI/bge-m3 for
# production deployment (768-dim, multilingual, better retrieval on telco
# documents).  This companion code uses all-MiniLM-L6-v2 (384-dim) as a
# lightweight demo fallback that runs without GPU.  When deploying to
# production, change the model name in embed_chunks() and update
# EMBEDDING_DIM to 768.  The Qdrant collection must be recreated when
# switching embedding models — vector dimensions are fixed at creation.
TOP_K = 5                 # Number of retrieved passages
FAITHFULNESS_GATE = 0.85
PRECISION_GATE = 0.80
RANDOM_SEED = 42

# ── Paths ──────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
RAG_DIR = ARTIFACTS_DIR / "rag"


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class Document:
    """A source document for the RAG corpus."""
    doc_id: str
    title: str
    source: str           # e.g., "3GPP TS 28.552", "Operator Runbook"
    content: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A chunked passage from a document."""
    chunk_id: str
    doc_id: str
    text: str
    source: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class AlertCard:
    """A structured alert card from the RAN anomaly detection pipeline.

    The first seven fields come from the Part 1 alert card.  The three
    ``root_cause_*`` fields are populated by the Layer 5 GNN root cause
    attribution service (§8.1) when available — they are ``None`` during
    early deployment stages (Stages 1–2) when the GNN is in shadow mode.
    """
    alert_id: str
    cell_id: str
    timestamp: str
    anomaly_score: float
    shap_top_features: Dict[str, float]
    peer_group_delta: Dict[str, float]
    severity: str  # "warning" | "major" | "critical"
    # Layer 5 GNN root cause attribution (optional — None when GNN is in
    # shadow mode or has not yet passed the governance gate; see §8.1)
    root_cause_type: Optional[str] = None      # e.g., "backhaul", "interference", "tilt"
    root_cause_node_id: Optional[str] = None   # GNN-attributed causal node
    root_cause_confidence: Optional[float] = None  # GNN confidence score [0, 1]


@dataclass
class Narration:
    """A generated NOC narration for an alert."""
    alert_id: str
    narrative: str
    cited_sources: List[str]
    recommended_action: str
    confidence: float
    faithfulness_score: float


# ── Stage 1: Document ingestion ────────────────────────────────────────────

def create_sample_corpus() -> List[Document]:
    """
    Create a sample telco document corpus for demonstration.

    Production deployment would ingest:
    - 3GPP TS 28.552 (PM measurements)
    - 3GPP TS 28.622 (NRM)
    - O-RAN WG3 E2SM-KPM specifications
    - Operator-specific runbooks and MoPs
    - Historical incident reports (PII-scrubbed)
    """
    documents = [
        Document(
            doc_id="ts28552_cqi",
            title="TS 28.552 — CQI Distribution Measurement",
            source="3GPP TS 28.552 v18.6.0",
            content=(
                "The CQI distribution measurement DRB.UECqiDistr provides the "
                "distribution of Channel Quality Indicator values reported by UEs "
                "across 16 bins (Bin0 through Bin15). Each bin corresponds to a "
                "CQI index as defined in TS 38.214 Table 5.2.2.1-2. The weighted "
                "mean CQI is computed as sum(bin_index * count) / total_count. "
                "A sustained drop in mean CQI below the cell's historical baseline "
                "(typically below CQI 7 for macro cells) indicates RF degradation "
                "that may be caused by interference, antenna misalignment, or "
                "environmental obstruction. Common root causes: electrical tilt "
                "drift (gradual, affects CQI and RSRP together), new physical "
                "obstruction (sudden, affects RSRP more than CQI), and cross-cell "
                "interference from neighbour parameter changes."
            ),
            metadata={"category": "pm_measurement", "kpis": ["avg_cqi"]},
        ),
        Document(
            doc_id="ts28552_ho",
            title="TS 28.552 — Handover Measurements",
            source="3GPP TS 28.552 v18.6.0",
            content=(
                "Handover success rate is derived from HO.ExeSucc / HO.ExeAtt. "
                "A sustained drop in handover success rate below 95% warrants "
                "investigation. Common causes include: neighbour relation table "
                "misconfiguration (missing or stale NRT entries), coverage gaps "
                "at cell boundaries, and timing advance threshold misalignment. "
                "For inter-frequency handovers, verify that the measurement gap "
                "configuration allows sufficient time for the UE to measure the "
                "target frequency. ANR (Automatic Neighbour Relation) function "
                "status should be checked — if ANR is disabled or malfunctioning, "
                "new physical neighbours may not appear in the NRT, causing "
                "handover failures at cell boundaries."
            ),
            metadata={"category": "pm_measurement", "kpis": ["handover_success_rate"]},
        ),
        Document(
            doc_id="runbook_042",
            title="Runbook RAN-042 — Antenna Tilt Drift Investigation",
            source="Operator Runbook v3.2",
            content=(
                "Procedure for investigating suspected antenna electrical tilt "
                "drift. Trigger: CQI decline pattern with concurrent RSRP stability "
                "(CQI drops while RSRP remains within normal range suggests tilt "
                "issue rather than path loss increase). Step 1: Verify current "
                "electrical tilt setting via RET (Remote Electrical Tilt) controller "
                "readback. Compare against planned tilt value in the RF planning "
                "database. Step 2: If tilt has drifted >1 degree from planned value, "
                "schedule RET recalibration. Step 3: If tilt matches planned value "
                "but CQI is degraded, check for new physical obstructions or "
                "vegetation growth affecting the antenna pattern. Step 4: Document "
                "findings in incident ticket and update cell baseline if tilt "
                "correction resolves the CQI anomaly."
            ),
            metadata={"category": "runbook", "kpis": ["avg_cqi"], "runbook_id": "RAN-042"},
        ),
        Document(
            doc_id="runbook_078",
            title="Runbook RAN-078 — Backhaul Capacity Investigation",
            source="Operator Runbook v3.2",
            content=(
                "Procedure for investigating suspected backhaul capacity exhaustion. "
                "Trigger: DL throughput degradation affecting multiple co-sited cells "
                "simultaneously, with PRB usage normal (excludes radio-side congestion). "
                "Step 1: Check transport link utilisation via NMS dashboard. If link "
                "utilisation exceeds 85% during peak hours, the backhaul is the "
                "bottleneck. Step 2: Verify QoS marking configuration — mismarked "
                "traffic can cause priority inversion. Step 3: If utilisation is "
                "normal, check for packet loss on the transport link (>0.1% loss "
                "indicates potential hardware fault). Step 4: For confirmed backhaul "
                "congestion, raise a capacity planning ticket for link upgrade. "
                "Interim mitigation: enable DL traffic shaping on the affected cells."
            ),
            metadata={"category": "runbook", "kpis": ["dl_throughput_mbps"], "runbook_id": "RAN-078"},
        ),
        Document(
            doc_id="ts28552_prb",
            title="TS 28.552 — PRB Usage Measurement",
            source="3GPP TS 28.552 v18.6.0",
            content=(
                "Physical Resource Block (PRB) usage rate RRU.PrbUsedDl measures the "
                "mean fraction of downlink PRBs scheduled per reporting interval. "
                "Sustained PRB usage above 80% indicates cell congestion — the "
                "scheduler has limited capacity to absorb traffic spikes. When PRB "
                "usage is high but throughput is low, the cell is congested and "
                "users are experiencing degraded quality. When PRB usage is low but "
                "throughput is also low, the issue is likely radio quality (CQI) "
                "rather than capacity. This distinction is critical for root cause "
                "determination: the GNN root cause layer uses the PRB-throughput "
                "ratio as an edge feature to distinguish congestion-driven from "
                "quality-driven degradation."
            ),
            metadata={"category": "pm_measurement", "kpis": ["dl_prb_usage_rate"]},
        ),
        Document(
            doc_id="incident_template",
            title="NOC Incident Classification Guide",
            source="Operator SOC Handbook v2.1",
            content=(
                "Incident classification for RAN anomalies follows a five-category "
                "taxonomy: (1) Radio — affects RF performance at a single cell "
                "(CQI, RSRP, SINR degradation); (2) Backhaul — affects throughput "
                "at multiple co-sited cells via shared transport; (3) Configuration "
                "— parameter mismatch causing performance deviation from planned "
                "values; (4) Interference — cross-cell or external interference "
                "affecting coverage or quality; (5) Load — traffic volume exceeding "
                "cell capacity (PRB exhaustion). Multi-cell correlated incidents "
                "most commonly fall into categories 2 (backhaul) and 3 (configuration), "
                "and should be escalated to Tier 2 for root cause analysis if "
                "affecting more than 3 cells simultaneously."
            ),
            metadata={"category": "operations_guide"},
        ),
    ]

    logger.info("Created sample corpus: %d documents", len(documents))
    return documents


# ── Stage 1b: Chunking ─────────────────────────────────────────────────────

def chunk_documents(documents: List[Document]) -> List[Chunk]:
    """
    Split documents into overlapping chunks for embedding.

    Uses character-based splitting at ~512 tokens (≈2048 chars).
    Production: use a token-aware splitter (e.g., tiktoken).
    """
    chunks = []
    char_size = CHUNK_SIZE * 4  # Approximate chars per chunk
    char_overlap = CHUNK_OVERLAP * 4

    for doc in documents:
        text = doc.content
        if len(text) <= char_size:
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_0",
                doc_id=doc.doc_id,
                text=text,
                source=doc.source,
                metadata=doc.metadata,
            ))
        else:
            start = 0
            idx = 0
            while start < len(text):
                end = min(start + char_size, len(text))
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_{idx}",
                    doc_id=doc.doc_id,
                    text=text[start:end],
                    source=doc.source,
                    metadata=doc.metadata,
                ))
                start = end - char_overlap
                idx += 1

    logger.info("Chunked %d documents into %d chunks", len(documents), len(chunks))
    return chunks


# ── Stage 2: Embedding and index construction ──────────────────────────────

def embed_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """
    Embed chunks using sentence-transformers.

    Production: use BAAI/bge-m3 for multilingual telco document retrieval.
    Demo: falls back to random embeddings if sentence-transformers unavailable.
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [c.text for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        logger.info(
            "Embedded %d chunks with all-MiniLM-L6-v2 (dim=%d)",
            len(chunks), embeddings.shape[1],
        )

    except ImportError:
        logger.warning(
            "sentence-transformers not installed. Using random embeddings for demo. "
            "Install with: pip install sentence-transformers"
        )
        rng = np.random.RandomState(RANDOM_SEED)
        for chunk in chunks:
            emb = rng.randn(EMBEDDING_DIM).astype(np.float32)
            chunk.embedding = emb / np.linalg.norm(emb)

    return chunks


def build_index(chunks: List[Chunk]) -> object:
    """
    Build a Qdrant vector index from embedded chunks.

    Uses Qdrant in-memory mode — no external server required.
    Falls back to a simple numpy-based index if qdrant-client is unavailable.
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance, PointStruct

        client = QdrantClient(":memory:")
        collection_name = "telco_rag"

        dim = len(chunks[0].embedding) if chunks[0].embedding is not None else EMBEDDING_DIM

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=i,
                vector=chunk.embedding.tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "source": chunk.source,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                },
            )
            for i, chunk in enumerate(chunks)
            if chunk.embedding is not None
        ]

        client.upsert(collection_name=collection_name, points=points)
        logger.info("Built Qdrant index: %d vectors in collection '%s'", len(points), collection_name)
        return {"type": "qdrant", "client": client, "collection": collection_name}

    except ImportError:
        logger.warning(
            "qdrant-client not installed. Using numpy-based index for demo. "
            "Install with: pip install qdrant-client"
        )
        embeddings = np.array([c.embedding for c in chunks if c.embedding is not None])
        return {"type": "numpy", "embeddings": embeddings, "chunks": chunks}


# ── Stage 3: Query construction ────────────────────────────────────────────

def create_sample_alerts() -> List[AlertCard]:
    """Generate sample alert cards for demonstration.

    Two of three alerts include Layer 5 GNN root cause attribution to
    demonstrate the cross-layer integration described in §8.2.  The third
    (ALERT-003) has no root cause — simulating early deployment when the
    GNN is in shadow mode.
    """
    return [
        AlertCard(
            alert_id="ALERT-001",
            cell_id="CELL_0042",
            timestamp="2024-06-15T14:30:00Z",
            anomaly_score=0.87,
            shap_top_features={
                "peer_zscore_avg_cqi": -3.4,
                "peer_zscore_dl_throughput_mbps": -2.1,
                "rolling_24h_avg_cqi_std": 1.8,
            },
            peer_group_delta={"avg_cqi": -2.9, "dl_throughput_mbps": -15.3},
            severity="major",
            # Layer 5 GNN: thermal tilt event at co-sited antenna
            root_cause_type="tilt",
            root_cause_node_id="CELL_0042",
            root_cause_confidence=0.81,
        ),
        AlertCard(
            alert_id="ALERT-002",
            cell_id="CELL_0117",
            timestamp="2024-06-15T15:00:00Z",
            anomaly_score=0.92,
            shap_top_features={
                "peer_zscore_handover_success_rate": -4.1,
                "rolling_4h_handover_success_rate_mean": -0.08,
                "peer_zscore_avg_cqi": -1.2,
            },
            peer_group_delta={"handover_success_rate": -8.5, "avg_cqi": -1.1},
            severity="critical",
            # Layer 5 GNN: NRT misconfiguration on neighbouring cell
            root_cause_type="configuration",
            root_cause_node_id="CELL_0118",
            root_cause_confidence=0.74,
        ),
        AlertCard(
            alert_id="ALERT-003",
            cell_id="CELL_0089",
            timestamp="2024-06-15T16:15:00Z",
            anomaly_score=0.78,
            shap_top_features={
                "peer_zscore_dl_prb_usage_rate": 3.2,
                "peer_zscore_dl_throughput_mbps": -2.8,
                "rolling_4h_dl_prb_usage_rate_mean": 0.12,
            },
            peer_group_delta={"dl_prb_usage_rate": 18.5, "dl_throughput_mbps": -22.1},
            severity="major",
            # No Layer 5 root cause — GNN in shadow mode for this cluster
        ),
    ]


def build_retrieval_query(alert: AlertCard) -> str:
    """
    Build a retrieval query from an alert card's structured fields.

    Combines alarm type, top SHAP features, severity, and — when available —
    the Layer 5 GNN root cause determination to construct a query that
    retrieves relevant 3GPP references and runbook procedures.  This is the
    Layer 5 → Layer 6 integration point described in §8.2: the GNN's causal
    hypothesis steers the RAG pipeline toward the correct remediation domain.
    """
    # Identify the dominant degradation pattern from SHAP
    top_feature = max(alert.shap_top_features.items(), key=lambda x: abs(x[1]))
    feature_name = top_feature[0].replace("peer_zscore_", "").replace("rolling_", "")

    # Map features to query terms
    kpi_query_map = {
        "avg_cqi": "CQI degradation antenna tilt interference root cause",
        "dl_throughput_mbps": "throughput degradation backhaul congestion capacity",
        "handover_success_rate": "handover failure neighbour relation NRT configuration",
        "dl_prb_usage_rate": "PRB congestion cell capacity load balancing",
        "rrc_conn_setup_success_rate": "RRC setup failure coverage gap",
    }

    base_query = None
    for kpi, query_terms in kpi_query_map.items():
        if kpi in feature_name:
            base_query = f"{query_terms} {alert.severity} anomaly investigation"
            break

    if base_query is None:
        base_query = f"RAN anomaly investigation {feature_name} {alert.severity}"

    # Layer 5 → Layer 6 integration: append GNN root cause context when
    # available.  This steers retrieval toward the correct remediation domain
    # (e.g., "backhaul" root cause retrieves transport-layer runbooks rather
    # than radio-layer procedures).  When root cause is unavailable (GNN in
    # shadow mode, Stage 1–2), the query falls back to SHAP-only terms above.
    rc_type = alert.root_cause_type
    rc_conf = alert.root_cause_confidence
    # Gate on GNN confidence: §12 specifies 0.72 as the circuit breaker
    # threshold for observe → plan transition.  Low-confidence attributions
    # should not steer retrieval — they may produce misleading narrations
    # that overstate certainty.
    if rc_type and (rc_conf is None or rc_conf >= 0.72):
        # Map root cause types to retrieval-relevant domain terms
        rc_query_map = {
            "backhaul": "transport backhaul link failure shared infrastructure",
            "interference": "inter-cell interference neighbour tilt antenna",
            "tilt": "antenna electrical tilt mechanical alignment thermal",
            "configuration": "parameter misconfiguration NRT ANR neighbour relation",
            "load": "capacity exhaustion traffic redistribution load balancing",
            "power": "transmit power coverage hole uplink",
        }
        rc_terms = rc_query_map.get(rc_type, rc_type)
        base_query = f"{base_query} {rc_terms}"
        if alert.root_cause_node_id:
            base_query = f"{base_query} node {alert.root_cause_node_id}"
    elif rc_type and rc_conf is not None and rc_conf < 0.72:
        logger.info(
            "GNN root cause confidence %.2f < 0.72 threshold; "
            "falling back to SHAP-only retrieval query", rc_conf
        )

    return base_query


# ── Stage 4: Retrieval ─────────────────────────────────────────────────────

def retrieve(
    query: str,
    index: dict,
    chunks: List[Chunk],
    top_k: int = TOP_K,
) -> List[Chunk]:
    """Retrieve top-k relevant chunks for a query."""

    if index["type"] == "qdrant":
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_emb = model.encode([query], normalize_embeddings=True)[0]
        except ImportError:
            rng = np.random.RandomState(hash(query) % 2**31)
            query_emb = rng.randn(EMBEDDING_DIM).astype(np.float32)
            query_emb = query_emb / np.linalg.norm(query_emb)

        results = index["client"].search(
            collection_name=index["collection"],
            query_vector=query_emb.tolist(),
            limit=top_k,
        )

        retrieved = []
        for hit in results:
            retrieved.append(Chunk(
                chunk_id=hit.payload["chunk_id"],
                doc_id=hit.payload["doc_id"],
                text=hit.payload["text"],
                source=hit.payload["source"],
                metadata=hit.payload.get("metadata", {}),
            ))
        return retrieved

    else:  # numpy fallback
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_emb = model.encode([query], normalize_embeddings=True)[0]
        except ImportError:
            rng = np.random.RandomState(hash(query) % 2**31)
            query_emb = rng.randn(EMBEDDING_DIM).astype(np.float32)
            query_emb = query_emb / np.linalg.norm(query_emb)

        embeddings = index["embeddings"]
        scores = embeddings @ query_emb
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [index["chunks"][i] for i in top_indices]


# ── Stage 5: Generation and faithfulness evaluation ────────────────────────

def generate_narration(
    alert: AlertCard,
    retrieved_chunks: List[Chunk],
) -> Narration:
    """
    Generate a NOC narration from alert + retrieved context.

    This uses template-based generation for demonstration.
    Production: substitute with vLLM-served Llama 3.1 8B or Nemotron 30B.
    """
    # Build context from retrieved chunks
    context_sources = [c.source for c in retrieved_chunks]
    context_text = "\n".join([c.text for c in retrieved_chunks[:3]])

    # Identify dominant degradation
    top_feature = max(alert.shap_top_features.items(), key=lambda x: abs(x[1]))
    feature_name = top_feature[0].replace("peer_zscore_", "").replace("rolling_", "")

    # Template-based narration (production: LLM generation)
    kpi_narratives = {
        "avg_cqi": (
            f"Cell {alert.cell_id} shows a CQI decline pattern "
            f"({alert.peer_group_delta.get('avg_cqi', 'N/A')} vs peer group). "
            "This pattern is consistent with antenna tilt drift. "
            "Recommended action: schedule physical inspection within 48 hours "
            "per Runbook RAN-042."
        ),
        "handover_success_rate": (
            f"Cell {alert.cell_id} shows degraded handover success rate "
            f"({alert.peer_group_delta.get('handover_success_rate', 'N/A')}% delta). "
            "Check NRT completeness — missing neighbour entries are the most "
            "common cause. Verify ANR function status."
        ),
        "dl_throughput_mbps": (
            f"Cell {alert.cell_id} shows throughput degradation "
            f"({alert.peer_group_delta.get('dl_throughput_mbps', 'N/A')} Mbps delta). "
            "If co-sited cells are similarly affected, suspect backhaul constraint. "
            "Check transport link utilisation per Runbook RAN-078."
        ),
        "dl_prb_usage_rate": (
            f"Cell {alert.cell_id} shows elevated PRB usage "
            f"({alert.peer_group_delta.get('dl_prb_usage_rate', 'N/A')}% above peers) "
            "with concurrent throughput degradation — indicates cell congestion. "
            "Consider load balancing via MLB parameter adjustment."
        ),
    }

    narrative = kpi_narratives.get(
        feature_name,
        f"Cell {alert.cell_id} flagged with anomaly score {alert.anomaly_score:.2f}. "
        "Manual investigation recommended."
    )

    # NOTE: This is a SIMPLIFIED keyword-matching heuristic for demonstration.
    # Production deployment MUST use DeepEval FaithfulnessMetric with a local
    # LLM as judge — the heuristic below measures whether retrieved chunks
    # contain keywords from the feature name, NOT whether every claim in the
    # generated narration is supported by the retrieved context.  The heuristic
    # score should NOT be compared against the §8.2 production gate (0.85).
    # See whitepaper §8.2 for the production evaluation architecture.
    matched_sources = [
        c.source for c in retrieved_chunks
        if any(kw in c.text.lower() for kw in feature_name.lower().split("_"))
    ]
    faithfulness = len(matched_sources) / max(len(retrieved_chunks), 1)

    return Narration(
        alert_id=alert.alert_id,
        narrative=narrative,
        cited_sources=list(set(context_sources[:3])),
        recommended_action=narrative.split("Recommended action: ")[-1].split(".")[0]
        if "Recommended action:" in narrative
        else "Manual investigation required",
        confidence=alert.anomaly_score,
        faithfulness_score=round(faithfulness, 3),
    )


def evaluate_pipeline(narrations: List[Narration]) -> Dict:
    """
    Evaluate the RAG pipeline against governance gates.

    WARNING: This function uses a SIMPLIFIED keyword-matching heuristic,
    NOT the DeepEval FaithfulnessMetric specified in whitepaper §8.2.
    The heuristic measures source-keyword overlap, not claim-level
    faithfulness.  Scores from this function should NOT be compared
    against the §8.2 production gates (0.85 / 0.80).

    Production: use DeepEval FaithfulnessMetric and RAGAS ContextualPrecision.
    Demo: simplified metrics based on source matching.

    The returned dict includes ``"evaluation_method": "keyword_heuristic_NOT_deepeval"``
    and ``"production_ready": false`` to make this explicit in any downstream reporting.
    """
    if not narrations:
        return {"faithfulness": 0.0, "precision": 0.0, "gate_passed": False}

    avg_faithfulness = np.mean([n.faithfulness_score for n in narrations])
    avg_confidence = np.mean([n.confidence for n in narrations])
    has_sources = np.mean([1.0 if n.cited_sources else 0.0 for n in narrations])

    # Simplified contextual precision: fraction of narrations with relevant sources
    contextual_precision = has_sources

    gate_passed = bool(
        avg_faithfulness >= FAITHFULNESS_GATE
        and contextual_precision >= PRECISION_GATE
    )

    evaluation = {
        "avg_faithfulness": round(float(avg_faithfulness), 4),
        "avg_contextual_precision": round(float(contextual_precision), 4),
        "avg_confidence": round(float(avg_confidence), 4),
        "n_narrations": len(narrations),
        "faithfulness_gate": FAITHFULNESS_GATE,
        "precision_gate": PRECISION_GATE,
        "gate_passed": gate_passed,
        "evaluation_method": "keyword_heuristic_NOT_deepeval",
        "production_ready": False,
        "note": (
            "SIMPLIFIED keyword-matching heuristic for demonstration only. "
            "Production deployment MUST use DeepEval FaithfulnessMetric "
            "(claim-level LLM-as-judge) and RAGAS ContextualPrecision on a "
            "200-question TeleQnA subset. The heuristic scores above should "
            "NOT be compared against the §8.2 production gates (0.85 / 0.80). "
            "See whitepaper §8.2 for the production evaluation architecture."
        ),
    }

    logger.info(
        "Evaluation [HEURISTIC — not production]: faithfulness=%.3f (gate %.2f), "
        "precision=%.3f (gate %.2f) → %s",
        avg_faithfulness, FAITHFULNESS_GATE,
        contextual_precision, PRECISION_GATE,
        "PASSED ✓" if gate_passed else "FAILED ✗",
    )

    return evaluation


# ── Main pipeline ──────────────────────────────────────────────────────────

def main() -> Dict:
    logger.info("=" * 70)
    logger.info("RAG Pipeline for NOC Intelligence")
    logger.info("=" * 70)

    # Stage 1: Document ingestion
    documents = create_sample_corpus()
    chunks = chunk_documents(documents)

    # Stage 2: Embedding and indexing
    chunks = embed_chunks(chunks)
    index = build_index(chunks)

    # Stage 3–5: Process sample alerts
    alerts = create_sample_alerts()
    narrations = []

    for alert in alerts:
        query = build_retrieval_query(alert)
        logger.info("Alert %s → query: '%s'", alert.alert_id, query[:80])

        retrieved = retrieve(query, index, chunks)
        logger.info("  Retrieved %d chunks from: %s",
                     len(retrieved), [c.source for c in retrieved[:3]])

        narration = generate_narration(alert, retrieved)
        narrations.append(narration)

        logger.info("  Narration: %s", narration.narrative[:100])
        logger.info("  Faithfulness: %.3f  Sources: %s",
                     narration.faithfulness_score, narration.cited_sources[:3])

    # Evaluation
    evaluation = evaluate_pipeline(narrations)

    # Write outputs
    RAG_DIR.mkdir(parents=True, exist_ok=True)

    index_metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_documents": len(documents),
        "n_chunks": len(chunks),
        "embedding_model": "all-MiniLM-L6-v2 (demo) / BAAI/bge-m3 (production)",
        "vector_store": "qdrant-in-memory" if index["type"] == "qdrant" else "numpy-fallback",
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "documents": [
            {"doc_id": d.doc_id, "title": d.title, "source": d.source}
            for d in documents
        ],
    }

    with open(RAG_DIR / "index_metadata.json", "w") as f:
        json.dump(index_metadata, f, indent=2)

    with open(RAG_DIR / "evaluation_report.json", "w") as f:
        json.dump(evaluation, f, indent=2)

    with open(RAG_DIR / "sample_narrations.json", "w") as f:
        json.dump([asdict(n) for n in narrations], f, indent=2)

    logger.info("Outputs written to %s", RAG_DIR)
    logger.info("=" * 70)
    logger.info("RAG pipeline complete. Gate: %s",
                "PASSED ✓" if evaluation["gate_passed"] else "FAILED ✗")
    logger.info("=" * 70)

    return evaluation


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
