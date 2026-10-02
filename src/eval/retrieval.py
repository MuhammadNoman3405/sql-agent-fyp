"""
retrieval.py  --  Phase 2, retrieval core
==========================================

Two jobs, both reusable as a library (import this from other scripts) and
runnable directly for a quick manual test:

  1. build_index(...)   embed every table of a database into Qdrant.
                         Two "flavors" of text can be embedded:
                           enriched = the Idea A plain-English description
                           raw      = just the table/column names (the old way)
  2. retrieve(...)      embed a question, return the top-K closest tables.

This is what turns "paste the whole schema" into "paste only what's relevant" --
the central idea Phase 2 measures.

Embeddings come from Ollama's embedding endpoint (nomic-embed-text), so nothing
new needs installing beyond `pip install qdrant-client` -- no sentence-transformers,
no separate model download infrastructure, same Ollama server Phase 1 already uses.

Qdrant: by default this talks to the Dockerized Qdrant from Phase 0
(http://localhost:6333, set up in docker-compose.yml). Pass --qdrant-memory to
use an in-process, on-disk-only collection instead, useful if you want to try
retrieval without starting Docker.

Usage (run from the repo root) -- manual smoke test
-----------------------------------------------------
  python src\\eval\\retrieval.py --dataset bird --db california_schools --query "which school has the highest SAT score"

Collections are named  <dataset>__<db_id>__level<k>__<flavor>
so clean vs degraded vs enriched-off-vs-on never collide.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

SOURCES = {
    "spider": "data/spider/spider/spider_data/spider_data/database/{db}/{db}.sqlite",
    "bird": "data/bird/bird/dev/dev_20240627/dev_databases/dev_databases/{db}/{db}.sqlite",
}


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------
def embed(text: str, model: str = EMBED_MODEL, timeout: int = 60) -> list:
    payload = {"model": model, "prompt": text}
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data.get("embedding") or data["embeddings"][0]


def ollama_preflight(model: str = EMBED_MODEL):
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=10) as r:
            names = [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        sys.exit("Cannot reach Ollama at http://localhost:11434 -- is it running?")
    if not any(n == model or n.startswith(model + ":") for n in names):
        sys.exit(f"Embedding model '{model}' not downloaded.\n  -> run:  ollama pull {model}")


# --------------------------------------------------------------------------
# Schema text builders -- what actually gets embedded
# --------------------------------------------------------------------------
def find_db(source: str, db: str) -> Path:
    default = Path(SOURCES[source].format(db=db))
    if default.exists():
        return default
    for hit in (Path("data") / source).rglob(f"{db}.sqlite"):
        if hit.parent.name == db:
            return hit
    return default


def degraded_path(source: str, db: str, level: int) -> Path:
    return Path("data") / "degraded" / source / db / f"level{level}" / f"{db}.sqlite"


def raw_table_text(table: str, col_names: list) -> str:
    """The 'old way': just names, no enrichment. This is the control group."""
    return f"{table} ({', '.join(col_names)})"


def enriched_table_text(table: str, enriched_entry: dict, col_names: list) -> str:
    desc = enriched_entry.get("description", "")
    return f"{table}: {desc} Columns: {', '.join(col_names)}"


def load_table_columns(db_path: Path) -> dict:
    """table -> [column names], read straight from the sqlite file."""
    import sqlite3
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )]
    out = {}
    for t in tables:
        out[t] = [c[1] for c in conn.execute(f'PRAGMA table_info("{t}")')]
    conn.close()
    return out


# --------------------------------------------------------------------------
# Qdrant
# --------------------------------------------------------------------------
def get_client(qdrant_url: str, memory: bool) -> QdrantClient:
    if memory:
        return QdrantClient(location=":memory:")
    return QdrantClient(url=qdrant_url)


def collection_name(dataset: str, db_id: str, level: int, flavor: str) -> str:
    return f"{dataset}__{db_id}__level{level}__{flavor}"


def build_index(client: QdrantClient, dataset: str, db_id: str, level: int, flavor: str,
                enriched_path: Path = None, embed_model: str = EMBED_MODEL, verbose: bool = True):
    """flavor: 'enriched' or 'raw'. Returns the collection name."""
    db_path = find_db(dataset, db_id) if level == 0 else degraded_path(dataset, db_id, level)
    if not db_path.exists():
        sys.exit(f"Database not found: {db_path}")
    cols = load_table_columns(db_path)

    enriched = {}
    if flavor == "enriched":
        path = enriched_path or Path("data/enriched") / dataset / db_id / f"level{level}.json"
        if not path.exists():
            sys.exit(f"No enrichment file at {path}. Run enrich_schema.py first, or use flavor='raw'.")
        enriched = json.loads(path.read_text(encoding="utf-8"))

    # embed first (so we know the real vector size from this model before creating the collection)
    texts, vecs = [], []
    for table, col_names in sorted(cols.items()):
        text = (enriched_table_text(table, enriched.get(table, {}), col_names) if flavor == "enriched"
                else raw_table_text(table, col_names))
        texts.append((table, text))
        vecs.append(embed(text, embed_model))

    name = collection_name(dataset, db_id, level, flavor)
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=len(vecs[0]), distance=qm.Distance.COSINE),
    )

    points = [
        qm.PointStruct(id=i, vector=vec, payload={"table": table, "text": text})
        for i, ((table, text), vec) in enumerate(zip(texts, vecs))
    ]
    if verbose:
        for table, _ in texts:
            print(f"    indexed: {table}")
    client.upsert(collection_name=name, points=points)
    return name


def retrieve(client: QdrantClient, collection: str, question: str, top_k: int,
            embed_model: str = EMBED_MODEL) -> list:
    """Returns [(table_name, score), ...] sorted best-first."""
    qvec = embed(question, embed_model)
    result = client.query_points(collection_name=collection, query=qvec, limit=top_k)
    return [(h.payload["table"], round(h.score, 4)) for h in result.points]


# --------------------------------------------------------------------------
# Manual smoke test
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Index one database and try a retrieval query.")
    ap.add_argument("--dataset", required=True, choices=["bird", "spider"])
    ap.add_argument("--db", required=True)
    ap.add_argument("--level", type=int, default=0, choices=[0, 1, 2, 3])
    ap.add_argument("--flavor", default="enriched", choices=["enriched", "raw"])
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--qdrant-memory", action="store_true")
    args = ap.parse_args()

    ollama_preflight()
    client = get_client(args.qdrant_url, args.qdrant_memory)
    print(f"Indexing {args.db} (level {args.level}, flavor={args.flavor}) ...")
    coll = build_index(client, args.dataset, args.db, args.level, args.flavor)
    print(f"\nQuery: {args.query!r}\nTop {args.top_k} tables:")
    for table, score in retrieve(client, coll, args.query, args.top_k):
        print(f"    {score:.4f}  {table}")


if __name__ == "__main__":
    main()
