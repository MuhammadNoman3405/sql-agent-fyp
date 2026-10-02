"""
enrich_schema.py  --  Phase 2, Idea A: offline schema enrichment
==================================================================

For every table in a database, this asks a local LLM to look at the table's
name, its columns, and a few real sample rows, and write a short plain-English
description of what the table actually holds.

Why this matters
-----------------
A retrieval system that searches on raw names like "frpm" or "satscores"
can't tell "which table has school enrollment numbers?" -- those words never
appear in the schema. But an enriched description ("frpm: free/reduced-price
meal counts and enrollment per school...") DOES contain the words a question
would use. This is the retrieval signal Phase 2's index is built on.

This step is OFFLINE and runs ONCE per database (not per question) -- the
descriptions are cached to disk and reused by every later experiment.

Usage (run from the repo root)
-------------------------------
  # one database
  python src\\eval\\enrich_schema.py --dataset bird --db california_schools

  # every database that has a clean copy on disk
  python src\\eval\\enrich_schema.py --dataset bird --all

  # a degraded (messy) copy instead of the clean one
  python src\\eval\\enrich_schema.py --dataset bird --db california_schools --level 2

Output
------
  data/enriched/<dataset>/<db_id>/level<k>.json
     { "<table>": {"description": "...", "num_columns": n, "num_sample_rows": n}, ... }

Only the Python standard library is used (plus whatever baseline.py already needs
for Ollama -- same pattern, nothing new to pip install).
"""

import argparse
import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_BASE = "http://localhost:11434"
OUTPUT_ROOT = Path("data/enriched")

SYSTEM_PROMPT = (
    "You are a senior data analyst documenting an unfamiliar production database. "
    "You will be shown one table's name, its columns, and a few real sample rows. "
    "Write ONE or TWO plain-English sentences describing what real-world information "
    "this table stores and what each row represents. "
    "Do not describe the column names themselves, describe the MEANING. "
    "Do not guess wildly beyond what the data shows. Output only the description, "
    "no preamble, no markdown, no quotes."
)


# --------------------------------------------------------------------------
# Finding databases (same auto-discovery approach as degrade_schema.py)
# --------------------------------------------------------------------------
SOURCES = {
    "spider": "data/spider/spider/spider_data/spider_data/database/{db}/{db}.sqlite",
    "bird": "data/bird/bird/dev/dev_20240627/dev_databases/dev_databases/{db}/{db}.sqlite",
}


def find_db(source: str, db: str) -> Path:
    default = Path(SOURCES[source].format(db=db))
    if default.exists():
        return default
    root = Path("data") / source
    for hit in root.rglob(f"{db}.sqlite"):
        if hit.parent.name == db:
            return hit
    return default


def degraded_path(source: str, db: str, level: int) -> Path:
    return Path("data") / "degraded" / source / db / f"level{level}" / f"{db}.sqlite"


def list_all_dbs(source: str):
    root = Path("data") / source
    found = {}
    for p in root.rglob("*.sqlite"):
        if p.parent.name == p.stem:
            found.setdefault(p.stem, p)
    return sorted(found)


# --------------------------------------------------------------------------
# Reading the schema + sample rows
# --------------------------------------------------------------------------
def q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def read_tables(db_path: Path, sample_rows: int = 3):
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    tables = [
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    info = {}
    for t in tables:
        cols = conn.execute(f"PRAGMA table_info({q(t)})").fetchall()
        col_names = [c[1] for c in cols]
        try:
            rows = conn.execute(f"SELECT * FROM {q(t)} LIMIT {sample_rows}").fetchall()
        except sqlite3.Error:
            rows = []
        info[t] = {"columns": cols, "col_names": col_names, "sample_rows": rows}
    conn.close()
    return info


def format_table_for_prompt(table_name: str, info: dict) -> str:
    col_lines = "\n".join(f"  - {c[1]} ({c[2] or 'TEXT'})" for c in info["columns"])
    if info["sample_rows"]:
        header = " | ".join(info["col_names"])
        sample_lines = "\n".join(
            " | ".join("NULL" if v is None else str(v)[:40] for v in row)
            for row in info["sample_rows"]
        )
        sample = f"{header}\n{sample_lines}"
    else:
        sample = "(table is empty)"
    return (
        f"Table name: {table_name}\n"
        f"Columns:\n{col_lines}\n\n"
        f"Sample rows:\n{sample}"
    )


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------
def ollama_preflight(model: str):
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=10) as r:
            names = [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        sys.exit(
            "Cannot reach Ollama at http://localhost:11434\n"
            "  -> Is Ollama installed and running? (open the Ollama app, or run: ollama serve)"
        )
    if not any(n == model or n.startswith(model + ":") for n in names):
        sys.exit(f"Model '{model}' is not downloaded yet.\n  -> run:  ollama pull {model}")


def ask_ollama(model: str, user_prompt: str, timeout: int = 300) -> str:
    payload = {
        "model": model, "stream": False,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def clean_description(text: str) -> str:
    text = text.strip().strip('"').strip()
    text = re.sub(r"^(description|summary)\s*:\s*", "", text, flags=re.I)
    return text


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def enrich_database(source: str, db_id: str, db_path: Path, level: int, model: str,
                     sample_rows: int, mock: bool, force: bool):
    out_path = OUTPUT_ROOT / source / db_id / f"level{level}.json"
    existing = {}
    if out_path.exists() and not force:
        existing = json.loads(out_path.read_text(encoding="utf-8"))

    tables = read_tables(db_path, sample_rows)
    result = dict(existing)
    new_count = 0
    for t, info in tables.items():
        if t in result and not force:
            continue
        prompt = format_table_for_prompt(t, info)
        if mock:
            desc = f"[mock] table '{t}' with {len(info['columns'])} columns."
        else:
            raw = ask_ollama(model, prompt)
            desc = clean_description(raw)
        result[t] = {
            "description": desc,
            "num_columns": len(info["columns"]),
            "num_sample_rows": len(info["sample_rows"]),
        }
        new_count += 1
        print(f"    {t}: {desc[:90]}{'...' if len(desc) > 90 else ''}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {new_count} new, {len(result) - new_count} cached, saved to {out_path}")
    return result


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Phase 2: write a plain-English description for every table.")
    ap.add_argument("--dataset", choices=["spider", "bird"], required=True)
    ap.add_argument("--db", nargs="+", help="specific database id(s)")
    ap.add_argument("--all", action="store_true", help="every database found under data/<dataset>/")
    ap.add_argument("--level", type=int, default=0, choices=[0, 1, 2, 3],
                    help="0 = clean schema, 1-3 = a degraded copy (must exist, see degrade_schema.py)")
    ap.add_argument("--model", default="qwen2.5-coder:14b")
    ap.add_argument("--sample-rows", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="re-generate even if a cached description exists")
    ap.add_argument("--mock", action="store_true", help="skip the LLM, write placeholder text (for testing)")
    args = ap.parse_args()

    if not args.db and not args.all:
        sys.exit("give either --db <id...> or --all")
    db_ids = args.db if args.db else list_all_dbs(args.dataset)
    if not db_ids:
        sys.exit(f"No databases found under data/{args.dataset}/")

    if not args.mock:
        ollama_preflight(args.model)

    for db_id in db_ids:
        db_path = (find_db(args.dataset, db_id) if args.level == 0
                   else degraded_path(args.dataset, db_id, args.level))
        if not db_path.exists():
            print(f"[SKIP] {db_id}: not found -> {db_path}")
            continue
        print(f"[{args.dataset}] {db_id}  level {args.level}")
        t0 = time.time()
        enrich_database(args.dataset, db_id, db_path, args.level, args.model,
                        args.sample_rows, args.mock, args.force)
        print(f"  ({time.time() - t0:.1f}s)\n")

    print("Done.")


if __name__ == "__main__":
    main()
