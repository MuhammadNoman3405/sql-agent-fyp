"""
retrieval_eval.py  --  Phase 2 experiment: does retrieval beat "paste everything"?
====================================================================================

Same 100 questions, same model, same seed as the Phase 1 baseline (baseline.py)
-- the ONLY thing that changes is how much schema the model gets to see:

    Phase 1 (full)     : every CREATE TABLE statement, always
    Phase 2 (this file) : only the top-K tables retrieval thinks are relevant,
                           optionally using the Idea A enriched descriptions
                           to do the retrieving

This produces the first real comparison for the paper: accuracy vs K, and
accuracy with enrichment on vs off, next to the Phase 1 number as the anchor.

Extra metric this script adds beyond execution accuracy: TABLE RECALL@K --
did the retrieved set actually contain every table the gold query needs?
If recall is low, no amount of SQL-writing skill can save the answer --
that's a retrieval failure, not a generation failure. Splitting failures
this way is exactly the kind of result reviewers want.

Usage (run from the repo root)
-------------------------------
  # one K value, enriched retrieval, 100 BIRD questions (same sample as baseline.py)
  python src\\eval\\retrieval_eval.py --dataset bird --k 3 --flavor enriched --n 100

  # sweep several K values in one go (builds the index once, reuses it)
  python src\\eval\\retrieval_eval.py --dataset bird --k 2 4 6 --flavor enriched --n 100

  # control group: raw names only, no enrichment
  python src\\eval\\retrieval_eval.py --dataset bird --k 4 --flavor raw --n 100

  # against a messy (degraded) schema
  python src\\eval\\retrieval_eval.py --dataset bird --k 4 --flavor enriched --level 2 --n 100

Needs Qdrant running (docker compose up -d) and the enrichment files already
built (enrich_schema.py) if --flavor enriched. Same Ollama setup as baseline.py.
Resumable the same way: progress is saved after every question.
"""

import argparse
import json
import random
import re
import sqlite3
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import baseline as bl          # reuse: locate_dataset, run_query, extract_sql, ask_ollama, ollama_preflight
import retrieval as rt         # reuse: embed, build_index, retrieve, get_client

RESULTS_DIR = Path("results")


# --------------------------------------------------------------------------
# Per-table CREATE TABLE text (so the prompt only contains retrieved tables)
# --------------------------------------------------------------------------
def get_create_statements(db_path: Path) -> dict:
    conn = bl.connect_ro(db_path)
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL"
    ).fetchall()
    conn.close()
    return {name: sql.strip() + ";" for name, sql in rows}


def schema_text_for_tables(creates: dict, tables: list) -> str:
    return "\n\n".join(creates[t] for t in tables if t in creates)


# --------------------------------------------------------------------------
# Approximate "which tables does the gold query actually touch"
# --------------------------------------------------------------------------
def gold_tables_used(gold_sql: str, all_tables: list) -> set:
    used = set()
    for t in all_tables:
        if re.search(r"(?<![\w`\"\[])" + re.escape(t) + r"(?![\w`\"\]])", gold_sql, re.I):
            used.add(t)
    return used


# --------------------------------------------------------------------------
# Index cache: build each database's collection once, reuse across questions/K values
# --------------------------------------------------------------------------
class IndexCache:
    def __init__(self, client, dataset, level, flavor, embed_model):
        self.client, self.dataset, self.level, self.flavor, self.embed_model = (
            client, dataset, level, flavor, embed_model)
        self.built = set()

    def ensure(self, db_id):
        if db_id in self.built:
            return rt.collection_name(self.dataset, db_id, self.level, self.flavor)
        print(f"  (building index for {db_id} ...)")
        name = rt.build_index(self.client, self.dataset, db_id, self.level, self.flavor,
                              embed_model=self.embed_model, verbose=False)
        self.built.add(db_id)
        return name


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Phase 2: accuracy with only top-K retrieved tables.")
    ap.add_argument("--dataset", choices=["bird", "spider"], default="bird")
    ap.add_argument("--model", default="qwen2.5-coder:14b")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--k", type=int, nargs="+", default=[4], help="one or more top-K values to test")
    ap.add_argument("--flavor", default="enriched", choices=["enriched", "raw"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42, help="MUST match baseline.py's seed to be comparable")
    ap.add_argument("--level", type=int, default=0, choices=[0, 1, 2, 3])
    ap.add_argument("--only-db", nargs="+")
    ap.add_argument("--no-evidence", action="store_true")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--qdrant-memory", action="store_true")
    ap.add_argument("--mock", choices=["gold", "bad"])
    args = ap.parse_args()

    dev_file, dbs = bl.locate_dataset(args.dataset)
    dev = json.loads(dev_file.read_text(encoding="utf-8"))
    gold_key = "SQL" if args.dataset == "bird" else "query"

    candidates = [
        i for i, ex in enumerate(dev)
        if ex["db_id"] in dbs and (not args.only_db or ex["db_id"] in args.only_db)
    ]
    if not candidates:
        sys.exit("No runnable questions found.")
    # SAME sampling as baseline.py: same seed + same candidate pool -> same question set
    picked = sorted(random.Random(args.seed).sample(candidates, min(args.n, len(candidates))))

    if not args.mock:
        bl.ollama_preflight(args.model)
        rt.ollama_preflight(args.embed_model)
    client = rt.get_client(args.qdrant_url, args.qdrant_memory)
    cache = IndexCache(client, args.dataset, args.level, args.flavor, args.embed_model)

    for k in args.k:
        run_one_k(args, dev, dbs, gold_key, picked, client, cache, k)


def run_one_k(args, dev, dbs, gold_key, picked, client, cache, k):
    model_tag = re.sub(r"[^A-Za-z0-9]+", "-", args.model).strip("-") if not args.mock else "mock-" + args.mock
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = f"retrieval_{args.dataset}_{model_tag}_{args.flavor}_K{k}_L{args.level}_n{len(picked)}_s{args.seed}"
    if args.only_db:
        stem += "_" + "-".join(args.only_db)
    jsonl_path = RESULTS_DIR / f"{stem}.jsonl"
    summary_path = RESULTS_DIR / f"{stem}_summary.json"

    records = bl.load_records(jsonl_path)
    done = {r["idx"] for r in records}
    print(f"\n=== K={k}  flavor={args.flavor}  dataset={args.dataset}  level={args.level} "
          f"questions={len(picked)}  already done={len(done)} ===")
    print(f"results file: {jsonl_path}")

    create_cache = {}  # db_id -> {table: CREATE TABLE text}
    table_list_cache = {}  # db_id -> [table names]

    with open(jsonl_path, "a", encoding="utf-8") as out:
        for i, idx in enumerate(picked, 1):
            if idx in done:
                continue
            ex = dev[idx]
            db_id = ex["db_id"]
            clean_db = dbs[db_id]
            pred_db = clean_db if args.level == 0 else rt.degraded_path(args.dataset, db_id, args.level)
            if not pred_db.exists():
                print(f"[{i}/{len(picked)}] {db_id}: SKIPPED (no level-{args.level} copy)")
                continue

            if db_id not in create_cache:
                create_cache[db_id] = get_create_statements(pred_db)
                table_list_cache[db_id] = list(create_cache[db_id])

            t_retr0 = time.time()
            collection = cache.ensure(db_id)
            retrieved = rt.retrieve(client, collection, ex["question"], k, args.embed_model)
            retrieved_tables = [t for t, _ in retrieved]
            retr_seconds = time.time() - t_retr0

            gold_sql = ex[gold_key]
            evidence = "" if (args.no_evidence or args.dataset != "bird") else (ex.get("evidence") or "")
            schema = schema_text_for_tables(create_cache[db_id], retrieved_tables)
            needed = gold_tables_used(gold_sql, table_list_cache[db_id])
            table_recall = (len(needed & set(retrieved_tables)) / len(needed)) if needed else None

            t0 = time.time()
            if args.mock == "gold":
                raw = gold_sql
            elif args.mock == "bad":
                raw = "SELECT 1"
            else:
                prompt = bl.build_user_prompt(schema, ex["question"], evidence)
                try:
                    raw = bl.ask_ollama(args.model, prompt, num_ctx=8192)
                except urllib.error.URLError as e:
                    sys.exit(f"\nLost connection to Ollama: {e}\nRerun the same command to resume.")
            pred_sql = bl.extract_sql(raw)
            gen_seconds = time.time() - t0

            gold_rows, gold_err = bl.run_query(clean_db, gold_sql)
            if gold_err:
                print(f"[{i}/{len(picked)}] {db_id}: gold query failed - excluded")
                continue
            pred_rows, pred_err = bl.run_query(pred_db, pred_sql)

            if pred_err:
                status = "exec_error"
            elif set(map(tuple, pred_rows)) == set(map(tuple, gold_rows)):
                status = "correct"
            else:
                status = "wrong_result"

            rec = {
                "idx": idx, "question_id": ex.get("question_id", idx), "db_id": db_id,
                "difficulty": ex.get("difficulty", "all"), "question": ex["question"],
                "gold_sql": gold_sql, "pred_sql": pred_sql, "status": status, "error": pred_err,
                "retrieved_tables": retrieved_tables, "gold_tables": sorted(needed),
                "table_recall": table_recall,
                "pred_num_rows": None if pred_rows is None else len(pred_rows),
                "gold_num_rows": len(gold_rows),
                "seconds": round(gen_seconds, 2), "retrieval_seconds": round(retr_seconds, 2),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            mark = {"correct": "OK   ", "wrong_result": "WRONG", "exec_error": "ERROR"}[status]
            recall_s = "n/a" if table_recall is None else f"{table_recall:.2f}"
            print(f"[{i}/{len(picked)}] {db_id:<24} {mark}  recall={recall_s}  ({gen_seconds:.1f}s)")

    summarize_retrieval(bl.load_records(jsonl_path), summary_path,
                        {"dataset": args.dataset, "model": model_tag, "flavor": args.flavor,
                         "k": k, "level": args.level, "seed": args.seed})


def summarize_retrieval(records, out_path, meta):
    total = len(records)
    if total == 0:
        print("No results yet.")
        return
    correct = sum(1 for r in records if r["status"] == "correct")
    wrong = sum(1 for r in records if r["status"] == "wrong_result")
    errs = sum(1 for r in records if r["status"] == "exec_error")
    recalls = [r["table_recall"] for r in records if r["table_recall"] is not None]
    avg_recall = sum(recalls) / len(recalls) if recalls else None
    full_recall_n = sum(1 for v in recalls if v >= 0.999)

    print("\n" + "=" * 62)
    print(f" RETRIEVAL RESULT   {meta['dataset']}  model={meta['model']}  "
          f"flavor={meta['flavor']}  K={meta['k']}  level={meta['level']}")
    print("=" * 62)
    print(f" questions scored     : {total}")
    print(f" EXECUTION ACCURACY   : {correct}/{total} = {100*correct/total:.1f}%")
    print(f" wrong but ran fine   : {wrong}  ({100*wrong/total:.1f}%)")
    print(f" SQL crashed          : {errs}  ({100*errs/total:.1f}%)")
    if avg_recall is not None:
        print(f" avg TABLE RECALL@K   : {avg_recall:.3f}   (all needed tables retrieved on {full_recall_n}/{len(recalls)} questions)")
    print("=" * 62)

    summary = {**meta, "scored": total, "correct": correct,
               "execution_accuracy": round(correct/total, 4),
               "wrong_result": wrong, "exec_error": errs,
               "avg_table_recall": round(avg_recall, 4) if avg_recall is not None else None,
               "full_recall_questions": full_recall_n}
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f" summary saved -> {out_path}")


if __name__ == "__main__":
    main()
