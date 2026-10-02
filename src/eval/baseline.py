"""
baseline.py  --  Phase 1: the "dumb baseline"
=============================================

The simplest possible Text-to-SQL system. No agents, no retrieval, no repair loop:

    question + WHOLE schema pasted in one prompt  ->  local LLM (Ollama) writes SQL
        ->  run it on the database  ->  compare rows with the gold query's rows

The accuracy number this script prints is your BASELINE. Every later improvement
(schema enrichment, verifier, repair loop) is measured against it.

Metric: EXECUTION ACCURACY (EX)
    A question counts as correct if the predicted query returns the same SET of rows
    as the gold query (this is the official BIRD/Spider style check).

Every wrong answer is also classified:
    exec_error   -> the SQL crashed (wrong column, syntax error, timeout ...)
    wrong_result -> the SQL ran fine but returned the WRONG rows  <-- "silent wrongness"
                    This is the dangerous failure your verifier will attack in Phase 3.

Usage (run from the repo root, e.g. C:\\Users\\mnoma\\Desktop\\sql-agent-fyp)
--------------------------------------------------------------------------
  # 1) test the harness WITHOUT any LLM (gold SQL is returned as the "prediction"; must give 100%)
  python src\\eval\\baseline.py --dataset bird --mock gold --n 20 --only-db california_schools

  # 2) tiny real run with Ollama (5 questions) to check everything works
  python src\\eval\\baseline.py --dataset bird --n 5 --only-db california_schools

  # 3) the real baseline: 100 random BIRD dev questions
  python src\\eval\\baseline.py --dataset bird --n 100

  # 4) same questions on a messy schema (needs the degraded DBs from degrade_schema.py)
  python src\\eval\\baseline.py --dataset bird --n 100 --level 2

  # Spider instead of BIRD
  python src\\eval\\baseline.py --dataset spider --n 100

The run is saved after EVERY question, so if it stops (laptop sleeps, Ctrl+C) just run
the same command again and it continues where it stopped.

Only the Python standard library is used. Nothing to pip install.
"""

import argparse
import json
import random
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

OLLAMA_BASE = "http://localhost:11434"
EXEC_TIMEOUT_SEC = 60  # any query running longer than this is killed
RESULTS_DIR = Path("results")

SYSTEM_PROMPT = (
    "You are an expert SQLite query writer. Given a database schema and a question, "
    "reply with exactly ONE SQLite SELECT query that answers the question. "
    "Output only the SQL. No explanation, no comments, no markdown."
)


# --------------------------------------------------------------------------
# Finding the data (works whatever folder nesting your unzip tool produced)
# --------------------------------------------------------------------------
def locate_dataset(dataset: str):
    root = Path("data") / dataset
    if not root.exists():
        sys.exit(f"Folder not found: {root}  (run this from the repo root)")
    dev_file = next(iter(sorted(root.rglob("dev.json"))), None)
    if dev_file is None:
        sys.exit(f"dev.json not found under {root}")
    dbs = {}
    for p in root.rglob("*.sqlite"):
        if p.parent.name == p.stem:  # <db_id>/<db_id>.sqlite
            dbs.setdefault(p.stem, p)
    return dev_file, dbs


def degraded_path(dataset: str, db_id: str, level: int) -> Path:
    return Path("data") / "degraded" / dataset / db_id / f"level{level}" / f"{db_id}.sqlite"


# --------------------------------------------------------------------------
# SQLite helpers
# --------------------------------------------------------------------------
def connect_ro(db_path: Path) -> sqlite3.Connection:
    uri = db_path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return conn


def get_schema_text(db_path: Path) -> str:
    """The schema exactly as the database defines it: all CREATE TABLE statements."""
    conn = connect_ro(db_path)
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
    finally:
        conn.close()
    return "\n\n".join(r[0].strip() + ";" for r in rows)


def run_query(db_path: Path, sql: str, timeout: int = EXEC_TIMEOUT_SEC):
    """Run a SELECT safely. Returns (rows, error). Read-only, time-limited, SELECT/WITH only."""
    if not re.match(r"^\s*(select|with)\b", sql or "", re.I):
        return None, "not_a_select_query"
    conn = connect_ro(db_path)
    deadline = time.time() + timeout
    conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 100000)
    try:
        return conn.execute(sql).fetchall(), None
    except sqlite3.Error as e:
        msg = str(e)
        return None, "timeout" if "interrupt" in msg.lower() else msg
    finally:
        conn.close()


# --------------------------------------------------------------------------
# The LLM part (Ollama)
# --------------------------------------------------------------------------
def build_user_prompt(schema: str, question: str, evidence: str) -> str:
    parts = [f"### SQLite database schema\n{schema}"]
    if evidence:
        parts.append(f"### Hint\n{evidence}")
    parts.append(f"### Question\n{question}")
    parts.append("### SQL query")
    return "\n\n".join(parts)


def extract_sql(text: str) -> str:
    """Pull the bare SQL out of whatever the model wrote."""
    m = re.search(r"```(?:sql|sqlite)?\s*(.*?)```", text, re.S | re.I)
    if m:
        text = m.group(1)
    text = re.sub(r"^\s*(sql|sqlite)\s*:\s*", "", text.strip(), flags=re.I)
    return text.split(";")[0].strip()


def ollama_preflight(model: str):
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=10) as r:
            names = [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        sys.exit(
            "Cannot reach Ollama at http://localhost:11434\n"
            "  -> Is Ollama installed and running? (open the Ollama app, or run: ollama serve)"
        )
    if not any(n == model or n.startswith(model + ":") or n == model + ":latest" for n in names):
        sys.exit(
            f"Model '{model}' is not downloaded yet.\n"
            f"  -> run:  ollama pull {model}\n"
            f"  (models you have: {', '.join(names) if names else 'none'})"
        )


def ask_ollama(model: str, user_prompt: str, num_ctx: int, timeout: int = 900) -> str:
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": num_ctx},
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


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def summarize(records: list, out_path: Path, meta: dict):
    total = len(records)
    if total == 0:
        print("No results yet.")
        return
    status = Counter(r["status"] for r in records)
    correct = status["correct"]
    by_diff = defaultdict(lambda: [0, 0])
    for r in records:
        by_diff[r["difficulty"]][1] += 1
        by_diff[r["difficulty"]][0] += r["status"] == "correct"

    avg_time = sum(r["seconds"] for r in records) / total
    print("\n" + "=" * 62)
    print(f" BASELINE RESULT   {meta['dataset']}  model={meta['model']}  level={meta['level']}")
    print("=" * 62)
    print(f" questions scored     : {total}")
    print(f" EXECUTION ACCURACY   : {correct}/{total} = {100 * correct / total:.1f}%")
    print(f" wrong but ran fine   : {status['wrong_result']}  ({100 * status['wrong_result'] / total:.1f}%)   <- silent wrongness")
    print(f" SQL crashed          : {status['exec_error']}  ({100 * status['exec_error'] / total:.1f}%)")
    print(f" avg seconds/question : {avg_time:.1f}")
    if len(by_diff) > 1:
        print(" by difficulty:")
        for d, (c, n) in sorted(by_diff.items()):
            print(f"    {d:<12} {c}/{n} = {100 * c / n:.1f}%")
    print("=" * 62)

    summary = {
        **meta,
        "scored": total,
        "correct": correct,
        "execution_accuracy": round(correct / total, 4),
        "wrong_result": status["wrong_result"],
        "exec_error": status["exec_error"],
        "avg_seconds": round(avg_time, 2),
        "by_difficulty": {d: {"correct": c, "total": n} for d, (c, n) in by_diff.items()},
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f" summary saved -> {out_path}")


def load_records(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Phase 1 baseline: whole schema in one prompt.")
    ap.add_argument("--dataset", choices=["bird", "spider"], default="bird")
    ap.add_argument("--model", default="qwen2.5-coder:7b", help="Ollama model name")
    ap.add_argument("--n", type=int, default=100, help="number of dev questions to sample")
    ap.add_argument("--seed", type=int, default=42, help="same seed = same questions every run")
    ap.add_argument("--level", type=int, default=0, choices=[0, 1, 2, 3],
                    help="schema messiness the model sees (0 = clean original)")
    ap.add_argument("--only-db", nargs="+", help="only use questions from these database ids")
    ap.add_argument("--no-evidence", action="store_true", help="do not give BIRD's hint text to the model")
    ap.add_argument("--num-ctx", type=int, default=8192, help="Ollama context window in tokens")
    ap.add_argument("--mock", choices=["gold", "bad"],
                    help="test the harness without an LLM: 'gold' returns the gold SQL, 'bad' returns SELECT 1")
    args = ap.parse_args()

    dev_file, dbs = locate_dataset(args.dataset)
    dev = json.loads(dev_file.read_text(encoding="utf-8"))
    gold_key = "SQL" if args.dataset == "bird" else "query"

    # Which questions can we run? (their clean database must exist on disk)
    candidates = [
        i for i, ex in enumerate(dev)
        if ex["db_id"] in dbs and (not args.only_db or ex["db_id"] in args.only_db)
    ]
    if not candidates:
        sys.exit("No runnable questions found (check --only-db spelling and the data folder).")
    picked = sorted(random.Random(args.seed).sample(candidates, min(args.n, len(candidates))))

    if not args.mock:
        ollama_preflight(args.model)

    model_tag = "mock-" + args.mock if args.mock else re.sub(r"[^A-Za-z0-9]+", "-", args.model).strip("-")
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = f"baseline_{args.dataset}_{model_tag}_L{args.level}_n{len(picked)}_s{args.seed}"
    if args.only_db:
        stem += "_" + "-".join(args.only_db)
    jsonl_path = RESULTS_DIR / f"{stem}.jsonl"
    summary_path = RESULTS_DIR / f"{stem}_summary.json"

    records = load_records(jsonl_path)
    done = {r["idx"] for r in records}
    print(f"dataset={args.dataset}  model={model_tag}  level={args.level}  "
          f"questions={len(picked)}  already done={len(done)}")
    print(f"results file: {jsonl_path}\n")

    skipped_missing = 0
    with open(jsonl_path, "a", encoding="utf-8") as out:
        for k, idx in enumerate(picked, 1):
            if idx in done:
                continue
            ex = dev[idx]
            db_id = ex["db_id"]
            clean_db = dbs[db_id]
            pred_db = clean_db if args.level == 0 else degraded_path(args.dataset, db_id, args.level)
            if not pred_db.exists():
                skipped_missing += 1
                print(f"[{k}/{len(picked)}] {db_id}: SKIPPED (no degraded level-{args.level} copy; run degrade_schema.py)")
                continue

            gold_sql = ex[gold_key]
            evidence = "" if (args.no_evidence or args.dataset != "bird") else (ex.get("evidence") or "")

            t0 = time.time()
            if args.mock == "gold":
                raw = gold_sql
            elif args.mock == "bad":
                raw = "SELECT 1"
            else:
                prompt = build_user_prompt(get_schema_text(pred_db), ex["question"], evidence)
                try:
                    raw = ask_ollama(args.model, prompt, args.num_ctx)
                except urllib.error.URLError as e:
                    sys.exit(f"\nLost connection to Ollama: {e}\nRun the same command again to resume.")
            pred_sql = extract_sql(raw)
            gen_seconds = time.time() - t0

            # Gold answer always comes from the CLEAN database; the prediction runs on the schema the model saw.
            gold_rows, gold_err = run_query(clean_db, gold_sql)
            if gold_err:
                print(f"[{k}/{len(picked)}] {db_id}: gold query failed ({gold_err[:60]}) - excluded")
                continue
            pred_rows, pred_err = run_query(pred_db, pred_sql)

            if pred_err:
                status = "exec_error"
            elif set(map(tuple, pred_rows)) == set(map(tuple, gold_rows)):
                status = "correct"
            else:
                status = "wrong_result"

            rec = {
                "idx": idx,
                "question_id": ex.get("question_id", idx),
                "db_id": db_id,
                "difficulty": ex.get("difficulty", "all"),
                "question": ex["question"],
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "status": status,
                "error": pred_err,
                "pred_num_rows": None if pred_rows is None else len(pred_rows),
                "gold_num_rows": len(gold_rows),
                "seconds": round(gen_seconds, 2),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            mark = {"correct": "OK   ", "wrong_result": "WRONG", "exec_error": "ERROR"}[status]
            print(f"[{k}/{len(picked)}] {db_id:<28} {mark} ({gen_seconds:.1f}s)")

    if skipped_missing:
        print(f"\nNote: {skipped_missing} questions skipped because their degraded database is missing.")

    meta = {"dataset": args.dataset, "model": model_tag, "level": args.level,
            "seed": args.seed, "evidence_used": not args.no_evidence and args.dataset == "bird"}
    summarize(load_records(jsonl_path), summary_path, meta)


if __name__ == "__main__":
    main()
