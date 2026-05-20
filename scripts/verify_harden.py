#!/usr/bin/env python3
"""3-gate verification of a harden iteration's newly-rewritten questions.

For every question replace-easy rewrote in harden iteration 4:
  harder        — must NOT still be >=5/6 parsers correct (LLM-judge)
  not_halluc    — no parser scored correct on a value absent from its markdown
                  (audit token-coverage check)
  gold_correct  — a fresh vision call agrees with the new gold

A question failing not_halluc or gold_correct is genuinely broken — it is
reverted to its prior version (popped from versions[]), so the bank never
ships a hallucinated or wrong-gold question. 'still easy' is kept (valid, just
not hard) and only logged. Reverted docs are re-evaluated and the dashboard
rebuilt.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from pilot_50_eval import format_to_struct                       # noqa: E402
from audit_qa import check_hallucination, vision_check           # noqa: E402
from v500_harden_chain import per_parser_scores, n_questions     # noqa: E402

RUN = REPO / "runs" / "v500"
QA, EV = RUN / "qa", RUN / "eval"
PY = str(REPO / ".venv" / "bin" / "python")
PARSERS = ["extend_performance_v1_0_0", "extend_performance_v2_0_0_advanced",
           "aws_textract", "azure_di", "reducto", "llamaparse"]


def sh(cmd: list, log: Path) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    with log.open("w") as f:
        return subprocess.run([str(c) for c in cmd], stdout=f,
                              stderr=subprocess.STDOUT, cwd=REPO).returncode


def md_path(parser: str, dp: str) -> Path:
    return RUN / "parses" / parser / f"{Path(dp).with_suffix('').as_posix()}.md"


def main() -> int:
    t0 = time.time()
    log_path = QA / "artifacts" / "harden_iter4" / "replace_easy_log.json"
    swap = json.loads(log_path.read_text())
    new_qids = [e["question_id"] for e in swap
                if str(e.get("outcome", "")).startswith("replaced")]
    print(f"verifying {len(new_qids)} newly-rewritten questions", flush=True)

    bank = json.loads((QA / "bank_r1.json").read_text())
    bank_by_id = {it["question_id"]: it for it in bank["items"]}
    rescored = json.loads((EV / "eval_r1_rescored.json").read_text())
    q_loc = {}
    for dp, page in rescored.items():
        for qid, q in page["questions"].items():
            q_loc[qid] = (dp, q)
    sample = {r["dest_path"]: r for r in csv.DictReader((RUN / "sample.csv").open())}
    stem_to_dest = {Path(r["dest_path"]).stem: r["dest_path"] for r in sample.values()}

    def gate(qid: str):
        it = bank_by_id.get(qid)
        if not it or qid not in q_loc:
            return None
        dp, eq = q_loc[qid]
        n_correct = len(eq.get("parsers_correct_llm", []))
        mds = {}
        for p in PARSERS:
            mp = md_path(p, dp)
            if mp.exists():
                mds[p] = mp.read_text()
        suspects = check_hallucination(eq, mds, format_to_struct(it.get("response_format", "")))
        img = RUN / "images" / f"{it['source_file']}.png"
        vis = (vision_check(img, it["question"], it["response_format"], it["gold_answer"])
               if img.exists() else {"verdict": "uncertain", "note": "no image"})
        broken_reason = ""
        if suspects:
            broken_reason = "hallucination"
        elif vis.get("verdict") == "gold_wrong":
            broken_reason = "gold_wrong"
        return {
            "qid": qid, "source_file": it["source_file"],
            "n_correct": n_correct, "still_easy": n_correct >= 5,
            "n_suspects": len(suspects), "vision_verdict": vis.get("verdict"),
            "vision_note": vis.get("note", ""),
            "verdict": "broken" if broken_reason else "ok",
            "broken_reason": broken_reason,
        }

    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(gate, q) for q in new_qids]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                results.append(r)
            if i % 50 == 0 or i == len(new_qids):
                print(f"  [{i}/{len(new_qids)}] gated ({(time.time()-t0)/60:.1f}min)",
                      flush=True)

    broken = [r for r in results if r["verdict"] == "broken"]
    still_easy = [r for r in results if r["verdict"] == "ok" and r["still_easy"]]
    ok_hard = [r for r in results if r["verdict"] == "ok" and not r["still_easy"]]
    print(f"\n=== verification ===")
    print(f"  ok & genuinely harder : {len(ok_hard)}")
    print(f"  ok but still easy     : {len(still_easy)}")
    print(f"  broken -> reverted    : {len(broken)}  "
          f"({sum(r['broken_reason']=='hallucination' for r in broken)} hallucination, "
          f"{sum(r['broken_reason']=='gold_wrong' for r in broken)} gold_wrong)")

    # revert broken questions to their versioned prior
    reverted_docs = set()
    for r in broken:
        it = bank_by_id[r["qid"]]
        versions = it.get("versions", [])
        if not versions:
            continue
        prior = versions.pop()
        for k in ("question", "gold_answer", "response_format", "capabilities"):
            if prior.get(k) is not None:
                it[k] = prior[k]
        it.setdefault("verify_log", []).append(
            {"iteration": "harden 4", "reverted": True, "reason": r["broken_reason"]})
        reverted_docs.add(it["source_file"])

    (QA / "bank_r1.json").write_text(
        json.dumps(bank, indent=2, ensure_ascii=False) + "\n")
    (RUN / "verify_harden4_report.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))

    # re-eval the reverted docs + rebuild
    if reverted_docs:
        cache = EV / "answers_flat"
        n_wiped = 0
        for sf in reverted_docs:
            for f in cache.glob(f"{sf}__*.json"):
                f.unlink()
                n_wiped += 1
        print(f"  reverted {len(broken)} questions across {len(reverted_docs)} "
              f"docs — wiped {n_wiped} cache files", flush=True)
        sh([PY, "scripts/pilot_50_eval.py", "--run-dir", RUN, "--max-workers", "12"],
           RUN / "verify_harden4_eval.log")
        sh([PY, "scripts/pilot_50_rescore.py", "--run-dir", RUN, "--max-workers", "12"],
           RUN / "verify_harden4_rescore.log")
        hist_path = EV / "iteration_history.json"
        hist = json.loads(hist_path.read_text())
        if hist and hist[-1].get("label") == "harden 4":
            rp = EV / "eval_r1_rescored.json"
            hist[-1]["n_questions"] = n_questions(rp)
            hist[-1]["scores"] = per_parser_scores(rp)
            hist[-1]["label"] = "harden 4 (verified)"
            hist_path.write_text(json.dumps(hist, indent=2))
        sh([PY, "scripts/pilot_50_build_html.py", "--run-dir", RUN, "--mode", "post"],
           RUN / "verify_harden4_html.log")
        if (RUN / "review_post.html").exists():
            shutil.copy(RUN / "review_post.html", RUN / "dashboard.html")

    print(f"DONE in {(time.time()-t0)/60:.1f}min — report: "
          f"runs/v500/verify_harden4_report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
