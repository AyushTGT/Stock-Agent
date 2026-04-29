"""
A/B comparison between PROMPT_V1 (minimal) and PROMPT_V2 (production).

Usage:
    python tests/eval/prompt_ab_test.py

Uses the agent's built-in self-evaluation (EvaluationScore) — no RAGAS call needed.
Saves results to tests/eval/results/ab_results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

_TEST_QUESTIONS = [
    ("PORTFOLIO_001", "Why is my portfolio down today? Give me the full causal breakdown."),
    ("PORTFOLIO_002", "What is my sector exposure and where am I most concentrated?"),
    ("PORTFOLIO_001", "Are there any conflicting signals I should know about?"),
    ("PORTFOLIO_003", "How does today's market trend affect a conservative investor like me?"),
    ("PORTFOLIO_002", "What should I do to reduce my concentration risk?"),
    ("PORTFOLIO_001", "Which news events had the biggest impact on my holdings today?"),
    ("PORTFOLIO_002", "Is Bajaj Finance a buy or should I hold given the current signals?"),
    ("PORTFOLIO_003", "Which of my holdings is acting as the best hedge today?"),
    ("PORTFOLIO_001", "What is the confidence level of today's analysis and why?"),
    ("PORTFOLIO_002", "How much of my portfolio loss today is attributable to FII selling?"),
]


def run_ab_test() -> None:
    from src.agent.financial_advisor import FinancialAdvisorAgent

    results_v1: list[dict] = []
    results_v2: list[dict] = []

    print(f"Running A/B test on {len(_TEST_QUESTIONS)} questions × 2 prompt versions...\n")

    for i, (pid, question) in enumerate(_TEST_QUESTIONS, 1):
        print(f"  [{i}/{len(_TEST_QUESTIONS)}] {question[:60]}...")

        for version, results_list in [("v1", results_v1), ("v2", results_v2)]:
            try:
                agent = FinancialAdvisorAgent(
                    portfolio_id=pid,
                    data_dir=DATA_DIR,
                    prompt_version=version,
                )
                _, eval_score, metrics = agent.chat(question)
                results_list.append({
                    "portfolio_id": pid,
                    "question": question,
                    "overall": eval_score.overall,
                    "causal_depth": eval_score.causal_depth,
                    "accuracy": eval_score.accuracy,
                    "completeness": eval_score.completeness,
                    "conflict_handling": eval_score.conflict_handling,
                    "actionability": eval_score.actionability,
                    "latency_ms": metrics.total_latency_ms,
                    "cost_usd": metrics.estimated_cost_usd,
                    "justification": eval_score.justification,
                })
            except Exception as exc:
                print(f"    ERROR ({version}): {exc}")
                results_list.append({"question": question, "overall": 0.0, "error": str(exc)})

    def avg(results: list[dict], key: str) -> float:
        vals = [r[key] for r in results if key in r and "error" not in r]
        return sum(vals) / len(vals) if vals else 0.0

    criteria = ["causal_depth", "accuracy", "completeness", "conflict_handling", "actionability", "overall"]

    output = {
        "v1": {"per_question": results_v1, "averages": {k: avg(results_v1, k) for k in criteria}},
        "v2": {"per_question": results_v2, "averages": {k: avg(results_v2, k) for k in criteria}},
        "n_questions": len(_TEST_QUESTIONS),
    }

    out_path = RESULTS_DIR / "ab_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    col_w = 22
    print(f"\n{'Criterion':<{col_w}} {'V1 (minimal)':<14} {'V2 (production)':<16} {'Delta'}")
    print("─" * 65)
    for k in criteria:
        v1_avg = output["v1"]["averages"][k]
        v2_avg = output["v2"]["averages"][k]
        delta = v2_avg - v1_avg
        delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
        color = "✓" if delta > 0 else ("✗" if delta < 0 else "=")
        print(f"{k:<{col_w}} {v1_avg:<14.2f} {v2_avg:<16.2f} {delta_str}  {color}")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    run_ab_test()
