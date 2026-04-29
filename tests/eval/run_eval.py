"""
RAGAS evaluation harness for the Financial Advisor Agent.

Usage:
    python tests/eval/run_eval.py [--limit N] [--portfolio PORTFOLIO_ID]

Requires: ragas, langchain-groq, datasets — all in requirements.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
TEST_SET_PATH = Path(__file__).parent / "test_set.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _check_deps() -> bool:
    missing = []
    for pkg in ("ragas", "langchain_groq", "datasets"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install ragas langchain-groq datasets")
        return False
    return True


def run_eval(limit: int = 30, portfolio_filter: str | None = None) -> None:
    if not _check_deps():
        sys.exit(1)

    from datasets import Dataset
    from langchain_groq import ChatGroq
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    from src.agent.financial_advisor import FinancialAdvisorAgent

    with open(TEST_SET_PATH, encoding="utf-8") as f:
        test_data = json.load(f)["questions"]

    if portfolio_filter:
        test_data = [q for q in test_data if q["portfolio_id"] == portfolio_filter]
    test_data = test_data[:limit]

    print(f"Running RAGAS eval on {len(test_data)} questions...")

    agents: dict[str, FinancialAdvisorAgent] = {}
    questions, answers, contexts_list, ground_truths = [], [], [], []

    for i, item in enumerate(test_data, 1):
        pid = item["portfolio_id"]
        if pid not in agents:
            agents[pid] = FinancialAdvisorAgent(
                portfolio_id=pid,
                data_dir=DATA_DIR,
                capture_contexts=True,
            )
        agent = agents[pid]

        print(f"  [{i}/{len(test_data)}] {item['id']} — {item['question'][:60]}...")
        try:
            response, _, _ = agent.chat(item["question"])
            ctx = agent.last_contexts if agent.last_contexts else [response[:500]]
        except Exception as exc:
            print(f"    ERROR: {exc}")
            response = ""
            ctx = ["error"]

        questions.append(item["question"])
        answers.append(response)
        contexts_list.append(ctx)
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        print("ERROR: GROQ_API_KEY not set")
        sys.exit(1)

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key)

    print("\nRunning RAGAS evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
    )

    scores = {
        "faithfulness": float(results["faithfulness"]),
        "answer_relevancy": float(results["answer_relevancy"]),
        "context_precision": float(results["context_precision"]),
        "context_recall": float(results["context_recall"]),
    }
    scores["overall"] = sum(scores.values()) / 4

    output_path = RESULTS_DIR / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"scores": scores, "n_questions": len(test_data)}, f, indent=2)

    width = 22
    print(f"\n{'Metric':<{width}} {'Score':<8} {'Interpretation'}")
    print("─" * 55)
    interpretations = {
        "faithfulness": "Responses grounded in tool data",
        "answer_relevancy": "Questions fully addressed",
        "context_precision": "Right tools called for each query",
        "context_recall": "Tool outputs contained needed info",
    }
    for metric, score in scores.items():
        if metric == "overall":
            continue
        bar = "●" * int(score * 10) + "○" * (10 - int(score * 10))
        print(f"{metric:<{width}} {score:.2f}    [{bar}]  {interpretations[metric]}")
    print("─" * 55)
    print(f"{'Overall':<{width}} {scores['overall']:.2f}")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS evaluation for Financial Advisor Agent")
    parser.add_argument("--limit", type=int, default=30, help="Max questions to evaluate")
    parser.add_argument("--portfolio", type=str, default=None, help="Filter to one portfolio ID")
    args = parser.parse_args()
    run_eval(limit=args.limit, portfolio_filter=args.portfolio)
