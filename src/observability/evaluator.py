from __future__ import annotations

import json
import os
from typing import Any

import openai

from src.agent.prompts import EVALUATOR_PROMPT
from src.models.types import EvaluationScore

_MODEL = "llama-3.3-70b-versatile"
_BASE_URL = "https://api.groq.com/openai/v1"


class ResponseEvaluator:
    """Standalone batch evaluator — re-scores existing Langfuse traces retroactively."""

    def __init__(self) -> None:
        self._client = openai.OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=_BASE_URL)
        self._lf: Any = None

        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        if secret_key:
            try:
                from langfuse import Langfuse  # type: ignore

                self._lf = Langfuse(
                    secret_key=secret_key,
                    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
                    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                )
            except Exception:  # noqa: BLE001
                pass

    def evaluate_response(self, user_question: str, response_text: str) -> EvaluationScore:
        prompt = EVALUATOR_PROMPT.format(response_text=response_text, user_question=user_question)
        try:
            result = self._client.chat.completions.create(
                model=_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = result.choices[0].message.content or ""
            raw = raw.strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            return EvaluationScore(**data)
        except Exception:  # noqa: BLE001
            return EvaluationScore(
                causal_depth=5.0, accuracy=5.0, completeness=5.0, conflict_handling=5.0,
                actionability=5.0, overall=5.0, justification="Evaluation failed — default score assigned.",
            )

    def batch_evaluate_from_langfuse(self, limit: int = 50) -> int:
        if self._lf is None:
            print("Langfuse not configured — skipping batch evaluation.")
            return 0

        evaluated = 0
        try:
            traces = self._lf.fetch_traces(limit=limit).data
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to fetch traces: {exc}")
            return 0

        for trace in traces:
            try:
                scores = self._lf.fetch_scores(trace_id=trace.id).data
                if any(s.name == "overall_quality" for s in scores):
                    continue
            except Exception:  # noqa: BLE001
                continue

            user_question = trace.input or ""
            response_text = trace.output or ""
            if not response_text:
                continue

            score = self.evaluate_response(user_question, response_text)
            try:
                self._lf.score(
                    trace_id=trace.id,
                    name="overall_quality",
                    value=score.overall / 10.0,
                    comment=score.justification,
                )
                evaluated += 1
                print(f"  Scored trace {trace.id[:8]}... -> {score.overall:.1f}/10")
            except Exception as exc:  # noqa: BLE001
                print(f"  Failed to score trace {trace.id[:8]}...: {exc}")

        if self._lf:
            try:
                self._lf.flush()
            except Exception:  # noqa: BLE001
                pass

        return evaluated
