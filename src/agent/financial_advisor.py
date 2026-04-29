from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import openai

from src.agent.prompts import EVALUATOR_PROMPT, SYSTEM_PROMPT_TEMPLATE
from src.agent.tools import TOOL_DEFINITIONS, ToolDispatcher
from src.data_loader import DataLoader
from src.models.types import EvaluationScore
from src.observability.tracer import LangfuseTracer

_MODEL = "llama-3.3-70b-versatile"
_BASE_URL = "https://api.groq.com/openai/v1"
_MAX_TOOL_ITERATIONS = 8
_HISTORY_WINDOW = 20


class FinancialAdvisorAgent:

    def __init__(self, portfolio_id: str, data_dir: Path, api_key: str | None = None) -> None:
        self.portfolio_id = portfolio_id
        resolved_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not resolved_key:
            raise ValueError("Groq API key is required. Pass api_key= or set GROQ_API_KEY in your environment.")
        self._client = openai.OpenAI(
            api_key=resolved_key,
            base_url=_BASE_URL,
        )
        self._dispatcher = ToolDispatcher(data_dir)
        self._tracer = LangfuseTracer()
        DataLoader.get_instance(data_dir)
        self._system_msg: dict = {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE}
        self._history: list[dict] = []

    def chat(self, user_message: str) -> tuple[str, EvaluationScore]:
        """Process one user turn. Returns (response_text, eval_score)."""
        trace = self._tracer.create_trace(
            name="financial-advisor-chat",
            user_id=self.portfolio_id,
            input=user_message,
        )

        self._history.append({"role": "user", "content": user_message})
        response_text = self._run_tool_loop(self._build_messages(), trace)
        self._history.append({"role": "assistant", "content": response_text})
        self._trim_history()

        eval_score = self._run_self_evaluation(user_message, response_text, trace)

        self._tracer.finalize_trace(
            trace=trace,
            output=response_text,
            score_value=eval_score.overall / 10.0,
            score_comment=eval_score.justification,
        )

        return response_text, eval_score

    def _run_tool_loop(self, history_messages: list[dict], trace: Any) -> str:
        messages: list[dict] = [self._system_msg] + history_messages

        for iteration in range(_MAX_TOOL_ITERATIONS):
            span = self._tracer.start_generation(
                trace=trace,
                name=f"groq-call-{iteration + 1}",
                model=_MODEL,
                input_messages=messages,
            )

            response = self._client.chat.completions.create(
                model=_MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                max_tokens=2048,
                parallel_tool_calls=False,
            )

            self._tracer.end_generation(span=span, output=response, usage=response.usage)

            assistant_msg = response.choices[0].message
            messages.append(self._serialize_assistant_message(assistant_msg))

            if not assistant_msg.tool_calls:
                return assistant_msg.content or ""

            for tool_call in assistant_msg.tool_calls:
                tool_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                tool_span = self._tracer.start_span(trace=trace, name=f"tool:{tool_name}", input=args)
                result = self._dispatcher.dispatch(tool_name, args)
                self._tracer.end_span(span=tool_span, output=result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": self._clip_tool_result(result),
                })

        # Reached max iterations without a final text response
        return assistant_msg.content or ""  # type: ignore[possibly-undefined]

    def _run_self_evaluation(
        self, user_question: str, response_text: str, trace: Any
    ) -> EvaluationScore:
        prompt = EVALUATOR_PROMPT.format(
            response_text=response_text,
            user_question=user_question,
        )

        span = self._tracer.start_generation(
            trace=trace,
            name="self-evaluation",
            model=_MODEL,
            input_messages=[{"role": "user", "content": prompt}],
        )

        try:
            eval_response = self._client.chat.completions.create(
                model=_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            self._tracer.end_generation(span=span, output=eval_response, usage=eval_response.usage)
            raw = eval_response.choices[0].message.content or ""
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
                causal_depth=5.0,
                accuracy=5.0,
                completeness=5.0,
                conflict_handling=5.0,
                actionability=5.0,
                overall=5.0,
                justification="Evaluation failed — default score assigned.",
            )

    def _build_messages(self) -> list[dict]:
        if len(self._history) <= _HISTORY_WINDOW:
            return list(self._history)
        return [self._history[0]] + self._history[-(_HISTORY_WINDOW - 1):]

    def _trim_history(self) -> None:
        max_stored = _HISTORY_WINDOW * 2
        if len(self._history) > max_stored:
            self._history = [self._history[0]] + self._history[-(max_stored - 1):]

    @staticmethod
    def _serialize_assistant_message(msg) -> dict:
        d: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        return d

    @staticmethod
    def _clip_tool_result(result: dict, max_chars: int = 3000) -> str:
        raw = json.dumps(result, default=str)
        if len(raw) <= max_chars:
            return raw
        if "chains" in result and isinstance(result["chains"], list):
            trimmed = {**result, "chains": result["chains"][:3]}
            raw = json.dumps(trimmed, default=str)
            if len(raw) <= max_chars:
                return raw
        return raw[:max_chars] + '... [truncated]"}'
