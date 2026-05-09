from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import openai

from src.agent.prompt_versions import PROMPT_MAP, PromptVersion
from src.agent.prompts import EVALUATOR_PROMPT
from src.agent.tools import TOOL_DEFINITIONS, ToolDispatcher
from src.data_loader import DataLoader
from src.guardrails.guardrails import (
    InputGuard,
    OutputGuard,
    default_eval_for_guardrail,
    refusal_response,
)
from src.models.types import EvaluationScore, TurnMetrics
from src.observability.tracer import LangfuseTracer

_GROQ_MODEL = "llama-3.3-70b-versatile"
_GEMINI_MODEL_DEFAULT = "gemini-2.0-flash-lite"
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_MAX_TOOL_ITERATIONS = 8
_HISTORY_WINDOW = 20
_GROQ_COST_INPUT_PER_M = 0.59
_GROQ_COST_OUTPUT_PER_M = 0.79
_GEMINI_COST_INPUT_PER_M = 0.075
_GEMINI_COST_OUTPUT_PER_M = 0.30


class FinancialAdvisorAgent:

    def __init__(
        self,
        portfolio_id: str,
        data_dir: Path,
        api_key: str | None = None,
        provider: str = "groq",
        model: str | None = None,
        prompt_version: PromptVersion = "v2",
        capture_contexts: bool = False,
    ) -> None:
        self.portfolio_id = portfolio_id
        self.provider = provider.lower()

        if self.provider == "gemini":
            resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
            if not resolved_key:
                raise ValueError("Gemini API key is required. Pass api_key= or set GEMINI_API_KEY in your environment.")
            self._model = model or _GEMINI_MODEL_DEFAULT
            self._cost_input = _GEMINI_COST_INPUT_PER_M
            self._cost_output = _GEMINI_COST_OUTPUT_PER_M
            self._client = openai.OpenAI(
                api_key=resolved_key,
                base_url=_GEMINI_BASE_URL,
            )
        else:
            resolved_key = api_key or os.environ.get("GROQ_API_KEY", "")
            if not resolved_key:
                raise ValueError("Groq API key is required. Pass api_key= or set GROQ_API_KEY in your environment.")
            self._model = model or _GROQ_MODEL
            self._cost_input = _GROQ_COST_INPUT_PER_M
            self._cost_output = _GROQ_COST_OUTPUT_PER_M
            self._client = openai.OpenAI(
                api_key=resolved_key,
                base_url=_GROQ_BASE_URL,
            )

        self._dispatcher = ToolDispatcher(data_dir)
        self._tracer = LangfuseTracer()
        self._input_guard = InputGuard()
        self._output_guard = OutputGuard()
        self._capture_contexts = capture_contexts
        self.last_contexts: list[str] = []
        DataLoader.get_instance(data_dir)
        self._system_msg: dict = {"role": "system", "content": PROMPT_MAP[prompt_version]}
        self._history: list[dict] = []

    def chat(self, user_message: str) -> tuple[str, EvaluationScore, TurnMetrics]:
        """Process one user turn. Returns (response_text, eval_score, turn_metrics)."""
        turn_start = time.perf_counter()
        total_prompt_tokens = 0
        total_completion_tokens = 0

        guard_result = self._input_guard.check(user_message)
        if not guard_result.allowed:
            elapsed_ms = (time.perf_counter() - turn_start) * 1000
            metrics = TurnMetrics(
                total_latency_ms=round(elapsed_ms, 1),
                tool_loop_latency_ms=0.0,
                eval_latency_ms=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                estimated_cost_usd=0.0,
                tool_calls_count=0,
            )
            return refusal_response(guard_result.category), default_eval_for_guardrail(), metrics

        trace = self._tracer.create_trace(
            name="financial-advisor-chat",
            user_id=self.portfolio_id,
            input=user_message,
        )

        self._history.append({"role": "user", "content": user_message})
        if self._capture_contexts:
            self.last_contexts = []

        loop_start = time.perf_counter()
        response_text, loop_tokens, tool_calls_count = self._run_tool_loop(self._build_messages(), trace)
        loop_ms = (time.perf_counter() - loop_start) * 1000
        total_prompt_tokens += loop_tokens[0]
        total_completion_tokens += loop_tokens[1]

        response_text = self._output_guard.check(response_text)
        self._history.append({"role": "assistant", "content": response_text})
        self._trim_history()

        eval_start = time.perf_counter()
        eval_score, eval_tokens = self._run_self_evaluation(user_message, response_text, trace)
        eval_ms = (time.perf_counter() - eval_start) * 1000
        total_prompt_tokens += eval_tokens[0]
        total_completion_tokens += eval_tokens[1]

        total_ms = (time.perf_counter() - turn_start) * 1000
        cost = (total_prompt_tokens / 1_000_000) * self._cost_input + (
            total_completion_tokens / 1_000_000
        ) * self._cost_output

        metrics = TurnMetrics(
            total_latency_ms=round(total_ms, 1),
            tool_loop_latency_ms=round(loop_ms, 1),
            eval_latency_ms=round(eval_ms, 1),
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            estimated_cost_usd=round(cost, 6),
            tool_calls_count=tool_calls_count,
        )

        self._tracer.finalize_trace(
            trace=trace,
            output=response_text,
            score_value=eval_score.overall / 10.0,
            score_comment=eval_score.justification,
        )

        return response_text, eval_score, metrics

    def _run_tool_loop(
        self, history_messages: list[dict], trace: Any
    ) -> tuple[str, tuple[int, int], int]:
        messages: list[dict] = [self._system_msg] + history_messages
        total_prompt = 0
        total_completion = 0
        tool_calls_count = 0
        assistant_msg = None

        for iteration in range(_MAX_TOOL_ITERATIONS):
            span = self._tracer.start_generation(
                trace=trace,
                name=f"llm-call-{iteration + 1}",
                model=self._model,
                input_messages=messages,
            )

            call_kwargs: dict = {
                "model": self._model,
                "messages": messages,
                "tools": TOOL_DEFINITIONS,
                "tool_choice": "auto",
                "max_tokens": 2048,
            }
            if self.provider == "groq":
                call_kwargs["parallel_tool_calls"] = False

            response = self._client.chat.completions.create(**call_kwargs)

            self._tracer.end_generation(span=span, output=response, usage=response.usage)
            if response.usage:
                total_prompt += response.usage.prompt_tokens or 0
                total_completion += response.usage.completion_tokens or 0

            assistant_msg = response.choices[0].message
            messages.append(self._serialize_assistant_message(assistant_msg))

            if not assistant_msg.tool_calls:
                return assistant_msg.content or "", (total_prompt, total_completion), tool_calls_count

            for tool_call in assistant_msg.tool_calls:
                tool_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                tool_span = self._tracer.start_span(trace=trace, name=f"tool:{tool_name}", input=args)
                result = self._dispatcher.dispatch(tool_name, args)
                self._tracer.end_span(span=tool_span, output=result)

                clipped = self._clip_tool_result(result)
                if self._capture_contexts:
                    self.last_contexts.append(clipped)
                tool_calls_count += 1

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": clipped,
                })

        return (assistant_msg.content or "") if assistant_msg else "", (total_prompt, total_completion), tool_calls_count

    def _run_self_evaluation(
        self, user_question: str, response_text: str, trace: Any
    ) -> tuple[EvaluationScore, tuple[int, int]]:
        prompt = EVALUATOR_PROMPT.format(
            response_text=response_text,
            user_question=user_question,
        )

        span = self._tracer.start_generation(
            trace=trace,
            name="self-evaluation",
            model=self._model,
            input_messages=[{"role": "user", "content": prompt}],
        )

        try:
            eval_response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            self._tracer.end_generation(span=span, output=eval_response, usage=eval_response.usage)
            prompt_tokens = eval_response.usage.prompt_tokens if eval_response.usage else 0
            completion_tokens = eval_response.usage.completion_tokens if eval_response.usage else 0
            raw = eval_response.choices[0].message.content or ""
            raw = raw.strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            return EvaluationScore(**data), (prompt_tokens, completion_tokens)
        except Exception:  # noqa: BLE001
            return EvaluationScore(
                causal_depth=5.0,
                accuracy=5.0,
                completeness=5.0,
                conflict_handling=5.0,
                actionability=5.0,
                overall=5.0,
                justification="Evaluation failed — default score assigned.",
            ), (0, 0)

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
