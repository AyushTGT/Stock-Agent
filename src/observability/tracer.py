from __future__ import annotations

import atexit
import os
from typing import Any


def _langfuse_available() -> bool:
    return bool(os.environ.get("LANGFUSE_SECRET_KEY"))


class _NoOpContext:
    def __getattr__(self, _name: str):
        return lambda *args, **kwargs: self


class LangfuseTracer:
    def __init__(self) -> None:
        self._enabled = _langfuse_available()
        self._lf: Any = None

        if self._enabled:
            try:
                from langfuse import Langfuse  # type: ignore

                self._lf = Langfuse(
                    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
                    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                )
                atexit.register(self._lf.flush)
            except Exception:  # noqa: BLE001
                self._enabled = False

    def create_trace(self, *, name: str, user_id: str, input: str) -> Any:
        if not self._enabled or self._lf is None:
            return _NoOpContext()
        try:
            return self._lf.trace(name=name, user_id=user_id, input=input)
        except Exception:  # noqa: BLE001
            return _NoOpContext()

    def finalize_trace(self, *, trace: Any, output: str, score_value: float, score_comment: str) -> None:
        if not self._enabled or isinstance(trace, _NoOpContext):
            return
        try:
            trace.update(output=output)
            self._lf.score(trace_id=trace.id, name="overall_quality", value=score_value, comment=score_comment)
            self._lf.flush()
        except Exception:  # noqa: BLE001
            pass

    def start_generation(self, *, trace: Any, name: str, model: str, input_messages: list[dict]) -> Any:
        if not self._enabled or isinstance(trace, _NoOpContext):
            return _NoOpContext()
        try:
            return trace.generation(name=name, model=model, input=input_messages)
        except Exception:  # noqa: BLE001
            return _NoOpContext()

    def end_generation(self, *, span: Any, output: Any, usage: Any = None) -> None:
        if not self._enabled or isinstance(span, _NoOpContext):
            return
        try:
            update_kwargs: dict = {"output": str(output)}
            if usage is not None:
                update_kwargs["usage"] = {
                    "input": getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)),
                    "output": getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0)),
                }
            span.end(**update_kwargs)
        except Exception:  # noqa: BLE001
            pass

    def start_span(self, *, trace: Any, name: str, input: Any) -> Any:
        if not self._enabled or isinstance(trace, _NoOpContext):
            return _NoOpContext()
        try:
            return trace.span(name=name, input=input)
        except Exception:  # noqa: BLE001
            return _NoOpContext()

    def end_span(self, *, span: Any, output: Any) -> None:
        if not self._enabled or isinstance(span, _NoOpContext):
            return
        try:
            span.end(output=output)
        except Exception:  # noqa: BLE001
            pass

    def flush(self) -> None:
        if self._enabled and self._lf is not None:
            try:
                self._lf.flush()
            except Exception:  # noqa: BLE001
                pass
