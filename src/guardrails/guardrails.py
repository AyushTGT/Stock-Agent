from __future__ import annotations

import logging
import os
import re

from src.models.types import EvaluationScore, GuardResult

logger = logging.getLogger(__name__)

_GUARANTEED_RETURNS = re.compile(
    r"guarante[ed]{1,2}\s+return|sure.{0,10}profit|100\s*%\s+return|risk.{0,10}free\s+profit",
    re.IGNORECASE,
)
_HARMFUL = re.compile(
    r"insider.{0,15}tip|pump.{0,10}dump|manipulate.{0,15}(stock|market)|front.{0,5}run",
    re.IGNORECASE,
)
_OUT_OF_SCOPE_KEYWORDS = {
    "cryptocurrency", "crypto", "bitcoin", "ethereum", "forex", "fx trading",
    "real estate", "insurance", "fixed deposit", "fd rate", "options trading",
    "futures trading", "commodities", "gold etf", "nft",
}

_REFUSAL_TEMPLATE = (
    "I'm a financial advisor for Indian equities and mutual funds on NSE/BSE. "
    "I can analyse your portfolio holdings, market trends, sector movements, and news impact. "
    "I'm not able to advise on {category}. "
    "What portfolio or market question can I help with?"
)

_DISCLAIMER = "\n\n⚠️ Note: Past performance is not a guarantee of future results."

_RESPONSE_VIOLATES = re.compile(
    r"guarante[ed]{1,2}|certain profit|sure to rise|will definitely",
    re.IGNORECASE,
)


def _default_eval() -> EvaluationScore:
    return EvaluationScore(
        causal_depth=0.0,
        accuracy=0.0,
        completeness=0.0,
        conflict_handling=0.0,
        actionability=0.0,
        overall=0.0,
        justification="Guardrail triggered — query blocked.",
    )


class InputGuard:
    def check(self, query: str) -> GuardResult:
        if _GUARANTEED_RETURNS.search(query):
            return GuardResult(
                allowed=False,
                category="guaranteed return claims",
                reason="Query requests or implies guaranteed investment returns.",
            )
        if _HARMFUL.search(query):
            return GuardResult(
                allowed=False,
                category="market manipulation or insider trading",
                reason="Query references illegal or harmful financial activity.",
            )
        lower = query.lower()
        for kw in _OUT_OF_SCOPE_KEYWORDS:
            if kw in lower:
                return GuardResult(
                    allowed=False,
                    category=kw,
                    reason=f"Topic '{kw}' is outside the scope of this advisor.",
                )

        if len(query.split()) > 15:
            llm_result = self._llm_classify(query)
            if llm_result is not None:
                return llm_result

        return GuardResult(allowed=True, category="OK", reason="")

    def _llm_classify(self, query: str) -> GuardResult | None:
        print("[DEBUG] guardrail LLM classify called...", flush=True)
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_key:
            return None
        try:
            import openai
            client = openai.OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1", timeout=10.0, max_retries=0)
            prompt = (
                "Classify this user query for an Indian stock market advisor into exactly one category:\n"
                "FINANCE_INDIA — valid question about NSE/BSE stocks, mutual funds, sectors, portfolio, market news\n"
                "OUT_OF_SCOPE — crypto, forex, real estate, insurance, international markets\n"
                "HARMFUL — insider trading, market manipulation, pump-and-dump\n"
                "AMBIGUOUS — unclear intent\n\n"
                f"Query: {query[:400]}\n\nRespond with one word only."
            )
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=5,
                messages=[{"role": "user", "content": prompt}],
            )
            label = (response.choices[0].message.content or "AMBIGUOUS").strip().upper()
            if label == "OUT_OF_SCOPE":
                return GuardResult(allowed=False, category="out-of-scope topic", reason="LLM classifier: out of scope")
            if label == "HARMFUL":
                return GuardResult(allowed=False, category="harmful request", reason="LLM classifier: harmful")
        except Exception as exc:
            logger.debug("LLM guard classify failed: %s", exc)
        return None


class OutputGuard:
    def check(self, response: str, is_portfolio_query: bool = False) -> str:
        if _RESPONSE_VIOLATES.search(response):
            match = _RESPONSE_VIOLATES.search(response)
            truncated = response[:match.start()] if match else response
            return truncated + _DISCLAIMER
        return response


def refusal_response(category: str) -> str:
    return _REFUSAL_TEMPLATE.format(category=category)


def default_eval_for_guardrail() -> EvaluationScore:
    return _default_eval()
