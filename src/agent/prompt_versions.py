from __future__ import annotations

from typing import Literal

from src.agent.prompts import SYSTEM_PROMPT_TEMPLATE

PROMPT_V1 = """\
You are a financial advisor for Indian stock market portfolios. \
Help users understand their portfolio performance and market conditions. \
Use the available tools to get data. Be helpful and accurate.
"""

PROMPT_V2 = SYSTEM_PROMPT_TEMPLATE

PromptVersion = Literal["v1", "v2"]

PROMPT_MAP: dict[PromptVersion, str] = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
}
