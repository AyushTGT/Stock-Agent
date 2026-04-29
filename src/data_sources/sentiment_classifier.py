from __future__ import annotations

import json
import logging
import os
import re

from src.models.types import NewsEntities

logger = logging.getLogger(__name__)

_SECTOR_KEYWORDS: dict[str, list[str]] = {
    "BANKING": ["bank", "rbi", "repo rate", "credit", "lending", "npa", "hdfc", "icici", "sbi", "kotak"],
    "INFORMATION_TECHNOLOGY": ["it", "software", "tech", "infosys", "tcs", "wipro", "hcl", "rupee", "dollar"],
    "PHARMACEUTICALS": ["pharma", "drug", "fda", "usfda", "cipla", "sun pharma", "divi"],
    "FMCG": ["fmcg", "consumer", "hul", "nestle", "dabur", "britannia", "volume growth"],
    "ENERGY": ["oil", "gas", "opec", "reliance", "ongc", "energy", "fuel"],
    "AUTOMOBILES": ["auto", "ev", "vehicle", "maruti", "tata motors", "bajaj", "mahindra"],
    "REAL_ESTATE": ["real estate", "housing", "property", "dlf", "lodha"],
    "METALS": ["steel", "metal", "aluminium", "tata steel", "jsw", "hindalco"],
    "TELECOM": ["telecom", "jio", "airtel", "vi", "bsnl", "spectrum"],
    "INFRASTRUCTURE": ["infra", "roads", "l&t", "adani", "construction"],
}

_AMBIGUITY_THRESHOLD = 0.15


def classify(text: str) -> tuple[str, float]:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]
    except ImportError:
        compound = 0.0

    if compound > _AMBIGUITY_THRESHOLD:
        return "POSITIVE", round(compound, 3)
    if compound < -_AMBIGUITY_THRESHOLD:
        return "NEGATIVE", round(compound, 3)

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key and len(text.split()) > 15:
        try:
            import openai
            client = openai.OpenAI(
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1",
            )
            prompt = (
                f'Classify the sentiment of this financial news headline as exactly one of: '
                f'POSITIVE, NEGATIVE, MIXED, NEUTRAL. Reply with only the word.\n\nHeadline: {text[:300]}'
            )
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=5,
                messages=[{"role": "user", "content": prompt}],
            )
            label = (response.choices[0].message.content or "NEUTRAL").strip().upper()
            if label in ("POSITIVE", "NEGATIVE", "MIXED", "NEUTRAL"):
                score_map = {"POSITIVE": 0.3, "NEGATIVE": -0.3, "MIXED": 0.05, "NEUTRAL": 0.0}
                return label, score_map[label]
        except Exception as exc:
            logger.debug("Groq sentiment classification failed: %s", exc)

    return "NEUTRAL", round(compound, 3)


def infer_scope(headline: str, entities: NewsEntities) -> str:
    if entities.stocks:
        return "STOCK_SPECIFIC"
    if entities.sectors:
        return "SECTOR_SPECIFIC"
    return "MARKET_WIDE"


def infer_entities(headline: str, known_symbols: list[str] | None = None) -> NewsEntities:
    lower = headline.lower()
    found_stocks: list[str] = []
    found_sectors: list[str] = []
    found_indices: list[str] = []

    if known_symbols:
        for sym in known_symbols:
            if sym.lower() in lower or sym.replace(".", "").lower() in lower:
                found_stocks.append(sym)

    for sector, keywords in _SECTOR_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            found_sectors.append(sector)

    for idx in ("NIFTY", "SENSEX", "BANKNIFTY", "NIFTYIT"):
        if idx.lower() in lower:
            found_indices.append(idx)

    return NewsEntities(
        sectors=list(dict.fromkeys(found_sectors)),
        stocks=list(dict.fromkeys(found_stocks)),
        indices=list(dict.fromkeys(found_indices)),
    )
