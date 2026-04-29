from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class RawNews:
    headline: str
    summary: str
    published_at: str
    source: str
    url: str
    entities_hint: list[str] = field(default_factory=list)


def _deduplicate(articles: list[RawNews], threshold: float = 0.80) -> list[RawNews]:
    seen: list[RawNews] = []
    for article in articles:
        duplicate = any(
            SequenceMatcher(None, article.headline.lower(), s.headline.lower()).ratio() > threshold
            for s in seen
        )
        if not duplicate:
            seen.append(article)
    return seen


def fetch_yahoo_news(symbols: list[str]) -> list[RawNews]:
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — skipping Yahoo news fetch")
        return []

    articles: list[RawNews] = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            for item in ticker.news or []:
                content = item.get("content") or {}
                title = content.get("title") or item.get("title") or ""
                if not title:
                    continue
                summary = content.get("summary") or content.get("description") or ""
                pub_date = content.get("pubDate") or ""
                provider = (content.get("provider") or {}).get("displayName") or "Yahoo Finance"
                canonical = (content.get("canonicalUrl") or {}).get("url") or ""
                articles.append(RawNews(
                    headline=title,
                    summary=summary,
                    published_at=pub_date,
                    source=provider,
                    url=canonical,
                    entities_hint=[symbol],
                ))
        except Exception as exc:
            logger.debug("Yahoo news failed for %s: %s", symbol, exc)

    return _deduplicate(articles)


def fetch_newsapi_headlines(query: str = "NSE OR NIFTY OR RBI OR Indian stock market") -> list[RawNews]:
    api_key = os.environ.get("NEWS_API_KEY", "")
    if not api_key:
        return []

    try:
        from newsapi import NewsApiClient
    except ImportError:
        logger.warning("newsapi-python not installed — skipping NewsAPI fetch")
        return []

    try:
        client = NewsApiClient(api_key=api_key)
        from_date = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
        response = client.get_everything(
            q=query,
            language="en",
            sort_by="publishedAt",
            page_size=50,
            from_param=from_date,
        )
        articles: list[RawNews] = []
        for item in response.get("articles") or []:
            headline = item.get("title") or ""
            if not headline or headline == "[Removed]":
                continue
            articles.append(RawNews(
                headline=headline,
                summary=item.get("description") or item.get("content") or "",
                published_at=item.get("publishedAt") or "",
                source=(item.get("source") or {}).get("name") or "NewsAPI",
                url=item.get("url") or "",
                entities_hint=[],
            ))
        return articles
    except Exception as exc:
        logger.warning("NewsAPI fetch failed: %s", exc)
        return []
