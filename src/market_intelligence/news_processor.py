from __future__ import annotations

from typing import Optional

from src.data_loader import DataLoader
from src.models.types import NewsArticle

_IMPACT_WEIGHT = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


class NewsProcessor:
    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader

    def get_news_for_query(
        self,
        sectors: list[str] | None = None,
        stocks: list[str] | None = None,
        indices: list[str] | None = None,
        scope_filter: Optional[str] = None,
        impact_filter: Optional[str] = None,
        max_results: int = 10,
    ) -> list[NewsArticle]:
        all_news = self._loader.get_news()
        filtered: list[NewsArticle] = []
        has_filters = bool(sectors or stocks or indices)

        for article in all_news:
            if has_filters:
                match = (
                    (sectors and any(s in article.entities.sectors for s in sectors))
                    or (stocks and any(s in article.entities.stocks for s in stocks))
                    or (indices and any(i in article.entities.indices for i in indices))
                )
                if not match:
                    continue
            if scope_filter and article.scope != scope_filter:
                continue
            if impact_filter and article.impact_level != impact_filter:
                continue
            filtered.append(article)

        filtered.sort(key=lambda a: _IMPACT_WEIGHT[a.impact_level] * abs(a.sentiment_score), reverse=True)
        return filtered[:max_results]

    def get_conflict_articles(self) -> list[NewsArticle]:
        return [a for a in self._loader.get_news() if a.conflict_flag]

    def get_high_impact_news(self) -> list[NewsArticle]:
        return [a for a in self._loader.get_news() if a.impact_level == "HIGH"]

    def get_market_wide_news(self) -> list[NewsArticle]:
        return [a for a in self._loader.get_news() if a.scope == "MARKET_WIDE"]

    def build_news_digest(self) -> str:
        lines: list[str] = ["=== NEWS DIGEST ===\n"]
        for article in self._loader.get_news():
            entities_parts: list[str] = []
            if article.entities.sectors:
                entities_parts.append("sectors=" + ",".join(article.entities.sectors))
            if article.entities.stocks:
                entities_parts.append("stocks=" + ",".join(article.entities.stocks))
            if article.entities.indices:
                entities_parts.append("idx=" + ",".join(article.entities.indices))
            entities_str = " | ".join(entities_parts) if entities_parts else "market-wide"
            conflict_marker = " [CONFLICT]" if article.conflict_flag else ""
            lines.append(
                f"[{article.id}] {article.impact_level} | {article.sentiment_score:+.2f} | "
                f"{article.scope} | {entities_str}{conflict_marker}\n"
                f"  {article.headline}"
            )
            if article.conflict_explanation:
                lines.append(f"  CONFLICT: {article.conflict_explanation}")
            lines.append(f"  Causal: {' / '.join(article.causal_factors[:2])}")
            lines.append("")
        return "\n".join(lines)

    def get_news_for_portfolio_stocks(self, stock_symbols: list[str], max_results: int = 15) -> list[NewsArticle]:
        sectors_set: set[str] = set()
        for sym in stock_symbols:
            stock = self._loader.get_stock(sym)
            if stock:
                sectors_set.add(stock.sector)
        return self.get_news_for_query(sectors=list(sectors_set), stocks=stock_symbols, max_results=max_results)
