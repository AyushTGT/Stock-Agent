from __future__ import annotations

from src.data_loader import DataLoader
from src.models.types import ConflictReport

_DIVERGENCE_THRESHOLD = 1.5


class ConflictResolver:
    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader

    def get_conflict_analysis(
        self, portfolio_id: str, news_ids: list[str] | None = None
    ) -> list[ConflictReport]:
        portfolio = self._loader.get_portfolio(portfolio_id)
        if not portfolio:
            return []

        portfolio_symbols = {h.symbol for h in portfolio.holdings.stocks}
        portfolio_sectors = {h.sector for h in portfolio.holdings.stocks}

        reports: list[ConflictReport] = []
        reports.extend(self._detect_news_conflicts(portfolio_symbols, news_ids))
        reports.extend(self._detect_sector_stock_divergence(portfolio_symbols, portfolio_sectors))
        reports.extend(self._detect_mixed_signals(portfolio_symbols, portfolio_sectors))

        seen: set[tuple] = set()
        unique: list[ConflictReport] = []
        for r in reports:
            key = (r.conflict_type, r.stock_symbol)
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique

    def _detect_news_conflicts(
        self, portfolio_symbols: set[str], news_ids: list[str] | None
    ) -> list[ConflictReport]:
        reports: list[ConflictReport] = []
        if news_ids:
            articles = [a for nid in news_ids if (a := self._loader.get_news_by_id(nid)) and a.conflict_flag]
        else:
            articles = [a for a in self._loader.get_news() if a.conflict_flag]

        for article in articles:
            relevant_stocks = set(article.entities.stocks) & portfolio_symbols
            if not relevant_stocks and not article.entities.sectors:
                relevant_stocks = {
                    h.symbol
                    for h in self._loader.get_portfolio(next(iter(portfolio_symbols), "")).holdings.stocks
                    if h.sector in article.entities.sectors
                } if portfolio_symbols else set()

            for symbol in relevant_stocks or {"PORTFOLIO"}:
                stock = self._loader.get_stock(symbol) if symbol != "PORTFOLIO" else None
                stock_move = stock.change_percent if stock else 0.0
                sector = stock.sector if stock else (article.entities.sectors[0] if article.entities.sectors else "")
                sector_perf = self._loader.get_sector_performance(sector)
                sector_move = sector_perf.change_percent if sector_perf else 0.0

                if article.sentiment_score > 0.2 and stock_move < 0:
                    conflict_type = "POSITIVE_NEWS_STOCK_FALLING"
                    explanation = (
                        f"Despite {article.headline[:60]}... (positive news, score: {article.sentiment_score:+.2f}), "
                        f"{symbol} fell {stock_move:.2f}%."
                    )
                    if article.conflict_explanation:
                        resolution = article.conflict_explanation
                    elif sector_move < -0.5:
                        resolution = (
                            f"Sector-wide headwinds ({sector} {sector_move:+.2f}%) are overpowering "
                            f"company-specific positives. Macro sentiment dominates micro news in risk-off conditions."
                        )
                    elif sector_move > 0.5:
                        resolution = (
                            f"Despite positive company news and a rising sector ({sector} {sector_move:+.2f}%), "
                            f"{symbol} is falling — likely a stock-specific negative catalyst or profit-taking "
                            f"after recent outperformance."
                        )
                    else:
                        resolution = (
                            f"Sector is near-flat ({sector} {sector_move:+.2f}%); stock weakness is driven by "
                            f"broader market risk-off sentiment rather than sector or company fundamentals."
                        )
                elif article.sentiment_score < -0.2 and stock_move > 0:
                    conflict_type = "NEGATIVE_NEWS_STOCK_RISING"
                    explanation = (
                        f"Despite {article.headline[:60]}... (negative news, score: {article.sentiment_score:+.2f}), "
                        f"{symbol} rose {stock_move:.2f}%."
                    )
                    if article.conflict_explanation:
                        resolution = article.conflict_explanation
                    elif sector_move > 0.5:
                        resolution = (
                            f"Sector-wide strength ({sector} {sector_move:+.2f}%) is lifting {symbol} despite "
                            f"company-specific negative news. Monitor for a delayed re-rating once the sector "
                            f"tailwind fades."
                        )
                    else:
                        resolution = (
                            f"Defensive buying or value-seeking behaviour overriding negative news. "
                            f"Possible short-covering or technical support at current levels."
                        )
                else:
                    continue

                reports.append(ConflictReport(
                    conflict_id=f"CONFLICT-{article.id}-{symbol}",
                    conflict_type=conflict_type,
                    news_id=article.id,
                    stock_symbol=symbol,
                    news_sentiment=article.sentiment,
                    stock_move=stock_move,
                    sector_move=sector_move,
                    explanation=explanation,
                    resolution=resolution,
                ))

        return reports

    def _detect_sector_stock_divergence(
        self, portfolio_symbols: set[str], portfolio_sectors: set[str]
    ) -> list[ConflictReport]:
        reports: list[ConflictReport] = []

        for symbol in portfolio_symbols:
            stock = self._loader.get_stock(symbol)
            if not stock:
                continue
            sector_perf = self._loader.get_sector_performance(stock.sector)
            if not sector_perf:
                continue

            stock_move = stock.change_percent
            sector_move = sector_perf.change_percent
            divergence = stock_move - sector_move

            if abs(divergence) < _DIVERGENCE_THRESHOLD:
                continue
            if (stock_move >= 0) == (sector_move >= 0):
                continue

            if stock_move > 0:
                explanation = (
                    f"{symbol} is +{stock_move:.2f}% while its sector ({stock.sector}) "
                    f"is {sector_move:.2f}% — stock outperforming the sector trend."
                )
                resolution = (
                    f"Stock-specific catalyst (e.g., earnings, deal win, product launch) "
                    f"overriding broader sector weakness. Check company-specific news for {symbol}."
                )
            else:
                explanation = (
                    f"{symbol} is {stock_move:.2f}% while its sector ({stock.sector}) "
                    f"is +{sector_move:.2f}% — stock underperforming sector rally."
                )
                resolution = (
                    f"Stock-specific negative catalyst (e.g., results miss, governance issue, downgrade) "
                    f"dragging {symbol} despite sector tailwinds."
                )

            reports.append(ConflictReport(
                conflict_id=f"DIV-{symbol}-{stock.sector}",
                conflict_type="SECTOR_STOCK_DIVERGENCE",
                news_id=None,
                stock_symbol=symbol,
                news_sentiment=None,
                stock_move=stock_move,
                sector_move=sector_move,
                explanation=explanation,
                resolution=resolution,
            ))

        return reports

    def _detect_mixed_signals(
        self, portfolio_symbols: set[str], portfolio_sectors: set[str]
    ) -> list[ConflictReport]:
        reports: list[ConflictReport] = []
        entity_articles: dict[str, list] = {}

        for article in self._loader.get_news():
            for sym in article.entities.stocks:
                if sym in portfolio_symbols:
                    entity_articles.setdefault(sym, []).append(article)
            for sector in article.entities.sectors:
                if sector in portfolio_sectors:
                    entity_articles.setdefault(f"SECTOR:{sector}", []).append(article)

        for entity, articles in entity_articles.items():
            if len(articles) < 2:
                continue
            positives = [a for a in articles if a.sentiment_score > 0.2]
            negatives = [a for a in articles if a.sentiment_score < -0.2]
            if not positives or not negatives:
                continue

            symbol = entity if not entity.startswith("SECTOR:") else "SECTOR"
            stock = self._loader.get_stock(symbol) if symbol != "SECTOR" else None
            stock_move = stock.change_percent if stock else 0.0
            sector_name = entity.replace("SECTOR:", "") if entity.startswith("SECTOR:") else (stock.sector if stock else "")
            sector_perf = self._loader.get_sector_performance(sector_name)
            sector_move = sector_perf.change_percent if sector_perf else 0.0
            net_sentiment = sum(a.sentiment_score for a in articles) / len(articles)

            reports.append(ConflictReport(
                conflict_id=f"MIXED-{entity}",
                conflict_type="MIXED_SIGNALS",
                news_id=None,
                stock_symbol=symbol,
                news_sentiment="MIXED",
                stock_move=stock_move,
                sector_move=sector_move,
                explanation=(
                    f"Conflicting signals for {entity}: "
                    f"Positive: '{positives[0].headline[:50]}...' vs "
                    f"Negative: '{negatives[0].headline[:50]}...'"
                ),
                resolution=(
                    f"Net sentiment score: {net_sentiment:+.2f}. "
                    f"{'Positive bias overall — stock likely to hold or recover.' if net_sentiment > 0 else 'Negative bias overall — stock under pressure.'}"
                ),
            ))

        return reports
