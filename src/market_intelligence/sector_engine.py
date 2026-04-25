from __future__ import annotations

from src.data_loader import DataLoader
from src.models.types import Portfolio, SectorAnalysis


class SectorEngine:
    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader

    def get_sector_analysis(self, sectors: list[str]) -> list[SectorAnalysis]:
        return [a for s in sectors if (a := self._analyze_sector(s))]

    def get_sectors_for_portfolio(self, portfolio: Portfolio) -> list[str]:
        sectors: set[str] = set()
        for holding in portfolio.holdings.stocks:
            sectors.add(holding.sector)
        for mf_holding in portfolio.holdings.mutual_funds:
            mf = self._loader.get_mutual_fund(mf_holding.scheme_code)
            if mf:
                for sector_name in mf.sector_allocation:
                    canonical = self._canonicalize_sector(sector_name)
                    if canonical:
                        sectors.add(canonical)
        return sorted(sectors)

    def get_all_sector_analyses(self) -> list[SectorAnalysis]:
        return self.get_sector_analysis(list(self._loader.get_all_sector_performances().keys()))

    def _analyze_sector(self, sector: str) -> SectorAnalysis | None:
        perf = self._loader.get_sector_performance(sector)
        weekly = self._loader.get_sector_weekly(sector)
        sector_info = self._loader.get_sector_info(sector)

        news_scores: list[float] = []
        news_count = 0
        for article in self._loader.get_news():
            if sector in article.entities.sectors:
                news_scores.append(article.sentiment_score)
                news_count += 1

        top_stocks: list[str] = []
        if sector_info:
            top_stocks = [sym for sym in sector_info.stocks[:8] if self._loader.get_stock(sym)]

        return SectorAnalysis(
            sector=sector,
            current_sentiment=perf.sentiment if perf else "NEUTRAL",
            today_change_percent=perf.change_percent if perf else 0.0,
            weekly_return=weekly.weekly_change_percent if weekly else 0.0,
            weekly_trend=weekly.trend if weekly else "SIDEWAYS",
            news_sentiment_avg=round(sum(news_scores) / len(news_scores), 3) if news_scores else 0.0,
            news_count=news_count,
            key_drivers=perf.key_drivers if perf else [],
            top_stocks=top_stocks,
            catalyst=weekly.catalyst if weekly else "",
        )

    def _canonicalize_sector(self, raw_sector: str) -> str | None:
        mapping = {
            "BANKING": "BANKING",
            "INFORMATION_TECHNOLOGY": "INFORMATION_TECHNOLOGY",
            "PHARMACEUTICALS": "PHARMACEUTICALS",
            "HEALTHCARE": "PHARMACEUTICALS",
            "FMCG": "FMCG",
            "CONSUMER_DISCRETIONARY": "FMCG",
            "ENERGY": "ENERGY",
            "METALS": "METALS",
            "REALTY": "REALTY",
            "INFRASTRUCTURE": "INFRASTRUCTURE",
            "INDUSTRIALS": "INFRASTRUCTURE",
            "FINANCIAL_SERVICES": "FINANCIAL_SERVICES",
            "AUTOMOBILE": "AUTOMOBILE",
        }
        return mapping.get(raw_sector.upper())
