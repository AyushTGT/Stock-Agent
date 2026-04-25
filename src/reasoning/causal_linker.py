from __future__ import annotations

from src.data_loader import DataLoader
from src.models.types import CausalChain, CausalLink, StockImpact


class CausalLinker:
    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader

    def get_causal_chain(self, portfolio_id: str, news_ids: list[str]) -> list[CausalChain]:
        portfolio = self._loader.get_portfolio(portfolio_id)
        if not portfolio:
            return []

        total_value = self._compute_total_value(portfolio)
        if total_value == 0:
            return []

        chains: list[CausalChain] = []

        for news_id in news_ids:
            article = self._loader.get_news_by_id(news_id)
            if not article:
                continue

            links: list[CausalLink] = []

            if article.scope == "STOCK_SPECIFIC":
                link = self._build_stock_direct_link(article, portfolio, total_value)
                if link:
                    links.append(link)
            else:
                affected_sectors = (
                    list(self._loader.get_all_sector_performances().keys())
                    if article.scope == "MARKET_WIDE"
                    else article.entities.sectors
                )
                for sector in affected_sectors:
                    link = self._build_sector_link(article, sector, portfolio, total_value)
                    if link and link.portfolio_stocks_impacted:
                        links.append(link)

            if not links:
                continue

            total_bp = sum(lnk.portfolio_contribution_bp for lnk in links)
            chains.append(
                CausalChain(
                    chain_id=f"CHAIN-{news_id}",
                    narrative=self._build_narrative(article, links, total_bp),
                    links=links,
                    total_portfolio_impact_bp=round(total_bp, 2),
                    total_portfolio_impact_percent=round(total_bp / 100, 4),
                )
            )

        chains.sort(key=lambda c: abs(c.total_portfolio_impact_bp), reverse=True)
        return chains

    def build_full_chain_for_portfolio(self, portfolio_id: str) -> list[CausalChain]:
        portfolio = self._loader.get_portfolio(portfolio_id)
        if not portfolio:
            return []

        portfolio_sectors = {h.sector for h in portfolio.holdings.stocks}
        portfolio_symbols = {h.symbol for h in portfolio.holdings.stocks}

        scored: list[tuple[float, str]] = []
        for article in self._loader.get_news():
            if article.impact_level == "LOW":
                continue
            score = {"HIGH": 3.0, "MEDIUM": 1.5}.get(article.impact_level, 1.0) * abs(article.sentiment_score)
            score += len(set(article.entities.sectors) & portfolio_sectors) * 2.0
            score += len(set(article.entities.stocks) & portfolio_symbols) * 3.0
            if article.scope == "MARKET_WIDE":
                score += 1.0
            scored.append((score, article.id))

        scored.sort(reverse=True)
        return self.get_causal_chain(portfolio_id, [nid for _, nid in scored[:7]])

    def _build_stock_direct_link(self, article, portfolio, total_value: float) -> CausalLink | None:
        impacted: list[StockImpact] = []
        for holding in portfolio.holdings.stocks:
            if holding.symbol not in article.entities.stocks:
                continue
            stock = self._loader.get_stock(holding.symbol)
            if not stock:
                continue
            weight = stock.current_price * holding.quantity / total_value * 100
            impacted.append(StockImpact(
                symbol=holding.symbol,
                portfolio_weight=round(weight, 3),
                stock_change_percent=stock.change_percent,
                contribution_bp=round(weight / 100 * stock.change_percent * 100, 2),
            ))

        if not impacted:
            return None

        first_sym = article.entities.stocks[0] if article.entities.stocks else ""
        first_stock = self._loader.get_stock(first_sym)
        sector = first_stock.sector if first_stock else "STOCK_SPECIFIC"
        sector_perf = self._loader.get_sector_performance(sector)

        total_contribution = sum(s.contribution_bp for s in impacted)
        return CausalLink(
            news_id=article.id,
            headline=article.headline,
            sentiment_score=article.sentiment_score,
            impact_level=article.impact_level,
            scope=article.scope,
            causal_factors=article.causal_factors,
            affected_sector=sector,
            sector_change_percent=sector_perf.change_percent if sector_perf else 0.0,
            portfolio_stocks_impacted=impacted,
            portfolio_contribution_bp=round(total_contribution, 2),
        )

    def _build_sector_link(self, article, sector: str, portfolio, total_value: float) -> CausalLink | None:
        sector_perf = self._loader.get_sector_performance(sector)
        sector_change = sector_perf.change_percent if sector_perf else 0.0

        impacted: list[StockImpact] = []
        for holding in portfolio.holdings.stocks:
            if holding.sector != sector:
                continue
            stock = self._loader.get_stock(holding.symbol)
            if not stock:
                continue
            weight = stock.current_price * holding.quantity / total_value * 100
            impacted.append(StockImpact(
                symbol=holding.symbol,
                portfolio_weight=round(weight, 3),
                stock_change_percent=stock.change_percent,
                contribution_bp=round(weight / 100 * stock.change_percent * 100, 2),
            ))

        if not impacted:
            return None

        total_contribution = sum(s.contribution_bp for s in impacted)
        return CausalLink(
            news_id=article.id,
            headline=article.headline,
            sentiment_score=article.sentiment_score,
            impact_level=article.impact_level,
            scope=article.scope,
            causal_factors=article.causal_factors,
            affected_sector=sector,
            sector_change_percent=sector_change,
            portfolio_stocks_impacted=impacted,
            portfolio_contribution_bp=round(total_contribution, 2),
        )

    def _build_narrative(self, article, links: list[CausalLink], total_bp: float) -> str:
        affected_sectors = list({lnk.affected_sector for lnk in links if lnk.affected_sector})
        total_stocks = len({s.symbol for s in (s for lnk in links for s in lnk.portfolio_stocks_impacted)})
        direction = "contributed" if total_bp >= 0 else "dragged"
        sector_str = " + ".join(affected_sectors[:3]) if affected_sectors else "portfolio"
        return (
            f"{article.headline[:80]} "
            f"(sentiment: {article.sentiment_score:+.2f}, {article.impact_level}) → "
            f"{sector_str} sector(s) → "
            f"{total_stocks} portfolio stock(s) → "
            f"{direction} {abs(total_bp):.1f}bp to portfolio"
        )

    def _compute_total_value(self, portfolio) -> float:
        total = 0.0
        for holding in portfolio.holdings.stocks:
            stock = self._loader.get_stock(holding.symbol)
            total += (stock.current_price if stock else holding.current_price) * holding.quantity
        for mf_holding in portfolio.holdings.mutual_funds:
            scheme = self._loader.get_mutual_fund(mf_holding.scheme_code)
            total += (scheme.current_nav if scheme else mf_holding.current_nav) * mf_holding.units
        return total
