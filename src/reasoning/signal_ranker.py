from __future__ import annotations

from src.data_loader import DataLoader
from src.models.types import (
    CausalChain,
    ConflictReport,
    ConfidenceComponents,
    ConfidenceScore,
    MarketTrend,
    PortfolioSummary,
)

_IMPACT_WEIGHT = {"HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}


class SignalRanker:
    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader

    def compute_confidence_score(
        self,
        portfolio_id: str,
        news_ids: list[str],
        causal_chains: list[CausalChain] | None = None,
        conflict_reports: list[ConflictReport] | None = None,
        market_trend: MarketTrend | None = None,
        portfolio_summary: PortfolioSummary | None = None,
    ) -> ConfidenceScore:
        causal_chains = causal_chains or []
        conflict_reports = conflict_reports or []

        portfolio = self._loader.get_portfolio(portfolio_id)
        portfolio_symbols = {h.symbol for h in portfolio.holdings.stocks} if portfolio else set()

        relevant = [a for nid in news_ids if (a := self._loader.get_news_by_id(nid))]
        high_impact = [a for a in relevant if a.impact_level == "HIGH"]
        if high_impact:
            news_strength = sum(abs(a.sentiment_score) for a in high_impact) / len(high_impact)
        elif relevant:
            news_strength = sum(abs(a.sentiment_score) for a in relevant) / len(relevant) * 0.7
        else:
            news_strength = 0.5

        bp_values = [c.total_portfolio_impact_bp for c in causal_chains]
        if bp_values:
            neg_count = sum(1 for bp in bp_values if bp < 0)
            pos_count = sum(1 for bp in bp_values if bp > 0)
            corroboration = max(neg_count, pos_count) / len(bp_values)
        else:
            corroboration = 0.5

        breadth = self._loader.get_market_breadth()
        if breadth and portfolio_summary:
            adr = breadth.advance_decline_ratio
            portfolio_down = portfolio_summary.day_pnl_percent < 0
            if portfolio_down and adr < 0.30:
                breadth_alignment = 1.0
            elif portfolio_down and adr < 0.50:
                breadth_alignment = 0.7
            elif portfolio_down and adr >= 0.50:
                breadth_alignment = 0.3
            elif not portfolio_down and adr >= 0.70:
                breadth_alignment = 1.0
            elif not portfolio_down and adr >= 0.50:
                breadth_alignment = 0.7
            else:
                breadth_alignment = 0.5
        else:
            breadth_alignment = 0.6

        mentioned_symbols: set[str] = set()
        for article in relevant:
            mentioned_symbols.update(article.entities.stocks)
        data_coverage = (
            len(portfolio_symbols & mentioned_symbols) / len(portfolio_symbols)
            if portfolio_symbols else 0.5
        )

        conflict_penalty = max(-0.30, -0.10 * len(conflict_reports))

        raw = (
            news_strength * 0.30
            + corroboration * 0.25
            + breadth_alignment * 0.20
            + data_coverage * 0.25
            + conflict_penalty
        )
        overall = max(0.0, min(1.0, raw))

        if overall > 0.70:
            interpretation = "HIGH"
            detail = "Strong corroborated signal with consistent news and market breadth alignment."
        elif overall > 0.40:
            interpretation = "MEDIUM"
            detail = "Moderate evidence — some conflicting signals or limited news coverage."
        else:
            interpretation = "LOW"
            detail = "Weak or conflicting signals — treat conclusions with caution."

        return ConfidenceScore(
            overall=round(overall, 3),
            components=ConfidenceComponents(
                news_strength=round(news_strength, 3),
                corroboration=round(corroboration, 3),
                breadth_alignment=round(breadth_alignment, 3),
                data_coverage=round(data_coverage, 3),
                conflict_penalty=round(conflict_penalty, 3),
            ),
            interpretation=interpretation,
            detail=detail,
        )
