from __future__ import annotations

from collections import Counter
from typing import Literal

from src.data_loader import DataLoader
from src.models.types import Portfolio, RiskAssessment, RiskFlag
from src.portfolio_analytics.allocation_analyzer import AllocationAnalyzer

_SECTOR_THRESHOLDS: list[tuple[float, Literal["CRITICAL", "HIGH", "MEDIUM"]]] = [
    (60.0, "CRITICAL"),
    (40.0, "HIGH"),
    (25.0, "MEDIUM"),
]
_STOCK_CONCENTRATION_THRESHOLD = 20.0
_HIGH_BETA_THRESHOLD = 1.3
_MF_OVERLAP_MIN_FUNDS = 3


class RiskDetector:
    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self._alloc = AllocationAnalyzer(loader)

    def get_risk_assessment(self, portfolio_id: str) -> RiskAssessment | None:
        portfolio = self._loader.get_portfolio(portfolio_id)
        if not portfolio:
            return None

        flags: list[RiskFlag] = []
        flags.extend(self._detect_sector_concentration(self._alloc.get_sector_exposure(portfolio_id)))
        flags.extend(self._detect_stock_concentration(portfolio))

        portfolio_beta = self._compute_weighted_beta(portfolio)
        if portfolio_beta > _HIGH_BETA_THRESHOLD:
            flags.append(RiskFlag(
                risk_type="HIGH_BETA",
                severity="MEDIUM",
                subject=f"Portfolio Beta: {portfolio_beta:.2f}",
                current_weight=portfolio_beta,
                threshold=_HIGH_BETA_THRESHOLD,
                explanation=(
                    f"Portfolio weighted beta of {portfolio_beta:.2f} indicates "
                    f"high market sensitivity — expect amplified swings vs NIFTY50."
                ),
            ))

        flags.extend(self._detect_mf_overlap(portfolio))

        severities = [f.severity for f in flags]
        if "CRITICAL" in severities:
            overall = "CRITICAL"
        elif "HIGH" in severities:
            overall = "HIGH"
        elif "MEDIUM" in severities:
            overall = "MEDIUM"
        else:
            overall = "LOW"

        if not flags:
            summary = "Portfolio is well-diversified with no critical risk flags."
        else:
            critical_flags = [f for f in flags if f.severity == "CRITICAL"]
            if critical_flags:
                summary = f"CRITICAL RISK: {critical_flags[0].explanation} Total {len(flags)} risk flag(s) detected."
            else:
                summary = f"{len(flags)} risk flag(s) detected. Review concentration and beta metrics."

        return RiskAssessment(
            overall_risk=overall,
            portfolio_beta=round(portfolio_beta, 3),
            flags=flags,
            summary=summary,
        )

    def _detect_sector_concentration(self, sector_exposure: dict[str, float]) -> list[RiskFlag]:
        flags: list[RiskFlag] = []
        for sector, weight in sector_exposure.items():
            for threshold, severity in _SECTOR_THRESHOLDS:
                if weight >= threshold:
                    flags.append(RiskFlag(
                        risk_type="CONCENTRATION_SECTOR",
                        severity=severity,
                        subject=sector,
                        current_weight=round(weight, 2),
                        threshold=threshold,
                        explanation=(
                            f"{sector} sector exposure is {weight:.1f}% "
                            f"(threshold: {threshold:.0f}%). "
                            f"This creates concentrated downside risk if sector underperforms."
                        ),
                    ))
                    break
        return flags

    def _detect_stock_concentration(self, portfolio: Portfolio) -> list[RiskFlag]:
        flags: list[RiskFlag] = []
        for holding in portfolio.holdings.stocks:
            if holding.weight_in_portfolio >= _STOCK_CONCENTRATION_THRESHOLD:
                flags.append(RiskFlag(
                    risk_type="CONCENTRATION_STOCK",
                    severity="HIGH",
                    subject=holding.symbol,
                    current_weight=round(holding.weight_in_portfolio, 2),
                    threshold=_STOCK_CONCENTRATION_THRESHOLD,
                    explanation=(
                        f"{holding.symbol} represents {holding.weight_in_portfolio:.1f}% "
                        f"of the portfolio — single-stock concentration risk."
                    ),
                ))
        return flags

    def _compute_weighted_beta(self, portfolio: Portfolio) -> float:
        total_stock_value = 0.0
        weighted_beta_sum = 0.0
        for holding in portfolio.holdings.stocks:
            stock = self._loader.get_stock(holding.symbol)
            if not stock:
                continue
            value = stock.current_price * holding.quantity
            total_stock_value += value
            weighted_beta_sum += stock.beta * value
        return weighted_beta_sum / total_stock_value if total_stock_value > 0 else 1.0

    def _detect_mf_overlap(self, portfolio: Portfolio) -> list[RiskFlag]:
        """Flag stocks that appear as top holdings in 3+ funds simultaneously."""
        symbol_fund_count: Counter = Counter()
        for mf_holding in portfolio.holdings.mutual_funds:
            scheme = self._loader.get_mutual_fund(mf_holding.scheme_code)
            if not scheme:
                continue
            for top in scheme.top_holdings:
                symbol_fund_count[top.symbol] += 1

        flags: list[RiskFlag] = []
        for symbol, count in symbol_fund_count.items():
            if count >= _MF_OVERLAP_MIN_FUNDS:
                flags.append(RiskFlag(
                    risk_type="MF_OVERLAP",
                    severity="LOW",
                    subject=symbol,
                    current_weight=float(count),
                    threshold=float(_MF_OVERLAP_MIN_FUNDS),
                    explanation=(
                        f"{symbol} appears as a top holding in {count} of your mutual funds, "
                        f"creating hidden concentration risk beyond direct stock holdings."
                    ),
                ))
        return flags
