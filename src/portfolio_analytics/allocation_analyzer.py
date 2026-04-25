from __future__ import annotations

from src.data_loader import DataLoader
from src.models.types import AllocationBreakdown, Portfolio

_EQUITY_CATEGORIES = {
    "FLEXI_CAP", "LARGE_CAP", "MID_CAP", "SMALL_CAP",
    "SECTORAL_IT", "SECTORAL_BANKING", "INDEX", "MULTI_CAP",
}
_DEBT_CATEGORIES = {"CORPORATE_BOND", "GILT", "LIQUID", "MONEY_MARKET", "DURATION"}
_HYBRID_CATEGORIES = {"BALANCED_ADVANTAGE", "HYBRID", "AGGRESSIVE_HYBRID", "CONSERVATIVE_HYBRID"}
_ARBITRAGE_CATEGORIES = {"ARBITRAGE"}


class AllocationAnalyzer:
    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader

    def get_allocation_breakdown(self, portfolio_id: str) -> AllocationBreakdown | None:
        portfolio = self._loader.get_portfolio(portfolio_id)
        if not portfolio:
            return None

        total_value = self._compute_total_value(portfolio)
        if total_value == 0:
            return None

        by_sector: dict[str, float] = {}
        by_asset_type: dict[str, float] = {}
        mf_indirect: dict[str, dict[str, float]] = {}

        stocks_value = 0.0
        for holding in portfolio.holdings.stocks:
            stock = self._loader.get_stock(holding.symbol)
            current_price = stock.current_price if stock else holding.current_price
            current_value = current_price * holding.quantity
            stocks_value += current_value
            pct = current_value / total_value * 100
            by_sector[holding.sector] = by_sector.get(holding.sector, 0.0) + pct

        by_asset_type["DIRECT_STOCKS"] = round(stocks_value / total_value * 100, 2)

        equity_mf_value = debt_mf_value = hybrid_mf_value = arbitrage_mf_value = 0.0

        for mf_holding in portfolio.holdings.mutual_funds:
            scheme = self._loader.get_mutual_fund(mf_holding.scheme_code)
            current_nav = scheme.current_nav if scheme else mf_holding.current_nav
            mf_value = current_nav * mf_holding.units
            mf_portfolio_weight = mf_value / total_value

            category = mf_holding.category.upper()
            if category in _EQUITY_CATEGORIES:
                equity_mf_value += mf_value
            elif category in _DEBT_CATEGORIES:
                debt_mf_value += mf_value
            elif category in _HYBRID_CATEGORIES:
                hybrid_mf_value += mf_value
            elif category in _ARBITRAGE_CATEGORIES:
                arbitrage_mf_value += mf_value
            else:
                equity_mf_value += mf_value

            if scheme and scheme.sector_allocation:
                mf_indirect[mf_holding.scheme_name] = {}
                for raw_sector, sector_pct in scheme.sector_allocation.items():
                    canonical = self._canonicalize(raw_sector)
                    if canonical is None:
                        continue
                    indirect_pct = mf_portfolio_weight * (sector_pct / 100) * 100
                    by_sector[canonical] = by_sector.get(canonical, 0.0) + indirect_pct
                    mf_indirect[mf_holding.scheme_name][canonical] = round(indirect_pct, 3)

        if equity_mf_value > 0:
            by_asset_type["EQUITY_MF"] = round(equity_mf_value / total_value * 100, 2)
        if debt_mf_value > 0:
            by_asset_type["DEBT_MF"] = round(debt_mf_value / total_value * 100, 2)
        if hybrid_mf_value > 0:
            by_asset_type["HYBRID_MF"] = round(hybrid_mf_value / total_value * 100, 2)
        if arbitrage_mf_value > 0:
            by_asset_type["ARBITRAGE_MF"] = round(arbitrage_mf_value / total_value * 100, 2)

        by_sector_rounded = {k: round(v, 2) for k, v in sorted(by_sector.items(), key=lambda x: -x[1])}

        return AllocationBreakdown(
            by_sector=by_sector_rounded,
            by_asset_type=by_asset_type,
            mf_indirect_sector_exposure=mf_indirect,
        )

    def get_sector_exposure(self, portfolio_id: str) -> dict[str, float]:
        breakdown = self.get_allocation_breakdown(portfolio_id)
        return breakdown.by_sector if breakdown else {}

    def _compute_total_value(self, portfolio: Portfolio) -> float:
        total = 0.0
        for holding in portfolio.holdings.stocks:
            stock = self._loader.get_stock(holding.symbol)
            total += (stock.current_price if stock else holding.current_price) * holding.quantity
        for mf_holding in portfolio.holdings.mutual_funds:
            scheme = self._loader.get_mutual_fund(mf_holding.scheme_code)
            total += (scheme.current_nav if scheme else mf_holding.current_nav) * mf_holding.units
        return total

    def _canonicalize(self, raw: str) -> str | None:
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
            "TELECOM": "TELECOM",
        }
        skip = {
            "DEBT", "CASH", "CASH_AND_EQUIV", "GOVERNMENT_SECURITIES",
            "AAA_CORPORATE_BONDS", "AA_CORPORATE_BONDS",
            "ARBITRAGE_POSITIONS", "OTHERS",
        }
        upper = raw.upper()
        if upper in skip:
            return None
        return mapping.get(upper)
