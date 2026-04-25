from __future__ import annotations

from src.data_loader import DataLoader
from src.models.types import HoldingPnL, PortfolioSummary


class PnLCalculator:
    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader

    def get_portfolio_summary(self, portfolio_id: str) -> PortfolioSummary | None:
        portfolio = self._loader.get_portfolio(portfolio_id)
        if not portfolio:
            return None

        stock_rows = self._compute_stock_holdings(portfolio)
        mf_rows = self._compute_mf_holdings(portfolio)
        all_rows = stock_rows + mf_rows

        total_value = sum(r["current_value"] for r in all_rows)
        if total_value == 0:
            total_value = 1.0

        holdings_pnl: list[HoldingPnL] = []
        for row in all_rows:
            weight = row["current_value"] / total_value * 100
            contribution = weight / 100 * row["day_pnl_percent"]
            holdings_pnl.append(HoldingPnL(
                symbol=row["symbol"],
                name=row["name"],
                holding_type=row["holding_type"],
                sector=row.get("sector", "DIVERSIFIED"),
                current_value=round(row["current_value"], 2),
                invested_value=round(row["invested_value"], 2),
                day_pnl_absolute=round(row["day_pnl_absolute"], 2),
                day_pnl_percent=round(row["day_pnl_percent"], 4),
                overall_gain_loss=round(row["current_value"] - row["invested_value"], 2),
                overall_gain_loss_percent=round(
                    (row["current_value"] - row["invested_value"]) / row["invested_value"] * 100, 4
                ) if row["invested_value"] else 0.0,
                portfolio_weight=round(weight, 4),
                contribution_to_day_pnl=round(contribution, 4),
            ))

        total_invested = sum(r["invested_value"] for r in all_rows)
        day_pnl_absolute = sum(r["day_pnl_absolute"] for r in all_rows)
        day_pnl_percent = sum(h.contribution_to_day_pnl for h in holdings_pnl)

        active = [h for h in holdings_pnl if h.day_pnl_percent != 0]
        top_gainer = max(active, key=lambda h: h.day_pnl_percent).symbol if active else None
        top_loser = min(active, key=lambda h: h.day_pnl_percent).symbol if active else None

        return PortfolioSummary(
            portfolio_id=portfolio_id,
            owner_name=portfolio.user_name,
            total_value=round(total_value, 2),
            total_invested=round(total_invested, 2),
            day_pnl_absolute=round(day_pnl_absolute, 2),
            day_pnl_percent=round(day_pnl_percent, 4),
            overall_gain_loss=round(total_value - total_invested, 2),
            overall_gain_loss_percent=round(
                (total_value - total_invested) / total_invested * 100, 4
            ) if total_invested else 0.0,
            holdings=holdings_pnl,
            top_gainer=top_gainer,
            top_loser=top_loser,
        )

    def _compute_stock_holdings(self, portfolio) -> list[dict]:
        rows = []
        for holding in portfolio.holdings.stocks:
            stock = self._loader.get_stock(holding.symbol)
            if not stock:
                current_price, change_pct = holding.current_price, 0.0
            else:
                current_price, change_pct = stock.current_price, stock.change_percent

            prev_close = current_price / (1 + change_pct / 100) if change_pct != -100 else current_price
            rows.append({
                "symbol": holding.symbol,
                "name": holding.name,
                "holding_type": "STOCK",
                "sector": holding.sector,
                "current_value": current_price * holding.quantity,
                "invested_value": holding.avg_buy_price * holding.quantity,
                "day_pnl_absolute": (current_price - prev_close) * holding.quantity,
                "day_pnl_percent": change_pct,
            })
        return rows

    def _compute_mf_holdings(self, portfolio) -> list[dict]:
        rows = []
        for mf_holding in portfolio.holdings.mutual_funds:
            scheme = self._loader.get_mutual_fund(mf_holding.scheme_code)
            if not scheme:
                current_nav, day_pct = mf_holding.current_nav, 0.0
            else:
                current_nav = scheme.current_nav
                day_pct = scheme.returns.get("1_day", 0.0)

            prev_nav = current_nav / (1 + day_pct / 100) if day_pct != -100 else current_nav
            rows.append({
                "symbol": mf_holding.scheme_code,
                "name": mf_holding.scheme_name,
                "holding_type": "MUTUAL_FUND",
                "sector": "DIVERSIFIED",
                "current_value": current_nav * mf_holding.units,
                "invested_value": mf_holding.avg_nav * mf_holding.units,
                "day_pnl_absolute": (current_nav - prev_nav) * mf_holding.units,
                "day_pnl_percent": day_pct,
            })
        return rows
