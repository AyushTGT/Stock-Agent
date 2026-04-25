from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import ClassVar, Optional

from src.models.types import (
    FIIDIIData,
    IndexData,
    IndexHistory,
    MacroCorrelation,
    MarketBreadth,
    MutualFundScheme,
    NewsArticle,
    OHLCBar,
    Portfolio,
    SectorInfo,
    SectorPerformance,
    SectorWeeklyPerformance,
    StockData,
    StockHistory,
)


class DataLoader:
    """Singleton. Loads all 6 JSON data files once; thread-safe via double-checked locking."""

    _instance: ClassVar[Optional["DataLoader"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._indices: dict[str, IndexData] = {}
        self._sector_performance: dict[str, SectorPerformance] = {}
        self._stocks: dict[str, StockData] = {}
        self._index_history: dict[str, IndexHistory] = {}
        self._stock_history: dict[str, StockHistory] = {}
        self._sector_weekly: dict[str, SectorWeeklyPerformance] = {}
        self._market_breadth: Optional[MarketBreadth] = None
        self._fii_data: Optional[FIIDIIData] = None
        self._dii_data: Optional[FIIDIIData] = None
        self._news: list[NewsArticle] = []
        self._news_by_id: dict[str, NewsArticle] = {}
        self._portfolios: dict[str, Portfolio] = {}
        self._mutual_funds: dict[str, MutualFundScheme] = {}
        self._sector_info: dict[str, SectorInfo] = {}
        self._macro_correlations: list[MacroCorrelation] = []
        self._defensive_sectors: list[str] = []
        self._rate_sensitive_sectors: list[str] = []

    @classmethod
    def get_instance(cls, data_dir: Path) -> "DataLoader":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = DataLoader(data_dir)
                    inst._load_all()
                    cls._instance = inst
        return cls._instance

    def _load_all(self) -> None:
        self._load_market_data()
        self._load_historical_data()
        self._load_news_data()
        self._load_portfolios()
        self._load_mutual_funds()
        self._load_sector_mapping()

    def _read_json(self, filename: str) -> dict:
        path = self._data_dir / filename
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def _load_market_data(self) -> None:
        raw = self._read_json("market_data.json")
        for name, data in raw.get("indices", {}).items():
            self._indices[name] = IndexData.model_validate(data)
        for name, data in raw.get("sector_performance", {}).items():
            self._sector_performance[name] = SectorPerformance.model_validate(data)
        for symbol, data in raw.get("stocks", {}).items():
            self._stocks[symbol] = StockData.model_validate(data)

    def _load_historical_data(self) -> None:
        raw = self._read_json("historical_data.json")
        for name, data in raw.get("index_history", {}).items():
            bars = [OHLCBar.model_validate(b) for b in data["data"]]
            self._index_history[name] = IndexHistory(
                data=bars,
                trend=data["trend"],
                cumulative_change_percent=data["cumulative_change_percent"],
                support_level=data.get("support_level"),
                resistance_level=data.get("resistance_level"),
            )
        for symbol, data in raw.get("stock_history", {}).items():
            bars = [OHLCBar.model_validate(b) for b in data["data"]]
            self._stock_history[symbol] = StockHistory(
                data=bars,
                trend=data["trend"],
                avg_volume_7d=data.get("avg_volume_7d"),
                volatility=data.get("volatility"),
            )
        for sector, data in raw.get("sector_weekly_performance", {}).items():
            self._sector_weekly[sector] = SectorWeeklyPerformance.model_validate(data)

        breadth_raw = raw.get("market_breadth", {}).get("nifty50", {})
        if breadth_raw:
            self._market_breadth = MarketBreadth.model_validate(breadth_raw)

        fii_dii = raw.get("fii_dii_data", {})
        if fii_dii:
            self._fii_data = FIIDIIData.model_validate(fii_dii["fii"])
            self._dii_data = FIIDIIData.model_validate(fii_dii["dii"])

    def _load_news_data(self) -> None:
        raw = self._read_json("news_data.json")
        for item in raw.get("news", []):
            article = NewsArticle.model_validate(item)
            self._news.append(article)
            self._news_by_id[article.id] = article

    def _load_portfolios(self) -> None:
        raw = self._read_json("portfolios.json")
        for pid, data in raw.get("portfolios", {}).items():
            self._portfolios[pid] = Portfolio.model_validate(data)

    def _load_mutual_funds(self) -> None:
        raw = self._read_json("mutual_funds.json")
        for code, data in raw.get("mutual_funds", {}).items():
            self._mutual_funds[code] = MutualFundScheme.model_validate(data)

    def _load_sector_mapping(self) -> None:
        raw = self._read_json("sector_mapping.json")
        for sector, data in raw.get("sectors", {}).items():
            self._sector_info[sector] = SectorInfo.model_validate(data)
        for event, data in raw.get("macro_correlations", {}).items():
            self._macro_correlations.append(MacroCorrelation(
                event=event,
                negative_impact=data.get("negative_impact", []),
                positive_impact=data.get("positive_impact", []),
                neutral=data.get("neutral", []),
            ))
        self._defensive_sectors = raw.get("defensive_sectors", [])
        self._rate_sensitive_sectors = raw.get("rate_sensitive_sectors", [])

    def get_stock(self, symbol: str) -> Optional[StockData]:
        return self._stocks.get(symbol)

    def get_all_stocks(self) -> dict[str, StockData]:
        return self._stocks

    def get_index(self, name: str) -> Optional[IndexData]:
        return self._indices.get(name)

    def get_all_indices(self) -> dict[str, IndexData]:
        return self._indices

    def get_sector_performance(self, name: str) -> Optional[SectorPerformance]:
        return self._sector_performance.get(name)

    def get_all_sector_performances(self) -> dict[str, SectorPerformance]:
        return self._sector_performance

    def get_index_history(self, name: str) -> Optional[IndexHistory]:
        return self._index_history.get(name)

    def get_all_index_histories(self) -> dict[str, IndexHistory]:
        return self._index_history

    def get_stock_history(self, symbol: str) -> Optional[StockHistory]:
        return self._stock_history.get(symbol)

    def get_sector_weekly(self, sector: str) -> Optional[SectorWeeklyPerformance]:
        return self._sector_weekly.get(sector)

    def get_all_sector_weekly(self) -> dict[str, SectorWeeklyPerformance]:
        return self._sector_weekly

    def get_market_breadth(self) -> Optional[MarketBreadth]:
        return self._market_breadth

    def get_fii_data(self) -> Optional[FIIDIIData]:
        return self._fii_data

    def get_dii_data(self) -> Optional[FIIDIIData]:
        return self._dii_data

    def get_news(self) -> list[NewsArticle]:
        return self._news

    def get_news_by_id(self, news_id: str) -> Optional[NewsArticle]:
        return self._news_by_id.get(news_id)

    def get_portfolio(self, portfolio_id: str) -> Optional[Portfolio]:
        return self._portfolios.get(portfolio_id)

    def get_all_portfolio_ids(self) -> list[str]:
        return list(self._portfolios.keys())

    def get_mutual_fund(self, scheme_code: str) -> Optional[MutualFundScheme]:
        return self._mutual_funds.get(scheme_code)

    def get_sector_info(self, sector: str) -> Optional[SectorInfo]:
        return self._sector_info.get(sector)

    def get_sector_stocks(self, sector: str) -> list[str]:
        info = self._sector_info.get(sector)
        return info.stocks if info else []

    def get_all_sectors(self) -> list[str]:
        return list(self._sector_info.keys())

    def get_macro_correlations(self) -> list[MacroCorrelation]:
        return self._macro_correlations

    def get_defensive_sectors(self) -> list[str]:
        return self._defensive_sectors

    def get_rate_sensitive_sectors(self) -> list[str]:
        return self._rate_sensitive_sectors

    def build_market_snapshot_text(self) -> str:
        lines: list[str] = ["=== MARKET SNAPSHOT ===\n"]

        lines.append("--- INDICES ---")
        for name, idx in self._indices.items():
            lines.append(f"{name}: {idx.current_value:,.2f}  ({idx.change_percent:+.2f}%)  [{idx.sentiment}]")

        lines.append("\n--- SECTOR PERFORMANCE (TODAY) ---")
        for sector, sp in self._sector_performance.items():
            drivers = " | ".join(sp.key_drivers[:2])
            lines.append(f"{sector}: {sp.change_percent:+.2f}%  [{sp.sentiment}]  — {drivers}")

        lines.append("\n--- STOCKS ---")
        for sym, st in self._stocks.items():
            lines.append(f"{sym} ({st.sector}): ₹{st.current_price:,.2f}  {st.change_percent:+.2f}%  β={st.beta}")

        lines.append("\n--- FII / DII FLOWS ---")
        if self._fii_data:
            lines.append(f"FII net: ₹{self._fii_data.net_value_cr:,.0f} Cr  (MTD: ₹{self._fii_data.mtd_net_cr:,.0f} Cr)")
        if self._dii_data:
            lines.append(f"DII net: +₹{self._dii_data.net_value_cr:,.0f} Cr  (MTD: +₹{self._dii_data.mtd_net_cr:,.0f} Cr)")

        if self._market_breadth:
            mb = self._market_breadth
            lines.append(
                f"\nMarket Breadth (NIFTY50): {mb.advances} advances / {mb.declines} declines "
                f"(A/D ratio: {mb.advance_decline_ratio:.2f})"
            )

        lines.append("\n--- SECTOR WEEKLY PERFORMANCE ---")
        for sector, sw in self._sector_weekly.items():
            lines.append(f"{sector}: {sw.weekly_change_percent:+.2f}% ({sw.trend}) — {sw.catalyst}")

        lines.append("\n--- MACRO CORRELATIONS ---")
        for mc in self._macro_correlations:
            neg = ", ".join(mc.negative_impact) or "none"
            pos = ", ".join(mc.positive_impact) or "none"
            lines.append(f"{mc.event}: ↓{neg}  ↑{pos}")

        return "\n".join(lines)
