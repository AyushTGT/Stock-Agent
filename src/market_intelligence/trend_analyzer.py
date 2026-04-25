from __future__ import annotations

from typing import Literal

import numpy as np

from src.data_loader import DataLoader
from src.models.types import IndexTrendDetail, MarketTrend

_BULLISH_THRESHOLD = 0.003
_BEARISH_THRESHOLD = -0.003


class TrendAnalyzer:
    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader

    def get_market_trend(self) -> MarketTrend:
        index_details: dict[str, IndexTrendDetail] = {}

        for index_name in ["NIFTY50", "SENSEX", "BANKNIFTY", "NIFTYIT", "NIFTYPHARMA"]:
            history = self._loader.get_index_history(index_name)
            current = self._loader.get_index(index_name)
            if not history or not current:
                continue

            closes = [bar.close for bar in history.data]
            slope = self._linear_slope(closes)
            index_details[index_name] = IndexTrendDetail(
                current_value=current.current_value,
                change_percent=current.change_percent,
                seven_day_trend=self._classify_slope(slope),
                slope=round(slope, 6),
                cumulative_7d_change=history.cumulative_change_percent,
            )

        overall = self._derive_overall_trend(index_details)

        fii = self._loader.get_fii_data()
        dii = self._loader.get_dii_data()
        fii_net = fii.net_value_cr if fii else 0.0
        dii_net = dii.net_value_cr if dii else 0.0

        if fii_net < 0 and dii_net > 0:
            fii_dii_summary = (
                f"FIIs net sellers ₹{abs(fii_net):,.0f} Cr; "
                f"DIIs absorbing with ₹{dii_net:,.0f} Cr net buying. "
                f"MTD FII outflow: ₹{abs(fii.mtd_net_cr):,.0f} Cr."
            )
        elif fii_net > 0:
            fii_dii_summary = f"FIIs net buyers ₹{fii_net:,.0f} Cr; DII activity: ₹{dii_net:,.0f} Cr."
        else:
            fii_dii_summary = f"FII net: ₹{fii_net:,.0f} Cr | DII net: ₹{dii_net:,.0f} Cr"

        breadth = self._loader.get_market_breadth()
        if breadth:
            advances, declines = breadth.advances, breadth.declines
            adr = breadth.advance_decline_ratio
            if adr < 0.30:
                breadth_signal = f"VERY WEAK ({advances}/{advances+declines} advancing) — broad selloff"
            elif adr < 0.50:
                breadth_signal = f"WEAK ({advances}/{advances+declines} advancing)"
            elif adr < 0.70:
                breadth_signal = f"MIXED ({advances}/{advances+declines} advancing)"
            else:
                breadth_signal = f"STRONG ({advances}/{advances+declines} advancing)"
        else:
            advances = declines = 0
            adr = 0.5
            breadth_signal = "DATA UNAVAILABLE"

        sentiment_indicator = "FEAR" if overall == "BEARISH" else "GREED" if overall == "BULLISH" else "NEUTRAL"

        return MarketTrend(
            overall_trend=overall,
            indices=index_details,
            fii_net_cr=fii_net,
            dii_net_cr=dii_net,
            fii_dii_summary=fii_dii_summary,
            breadth_advances=advances,
            breadth_declines=declines,
            advance_decline_ratio=adr,
            breadth_signal=breadth_signal,
            sentiment_indicator=sentiment_indicator,
        )

    def _linear_slope(self, closes: list[float]) -> float:
        """Least-squares slope normalized to starting price."""
        if len(closes) < 2:
            return 0.0
        x = np.arange(len(closes), dtype=float)
        slope, _ = np.polyfit(x, closes, 1)
        return float(slope / closes[0])

    def _classify_slope(self, slope: float) -> Literal["BULLISH", "BEARISH", "NEUTRAL"]:
        if slope > _BULLISH_THRESHOLD:
            return "BULLISH"
        if slope < _BEARISH_THRESHOLD:
            return "BEARISH"
        return "NEUTRAL"

    def _derive_overall_trend(
        self, index_details: dict[str, IndexTrendDetail]
    ) -> Literal["BULLISH", "BEARISH", "NEUTRAL", "MIXED"]:
        counts: dict[str, int] = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
        for detail in index_details.values():
            counts[detail.seven_day_trend] += 1

        total = sum(counts.values())
        if total == 0:
            return "NEUTRAL"
        if counts["BEARISH"] >= total * 0.6:
            return "BEARISH"
        if counts["BULLISH"] >= total * 0.6:
            return "BULLISH"
        if counts["BEARISH"] > counts["BULLISH"]:
            return "MIXED"
        return "NEUTRAL"
