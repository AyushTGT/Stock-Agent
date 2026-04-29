from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from src.models.types import (
    IndexData,
    IndexHistory,
    OHLCBar,
    SectorPerformance,
    StockData,
    StockHistory,
)

logger = logging.getLogger(__name__)

_NSE_INDEX_MAP = {
    "NIFTY50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANKNIFTY": "^NSEBANK",
    "NIFTYIT": "^CNXIT",
    "NIFTYPHARMA": "^CNXPHARMA",
}

_INDEX_SENTIMENT_MAP = {
    True: "BULLISH",
    False: "BEARISH",
}


def _sentiment(change_pct: float) -> str:
    if change_pct > 0.5:
        return "BULLISH"
    if change_pct < -0.5:
        return "BEARISH"
    return "NEUTRAL"


def fetch_stocks(symbols: list[str]) -> dict[str, StockData]:
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — skipping live stock fetch")
        return {}

    result: dict[str, StockData] = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            info = ticker.info
            hist = ticker.history(period="2d")
            if hist.empty or len(hist) < 2:
                continue
            current = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[-2])
            change_abs = current - prev_close
            change_pct = (change_abs / prev_close) * 100 if prev_close else 0.0

            result[symbol] = StockData(
                name=info.get("longName") or info.get("shortName") or symbol,
                sector=info.get("sector") or "Unknown",
                sub_sector=info.get("industry"),
                current_price=current,
                previous_close=prev_close,
                change_percent=round(change_pct, 2),
                change_absolute=round(change_abs, 2),
                volume=int(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else None,
                avg_volume_20d=int(info.get("averageVolume") or 0) or None,
                market_cap_cr=round(info["marketCap"] / 1e7, 2) if info.get("marketCap") else None,
                pe_ratio=info.get("trailingPE"),
                beta=info.get("beta") or 1.0,
            )
        except Exception as exc:
            logger.debug("Failed to fetch %s: %s", symbol, exc)
    return result


def fetch_stock_histories(symbols: list[str]) -> dict[str, StockHistory]:
    try:
        import yfinance as yf
    except ImportError:
        return {}

    result: dict[str, StockHistory] = {}
    end = datetime.today()
    start = end - timedelta(days=10)

    for symbol in symbols:
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            hist = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
            if len(hist) < 3:
                continue
            bars = [
                OHLCBar(
                    date=str(idx.date()),
                    close=round(float(row["Close"]), 2),
                    volume=int(row["Volume"]) if "Volume" in hist.columns else None,
                    change_percent=round(
                        ((float(row["Close"]) - float(hist["Close"].iloc[i - 1])) / float(hist["Close"].iloc[i - 1])) * 100,
                        2,
                    ) if i > 0 else 0.0,
                )
                for i, (idx, row) in enumerate(hist.iterrows())
            ][-7:]

            closes = [b.close for b in bars]
            trend = "BULLISH" if closes[-1] > closes[0] else "BEARISH" if closes[-1] < closes[0] else "NEUTRAL"
            avg_vol = sum(b.volume for b in bars if b.volume) / max(1, sum(1 for b in bars if b.volume))

            result[symbol] = StockHistory(
                data=bars,
                trend=trend,
                avg_volume_7d=round(avg_vol, 0),
                volatility="HIGH" if abs(closes[-1] - closes[0]) / closes[0] * 100 > 3 else "MEDIUM",
            )
        except Exception as exc:
            logger.debug("Failed history for %s: %s", symbol, exc)
    return result


def fetch_indices() -> dict[str, IndexData]:
    try:
        import yfinance as yf
    except ImportError:
        return {}

    result: dict[str, IndexData] = {}
    for name, ticker_sym in _NSE_INDEX_MAP.items():
        try:
            ticker = yf.Ticker(ticker_sym)
            hist = ticker.history(period="2d")
            if hist.empty or len(hist) < 2:
                continue
            current = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            change_abs = current - prev
            change_pct = (change_abs / prev) * 100 if prev else 0.0
            info = ticker.info

            result[name] = IndexData(
                name=name,
                current_value=round(current, 2),
                previous_close=round(prev, 2),
                change_percent=round(change_pct, 2),
                change_absolute=round(change_abs, 2),
                day_high=round(float(hist["High"].iloc[-1]), 2) if "High" in hist.columns else None,
                day_low=round(float(hist["Low"].iloc[-1]), 2) if "Low" in hist.columns else None,
                **{"52_week_high": info.get("fiftyTwoWeekHigh")},
                **{"52_week_low": info.get("fiftyTwoWeekLow")},
                sentiment=_sentiment(change_pct),
            )
        except Exception as exc:
            logger.debug("Failed to fetch index %s: %s", name, exc)
    return result


def fetch_index_histories() -> dict[str, IndexHistory]:
    try:
        import yfinance as yf
    except ImportError:
        return {}

    result: dict[str, IndexHistory] = {}
    end = datetime.today()
    start = end - timedelta(days=10)

    for name, ticker_sym in _NSE_INDEX_MAP.items():
        try:
            ticker = yf.Ticker(ticker_sym)
            hist = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
            if len(hist) < 3:
                continue
            bars = [
                OHLCBar(
                    date=str(idx.date()),
                    close=round(float(row["Close"]), 2),
                    change_percent=round(
                        ((float(row["Close"]) - float(hist["Close"].iloc[i - 1])) / float(hist["Close"].iloc[i - 1])) * 100,
                        2,
                    ) if i > 0 else 0.0,
                )
                for i, (idx, row) in enumerate(hist.iterrows())
            ][-7:]

            closes = [b.close for b in bars]
            trend = "BULLISH" if closes[-1] > closes[0] else "BEARISH" if closes[-1] < closes[0] else "NEUTRAL"
            cumulative = round((closes[-1] - closes[0]) / closes[0] * 100, 2) if closes[0] else 0.0

            result[name] = IndexHistory(
                data=bars,
                trend=trend,
                cumulative_change_percent=cumulative,
            )
        except Exception as exc:
            logger.debug("Failed history for index %s: %s", name, exc)
    return result


def fetch_sector_performance(sector_stocks_map: dict[str, list[str]]) -> dict[str, SectorPerformance]:
    stocks = fetch_stocks([s for stocks in sector_stocks_map.values() for s in stocks])
    result: dict[str, SectorPerformance] = {}

    for sector, symbols in sector_stocks_map.items():
        changes = [stocks[s].change_percent for s in symbols if s in stocks]
        if not changes:
            continue
        avg_change = sum(changes) / len(changes)
        gainers = sorted([s for s in symbols if s in stocks and stocks[s].change_percent > 0],
                         key=lambda s: stocks[s].change_percent, reverse=True)[:3]
        losers = sorted([s for s in symbols if s in stocks and stocks[s].change_percent < 0],
                        key=lambda s: stocks[s].change_percent)[:3]
        result[sector] = SectorPerformance(
            change_percent=round(avg_change, 2),
            sentiment=_sentiment(avg_change),
            key_drivers=[f"{s} {stocks[s].change_percent:+.2f}%" for s in (gainers + losers)[:3]],
            top_gainers=gainers,
            top_losers=losers,
        )
    return result
