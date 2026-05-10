from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data_loader import DataLoader
from src.market_intelligence.news_processor import NewsProcessor
from src.market_intelligence.sector_engine import SectorEngine
from src.market_intelligence.trend_analyzer import TrendAnalyzer
from src.portfolio_analytics.allocation_analyzer import AllocationAnalyzer
from src.portfolio_analytics.pnl_calculator import PnLCalculator
from src.portfolio_analytics.risk_detector import RiskDetector
from src.reasoning.causal_linker import CausalLinker
from src.reasoning.conflict_resolver import ConflictResolver
from src.reasoning.signal_ranker import SignalRanker

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_summary",
            "description": (
                "Returns the daily P&L, total value, and per-holding breakdown for a portfolio. "
                "Each holding shows current value, day change %, and its weighted contribution to "
                "total portfolio P&L in percentage points. Use this first to understand how the "
                "portfolio is performing today."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {
                        "type": "string",
                        "description": "Portfolio identifier, e.g. PORTFOLIO_001, PORTFOLIO_002, PORTFOLIO_003",
                    }
                },
                "required": ["portfolio_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_allocation_breakdown",
            "description": (
                "Returns the full asset allocation breakdown: by sector (including indirect exposure "
                "via mutual fund drill-through) and by asset type (DIRECT_STOCKS, EQUITY_MF, DEBT_MF, "
                "HYBRID_MF, ARBITRAGE_MF). MF indirect sector exposure is decomposed from each fund's "
                "sector_allocation. Essential for understanding true diversification."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {
                        "type": "string",
                        "description": "Portfolio identifier",
                    }
                },
                "required": ["portfolio_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_risk_assessment",
            "description": (
                "Returns a full risk report with flags for: sector concentration (CRITICAL >60%, "
                "HIGH >40%, MEDIUM >25%), single-stock concentration (>20%), high portfolio beta (>1.3), "
                "and mutual fund overlap (same underlying stock in 3+ funds). Includes severity levels "
                "and actionable explanation for each flag."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {
                        "type": "string",
                        "description": "Portfolio identifier",
                    }
                },
                "required": ["portfolio_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_trend",
            "description": (
                "Returns a 7-day market trend analysis for the 5 major indices (NIFTY50, SENSEX, "
                "BANKNIFTY, NIFTYIT, NIFTYPHARMA) using linear regression slopes. Reports FII/DII "
                "net flows for the week, market breadth (advance/decline ratio), and an overall "
                "market direction (BULLISH / BEARISH / NEUTRAL / MIXED). Use to set macro context."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sector_analysis",
            "description": (
                "Returns today's change %, weekly return, news sentiment average, and key drivers "
                "for each requested sector. If no sectors specified, returns all available sectors. "
                "Use to understand which sectors are under pressure and why."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sectors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of sector names to analyse, e.g. ['BANKING', 'INFORMATION_TECHNOLOGY']. "
                            "Leave empty to get all sectors."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_causal_chain",
            "description": (
                "Builds quantified causal chains: News Article → Sector Impact → Individual Stock → "
                "Portfolio P&L in basis points (1bp = 0.01%). Each chain shows exactly which news "
                "drove which sector, which portfolio holdings were affected, their portfolio weights, "
                "and the contribution in basis points. Chains are sorted by absolute impact. This is "
                "the core reasoning tool — use it to explain WHY the portfolio moved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {
                        "type": "string",
                        "description": "Portfolio identifier",
                    },
                    "news_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of news IDs to trace, e.g. ['NEWS001', 'NEWS011']. "
                            "Leave empty to auto-select the top 7 most relevant articles."
                        ),
                    },
                },
                "required": ["portfolio_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_conflict_analysis",
            "description": (
                "Detects contradictory signals in the market for the given portfolio: "
                "(1) POSITIVE_NEWS_STOCK_FALLING — positive news but stock is down; "
                "(2) NEGATIVE_NEWS_STOCK_RISING — negative news but stock is up; "
                "(3) SECTOR_STOCK_DIVERGENCE — stock moves against its sector; "
                "(4) MIXED_SIGNALS — multiple articles on the same entity with opposite sentiments. "
                "Each report includes an explanation and a resolution of the contradiction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {
                        "type": "string",
                        "description": "Portfolio identifier",
                    },
                    "news_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of news IDs to limit conflict scan to",
                    },
                },
                "required": ["portfolio_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news_for_query",
            "description": (
                "Filters and ranks news articles by relevance to given sectors, stocks, or indices. "
                "Articles are ranked by impact_level × |sentiment_score|. Returns headlines, "
                "sentiment, scope, impact level, causal factors, and whether a conflict flag is set."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sectors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by sector names",
                    },
                    "stocks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by stock symbols",
                    },
                    "indices": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by index names",
                    },
                    "scope_filter": {
                        "type": "string",
                        "enum": ["MARKET_WIDE", "SECTOR_SPECIFIC", "STOCK_SPECIFIC"],
                        "description": "Optionally limit to one scope",
                    },
                    "impact_filter": {
                        "type": "string",
                        "enum": ["HIGH", "MEDIUM", "LOW"],
                        "description": "Optionally limit to one impact level",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of articles to return (default 10)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_confidence_score",
            "description": (
                "Computes a deterministic confidence score (0–1) for the current analysis using five "
                "components: news_strength (0.30 weight), corroboration (0.25), breadth_alignment "
                "(0.20), data_coverage (0.25), and a conflict_penalty (−0.10 per unresolved conflict, "
                "capped at −0.30). Returns overall score, component breakdown, and interpretation "
                "(HIGH / MEDIUM / LOW). Call this last, after building causal chains and conflict reports."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {
                        "type": "string",
                        "description": "Portfolio identifier",
                    },
                    "news_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "News IDs used in this analysis",
                    },
                    "conflict_count": {
                        "type": "integer",
                        "description": "Number of unresolved conflicts from get_conflict_analysis",
                    },
                },
                "required": ["portfolio_id", "news_ids"],
            },
        },
    },
]


class ToolDispatcher:
    def __init__(self, data_dir: Path) -> None:
        print("[DEBUG] ToolDispatcher: loading DataLoader...", flush=True)
        self._loader = DataLoader.get_instance(data_dir)
        print("[DEBUG] ToolDispatcher: DataLoader ready", flush=True)

        self._pnl = PnLCalculator(self._loader)
        self._alloc = AllocationAnalyzer(self._loader)
        self._risk = RiskDetector(self._loader)
        self._trend = TrendAnalyzer(self._loader)
        self._sector = SectorEngine(self._loader)
        self._news = NewsProcessor(self._loader)
        self._causal = CausalLinker(self._loader)
        self._conflict = ConflictResolver(self._loader)
        self._ranker = SignalRanker(self._loader)
        print("[DEBUG] ToolDispatcher: all modules ready, starting RAG init...", flush=True)

        try:
            from src.rag.news_ingester import ingest_from_loader
            from src.rag.vector_store import VectorStore
            print("[DEBUG] ToolDispatcher: ingesting news...", flush=True)
            ingest_from_loader(self._loader)
            print("[DEBUG] ToolDispatcher: news ingested, loading VectorStore...", flush=True)
            self._vector_store = VectorStore.get_instance()
            print("[DEBUG] ToolDispatcher: VectorStore ready", flush=True)
        except Exception as e:
            print(f"[DEBUG] ToolDispatcher: RAG failed with: {e}", flush=True)
            self._vector_store = None

    def dispatch(self, tool_name: str, tool_input: dict[str, Any]) -> dict:
        try:
            return self._route(tool_name, tool_input)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "tool": tool_name}

    def _route(self, name: str, inp: dict) -> dict:
        if name == "get_portfolio_summary":
            result = self._pnl.get_portfolio_summary(inp["portfolio_id"])
            return result.model_dump() if result else {"error": "Portfolio not found"}

        if name == "get_allocation_breakdown":
            result = self._alloc.get_allocation_breakdown(inp["portfolio_id"])
            return result.model_dump() if result else {"error": "Portfolio not found"}

        if name == "get_risk_assessment":
            result = self._risk.get_risk_assessment(inp["portfolio_id"])
            return result.model_dump() if result else {"error": "Portfolio not found"}

        if name == "get_market_trend":
            result = self._trend.get_market_trend()
            return result.model_dump()

        if name == "get_sector_analysis":
            sectors = inp.get("sectors") or []
            results = self._sector.get_sector_analysis(sectors if sectors else None)
            return {"sectors": [r.model_dump() for r in results]}

        if name == "get_causal_chain":
            portfolio_id = inp["portfolio_id"]
            news_ids = inp.get("news_ids") or []
            if news_ids:
                chains = self._causal.get_causal_chain(portfolio_id, news_ids)
            else:
                chains = self._causal.build_full_chain_for_portfolio(portfolio_id)
            return {"chains": [c.model_dump() for c in chains]}

        if name == "get_conflict_analysis":
            portfolio_id = inp["portfolio_id"]
            news_ids = inp.get("news_ids") or None
            reports = self._conflict.get_conflict_analysis(portfolio_id, news_ids)
            return {"conflicts": [r.model_dump() for r in reports]}

        if name == "get_news_for_query":
            max_results = inp.get("max_results", 10)
            articles = self._rag_news_query(inp, max_results)
            if not articles:
                articles = self._news.get_news_for_query(
                    sectors=inp.get("sectors") or [],
                    stocks=inp.get("stocks") or [],
                    indices=inp.get("indices") or [],
                    scope_filter=inp.get("scope_filter"),
                    impact_filter=inp.get("impact_filter"),
                    max_results=max_results,
                )
            return {"articles": [a.model_dump() for a in articles]}

        if name == "compute_confidence_score":
            from src.models.types import ConflictReport

            conflict_count = inp.get("conflict_count", 0)
            dummy_conflicts = [
                ConflictReport(
                    conflict_id=f"DUMMY-{i}",
                    conflict_type="MIXED_SIGNALS",
                    news_id=None,
                    stock_symbol="UNKNOWN",
                    news_sentiment=None,
                    stock_move=0.0,
                    sector_move=0.0,
                    explanation="",
                    resolution="",
                )
                for i in range(conflict_count)
            ]
            score = self._ranker.compute_confidence_score(
                portfolio_id=inp["portfolio_id"],
                news_ids=inp.get("news_ids") or [],
                conflict_reports=dummy_conflicts,
            )
            return score.model_dump()

        return {"error": f"Unknown tool '{name}'", "available": [t["function"]["name"] for t in TOOL_DEFINITIONS]}

    def _rag_news_query(self, inp: dict, max_results: int) -> list:
        if self._vector_store is None or not self._vector_store.available:
            return []
        parts = (inp.get("sectors") or []) + (inp.get("stocks") or []) + (inp.get("indices") or [])
        if not parts:
            return []
        query_text = " ".join(parts)
        try:
            articles = self._vector_store.query(query_text, n_results=max_results)
            scope_filter = inp.get("scope_filter")
            impact_filter = inp.get("impact_filter")
            if scope_filter:
                articles = [a for a in articles if a.scope == scope_filter]
            if impact_filter:
                articles = [a for a in articles if a.impact_level == impact_filter]
            return articles
        except Exception:
            return []
