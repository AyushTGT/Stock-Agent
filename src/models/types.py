from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class OHLCBar(BaseModel):
    date: str
    close: float
    volume: Optional[int] = None
    change_percent: Optional[float] = None


class IndexHistory(BaseModel):
    data: list[OHLCBar]
    trend: str
    cumulative_change_percent: float
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None


class StockHistory(BaseModel):
    data: list[OHLCBar]
    trend: str
    avg_volume_7d: Optional[float] = None
    volatility: Optional[str] = None


class SectorWeeklyPerformance(BaseModel):
    weekly_change_percent: float
    trend: str
    catalyst: str


class MarketBreadth(BaseModel):
    advances: int
    declines: int
    unchanged: int
    advance_decline_ratio: float


class FIIDIIData(BaseModel):
    buy_value_cr: float
    sell_value_cr: float
    net_value_cr: float
    mtd_net_cr: float
    ytd_net_cr: float


class IndexData(BaseModel):
    name: str
    current_value: float
    previous_close: float
    change_percent: float
    change_absolute: float
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    week_52_high: Optional[float] = Field(None, alias="52_week_high")
    week_52_low: Optional[float] = Field(None, alias="52_week_low")
    sentiment: str

    model_config = {"populate_by_name": True}


class SectorPerformance(BaseModel):
    change_percent: float
    sentiment: str
    key_drivers: list[str]
    top_gainers: list[str] = []
    top_losers: list[str] = []


class StockData(BaseModel):
    name: str
    sector: str
    sub_sector: Optional[str] = None
    current_price: float
    previous_close: float
    change_percent: float
    change_absolute: float
    volume: Optional[int] = None
    avg_volume_20d: Optional[int] = None
    market_cap_cr: Optional[float] = None
    pe_ratio: Optional[float] = None
    beta: float = 1.0
    week_52_high: Optional[float] = Field(None, alias="52_week_high")
    week_52_low: Optional[float] = Field(None, alias="52_week_low")

    model_config = {"populate_by_name": True}


class NewsEntities(BaseModel):
    sectors: list[str] = []
    stocks: list[str] = []
    indices: list[str] = []


class NewsArticle(BaseModel):
    id: str
    headline: str
    summary: str
    published_at: str
    source: str
    sentiment: Literal["POSITIVE", "NEGATIVE", "MIXED", "NEUTRAL"]
    sentiment_score: float
    scope: Literal["MARKET_WIDE", "SECTOR_SPECIFIC", "STOCK_SPECIFIC"]
    impact_level: Literal["HIGH", "MEDIUM", "LOW"]
    entities: NewsEntities
    causal_factors: list[str]
    conflict_flag: bool = False
    conflict_explanation: Optional[str] = None


class MFHolding(BaseModel):
    scheme_code: str
    scheme_name: str
    category: str
    units: float
    avg_nav: float
    current_nav: float
    weight_in_portfolio: float


class StockHolding(BaseModel):
    symbol: str
    name: str
    sector: str
    quantity: int
    avg_buy_price: float
    current_price: float
    weight_in_portfolio: float


class PortfolioHoldings(BaseModel):
    stocks: list[StockHolding]
    mutual_funds: list[MFHolding]


class Portfolio(BaseModel):
    user_id: str
    user_name: str
    portfolio_type: str
    risk_profile: str
    investment_horizon: str
    description: str
    total_investment: float
    current_value: float
    holdings: PortfolioHoldings


class MFTopHolding(BaseModel):
    symbol: str
    weight: float
    sector: str


class MutualFundScheme(BaseModel):
    scheme_code: str
    scheme_name: str
    amc: str
    category: str
    risk_rating: str
    current_nav: float
    previous_nav: float
    nav_change_percent: float
    aum_cr: float
    expense_ratio: float
    benchmark: str
    returns: dict[str, float]
    top_holdings: list[MFTopHolding] = []
    sector_allocation: dict[str, float] = {}


class SectorInfo(BaseModel):
    description: str
    stocks: list[str]
    rate_sensitive: bool = False
    defensive: bool = False
    export_oriented: bool = False
    cyclical: bool = False
    commodity_linked: bool = False


class MacroCorrelation(BaseModel):
    event: str
    negative_impact: list[str]
    positive_impact: list[str]
    neutral: list[str]


class HoldingPnL(BaseModel):
    symbol: str
    name: str
    holding_type: Literal["STOCK", "MUTUAL_FUND"]
    sector: str
    current_value: float
    invested_value: float
    day_pnl_absolute: float
    day_pnl_percent: float
    overall_gain_loss: float
    overall_gain_loss_percent: float
    portfolio_weight: float
    contribution_to_day_pnl: float  # portfolio_weight × day_pnl_percent


class PortfolioSummary(BaseModel):
    portfolio_id: str
    owner_name: str
    total_value: float
    total_invested: float
    day_pnl_absolute: float
    day_pnl_percent: float
    overall_gain_loss: float
    overall_gain_loss_percent: float
    holdings: list[HoldingPnL]
    top_gainer: Optional[str] = None
    top_loser: Optional[str] = None


class AllocationBreakdown(BaseModel):
    by_sector: dict[str, float]
    by_asset_type: dict[str, float]
    mf_indirect_sector_exposure: dict[str, dict[str, float]] = {}


class RiskFlag(BaseModel):
    risk_type: Literal["CONCENTRATION_SECTOR", "CONCENTRATION_STOCK", "HIGH_BETA", "MF_OVERLAP"]
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    subject: str
    current_weight: float
    threshold: float
    explanation: str


class RiskAssessment(BaseModel):
    overall_risk: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    portfolio_beta: float
    flags: list[RiskFlag]
    summary: str


class IndexTrendDetail(BaseModel):
    current_value: float
    change_percent: float
    seven_day_trend: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    slope: float
    cumulative_7d_change: float


class MarketTrend(BaseModel):
    overall_trend: Literal["BULLISH", "BEARISH", "NEUTRAL", "MIXED"]
    indices: dict[str, IndexTrendDetail]
    fii_net_cr: float
    dii_net_cr: float
    fii_dii_summary: str
    breadth_advances: int
    breadth_declines: int
    advance_decline_ratio: float
    breadth_signal: str
    sentiment_indicator: str


class SectorAnalysis(BaseModel):
    sector: str
    current_sentiment: str
    today_change_percent: float
    weekly_return: float
    weekly_trend: str
    news_sentiment_avg: float
    news_count: int
    key_drivers: list[str]
    top_stocks: list[str]
    catalyst: str


class StockImpact(BaseModel):
    symbol: str
    portfolio_weight: float
    stock_change_percent: float
    contribution_bp: float


class CausalLink(BaseModel):
    news_id: str
    headline: str
    sentiment_score: float
    impact_level: str
    scope: str
    causal_factors: list[str]
    affected_sector: str
    sector_change_percent: float
    portfolio_stocks_impacted: list[StockImpact]
    portfolio_contribution_bp: float


class CausalChain(BaseModel):
    chain_id: str
    narrative: str
    links: list[CausalLink]
    total_portfolio_impact_bp: float
    total_portfolio_impact_percent: float


class ConflictReport(BaseModel):
    conflict_id: str
    conflict_type: Literal[
        "POSITIVE_NEWS_STOCK_FALLING",
        "NEGATIVE_NEWS_STOCK_RISING",
        "SECTOR_STOCK_DIVERGENCE",
        "MIXED_SIGNALS",
    ]
    news_id: Optional[str]
    stock_symbol: str
    news_sentiment: Optional[str]
    stock_move: float
    sector_move: float
    explanation: str
    resolution: str


class ConfidenceComponents(BaseModel):
    news_strength: float
    corroboration: float
    breadth_alignment: float
    data_coverage: float
    conflict_penalty: float


class ConfidenceScore(BaseModel):
    overall: float
    components: ConfidenceComponents
    interpretation: Literal["HIGH", "MEDIUM", "LOW"]
    detail: str


class EvaluationScore(BaseModel):
    causal_depth: float = Field(ge=0, le=10)
    accuracy: float = Field(ge=0, le=10)
    completeness: float = Field(ge=0, le=10)
    conflict_handling: float = Field(ge=0, le=10)
    actionability: float = Field(ge=0, le=10)
    overall: float = Field(ge=0, le=10)
    justification: str


class GuardResult(BaseModel):
    allowed: bool
    category: str
    reason: str


class TurnMetrics(BaseModel):
    total_latency_ms: float
    tool_loop_latency_ms: float
    eval_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    tool_calls_count: int
