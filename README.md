# Autonomous Financial Advisor Agent

An **agentic AI system** that reasons causally about Indian stock market movements and their portfolio impact. Powered by LLaMA 3.3 70B via Groq with Langfuse observability.

## What Makes This Different

Most financial chatbots retrieve data and summarize it. This agent **reasons causally**:

```
RBI Hawkish Stance (−0.72 sentiment, HIGH impact)
  → Banking sector −2.45%
    → HDFC Bank (22.6% portfolio weight) −3.51% → −79.3bp portfolio impact
    → ICICI Bank (13.8% weight) −3.13% → −43.2bp
    → SBI (14.7% weight) −3.02% → −44.4bp
  Total: −233bp (−2.33%) portfolio P&L impact
```

It also detects and resolves **conflicting signals**:
- Bajaj Finance: strong earnings guidance (positive news) but stock falls — resolved: sector headwinds from RBI dominate
- HUL: weak volume growth (negative news) but stock rises — resolved: defensive buying in risk-off environment

---

## Architecture

```
User Query
    │
    ▼
FinancialAdvisorAgent.chat(message)
    │
    ├── LLM Call #1: LLaMA 3.3 70B + 9 tools (agentic loop, max 8 iterations)
    │   Each tool call → pure-Python analytics → Langfuse span
    │
    └── LLM Call #2: Self-evaluation (isolated, no history, no tools)
        → EvaluationScore (5 criteria × 0–10) → Langfuse score
```

### Reasoning Layer (the differentiator)

| Module | Purpose |
|--------|---------|
| `causal_linker.py` | News → Sector → Stock → Portfolio in basis points |
| `conflict_resolver.py` | 4-type conflict detection + resolution |
| `signal_ranker.py` | Deterministic confidence score (no LLM) |

### 9 Agent Tools

| Tool | What It Does |
|------|-------------|
| `get_portfolio_summary` | Daily P&L with weighted contribution per holding |
| `get_allocation_breakdown` | Sector + asset type allocation with MF drill-through |
| `get_risk_assessment` | Concentration, beta, MF overlap risk flags |
| `get_market_trend` | 7-day regression slope for 5 indices + FII/DII |
| `get_sector_analysis` | Weekly return, sentiment avg, key drivers per sector |
| `get_causal_chain` | Full quantified news→sector→stock→portfolio chains |
| `get_conflict_analysis` | Contradictory signal detection + resolution |
| `get_news_for_query` | Filter + rank news by sector/stock/index |
| `compute_confidence_score` | 5-component deterministic confidence formula |

---

## Setup

### 1. Clone and install

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=gsk_...          # required — free at console.groq.com

# Optional — Langfuse observability
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Run

**CLI (interactive):**
```bash
python main.py
```

**CLI (direct portfolio + single query):**
```bash
python main.py --portfolio PORTFOLIO_002 --query "Why is my portfolio down today?"
```

**Streamlit web UI:**
```bash
streamlit run app.py
```

---

## Example Queries

```
Why is my portfolio down today? Give me the full causal breakdown.
Which news events had the biggest impact on my holdings?
Are there any conflicting signals I should know about?
What is my sector exposure and where am I most concentrated?
What's the overall market trend and how does it affect me?
What should I do to reduce my concentration risk?
```

---

## Three Portfolio Scenarios

| Portfolio | Owner | Profile | Key Insight |
|-----------|-------|---------|-------------|
| `PORTFOLIO_001` | Rahul Sharma | Diversified (8 stocks + 4 MFs) | Balanced IT gains offset by Banking losses |
| `PORTFOLIO_002` | Priya Patel | Banking-heavy (CRITICAL) | 91.58% Banking+FS → −233bp from RBI news alone |
| `PORTFOLIO_003` | Arun Krishnamurthy | Conservative (34% debt) | Near-flat despite broad selloff — FMCG + debt buffer |

---

## Observability (Langfuse)

When `LANGFUSE_SECRET_KEY` is set, every turn creates:

```
TRACE: financial-advisor-chat
  ├── GENERATION: groq-call-1 (input tokens, output tokens)
  ├── SPAN: tool:get_portfolio_summary
  ├── SPAN: tool:get_causal_chain
  ├── SPAN: tool:get_conflict_analysis
  ├── GENERATION: self-evaluation
  └── SCORE: overall_quality (0–1 scale, 5-criterion mean)
```

**Batch retroactive scoring:**
```python
from src.observability.evaluator import ResponseEvaluator
evaluator = ResponseEvaluator()
count = evaluator.batch_evaluate_from_langfuse(limit=50)
print(f"Scored {count} traces")
```

---

## Confidence Score Formula

```
news_strength    × 0.30   (avg |sentiment| of HIGH-impact news)
corroboration    × 0.25   (fraction of chains pointing same direction)
breadth_alignment × 0.20  (portfolio direction vs market breadth)
data_coverage    × 0.25   (portfolio stocks mentioned in news)
conflict_penalty (additive, −0.10 per conflict, capped −0.30)

overall = clamp(sum, 0.0, 1.0)
> 0.70 → HIGH | > 0.40 → MEDIUM | ≤ 0.40 → LOW
```

---

## Project Structure

```
Stock Agent/
├── main.py                          # CLI entry point
├── app.py                           # Streamlit web UI
├── requirements.txt
├── .env.example
├── data/
│   ├── market_data.json             # 40+ stocks, 5 indices, 10 sectors
│   ├── historical_data.json         # 7-day OHLC, FII/DII, market breadth
│   ├── news_data.json               # 25 articles with sentiment + conflict flags
│   ├── portfolios.json              # 3 user portfolios
│   ├── mutual_funds.json            # 12 MF schemes with sector allocation
│   └── sector_mapping.json          # sector→stock map + macro correlations
└── src/
    ├── models/types.py              # All Pydantic v2 models
    ├── data_loader.py               # Thread-safe singleton loader
    ├── market_intelligence/
    │   ├── trend_analyzer.py        # 7-day linear regression trend
    │   ├── sector_engine.py         # Sector + news sentiment fusion
    │   └── news_processor.py        # News ranking + digest builder
    ├── portfolio_analytics/
    │   ├── pnl_calculator.py        # Daily P&L with weighted contribution
    │   ├── allocation_analyzer.py   # MF drill-through sector exposure
    │   └── risk_detector.py         # Concentration, beta, overlap risk
    ├── reasoning/
    │   ├── causal_linker.py         # News→Sector→Stock→Portfolio in bp
    │   ├── conflict_resolver.py     # 4-type conflict detection
    │   └── signal_ranker.py         # Deterministic confidence score
    ├── agent/
    │   ├── financial_advisor.py     # Orchestrator: 2 LLM calls per turn
    │   ├── tools.py                 # 9 tool definitions + ToolDispatcher
    │   └── prompts.py               # System prompt + evaluator prompt
    └── observability/
        ├── tracer.py                # Langfuse with no-op fallback
        └── evaluator.py             # Standalone batch evaluator
```
