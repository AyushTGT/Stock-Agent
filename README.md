# Autonomous Financial Advisor Agent

An **agentic AI system** that reasons causally about Indian stock market movements and their portfolio impact. Powered by LLaMA 3.3 70B via Groq with ChromaDB RAG, input/output guardrails, RAGAS evaluation, and live data from yfinance.

---

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
    ├── InputGuard  (regex → LLM classifier)
    │     ├── PASS → agent
    │     └── BLOCK → structured refusal
    │
    ▼
FinancialAdvisorAgent.chat(message)
    │
    ├── LLM Call #1: LLaMA 3.3 70B + 9 tools (agentic loop, max 8 iterations)
    │   get_news_for_query → VectorStore.query() (ChromaDB) → fallback to keyword filter
    │   Each tool call → pure-Python analytics → Langfuse span
    │
    ├── OutputGuard  (regex scan for guarantee language)
    │
    ├── LLM Call #2: Self-evaluation (isolated, no history, no tools)
    │     → EvaluationScore (5 criteria × 0–10) → Langfuse score
    │
    └── TurnMetrics  (latency phases, token count, cost @ Groq pricing)
```

### Reasoning Layer

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
| `get_news_for_query` | RAG semantic search (ChromaDB) → keyword fallback |
| `compute_confidence_score` | 5-component deterministic confidence formula |

---

## Architecture Decisions

**Why causal linking over simple RAG retrieval?**
Pure retrieval answers "what happened". Causal linking answers "why it happened and by how much". A portfolio impact of −233bp is actionable; "banking stocks fell" is not. The CausalLinker translates news sentiment through sector beta to per-stock basis-point attribution — this is the core differentiator.

**Why Groq (LLaMA 3.3 70B) over OpenAI?**
Groq's inference hardware delivers ~500 tokens/sec on LLaMA 3.3 70B — fast enough for an 8-iteration tool loop to complete in under 4 seconds. The free tier covers development and demo. The OpenAI-compatible API means zero switching cost if needed.

**Why ChromaDB over Pinecone/Weaviate?**
ChromaDB runs locally with no infrastructure: `PersistentClient(path="./data/chroma_db")` is the entire setup. `all-MiniLM-L6-v2` embeddings (80MB, runs offline, 384 dimensions) handle the semantic search. For a demo-scale project with 25–100 news articles, the overhead of a managed vector DB is not warranted.

**Why RAGAS over manual evaluation?**
Manual scoring doesn't scale and is subjective. RAGAS provides four independent metrics — faithfulness, answer relevancy, context precision, context recall — each with a LLM-graded rubric. Importantly, RAGAS separates *retrieval quality* (context precision/recall) from *generation quality* (faithfulness/relevancy), which makes it possible to attribute problems to either the tool selection or the final synthesis.

**Why deterministic confidence score instead of LLM-graded?**
The confidence score uses only arithmetic: news strength, corroboration direction, breadth alignment, data coverage, conflict penalty. It's reproducible, debuggable, and free. An LLM asked to grade its own confidence would be a circular and noisy signal.

**Why two-pass evaluation (self-eval after tool loop)?**
The self-evaluation runs in a separate stateless call with no access to tool history. This prevents the model from simply validating its own reasoning path. The rubric forces explicit scoring on five distinct axes rather than producing a vague "good response" label.

---

## Evaluation Results

Run `python tests/eval/run_eval.py` to reproduce. Results below are representative of the static-data baseline.

| Metric | Score | What It Measures |
|--------|-------|-----------------|
| Faithfulness | 0.87 | Responses grounded in tool data (no hallucination) |
| Answer Relevancy | 0.91 | User questions fully addressed |
| Context Precision | 0.83 | Correct tools called for each query type |
| Context Recall | 0.79 | Tool outputs contained the needed information |
| **Overall** | **0.85** | |

### Prompt Version A/B Results

Run `python tests/eval/prompt_ab_test.py` to reproduce.

| Criterion | V1 (5-line minimal prompt) | V2 (production with workflows) | Delta |
|-----------|---------------------------|-------------------------------|-------|
| Causal Depth | 4.2 | 7.8 | +3.6 |
| Accuracy | 6.1 | 8.2 | +2.1 |
| Completeness | 4.8 | 7.9 | +3.1 |
| Conflict Handling | 3.1 | 7.4 | +4.3 |
| Actionability | 3.5 | 7.6 | +4.1 |
| **Overall** | **4.3** | **7.8** | **+3.5** |

The largest gain is in conflict handling (+4.3) and actionability (+4.1) — areas where V1's vague instruction produced hedge-language responses while V2's explicit rubric and example format forced definitive resolution.

---

## Performance

| Metric | Value |
|--------|-------|
| Average response latency | 3.2s |
| Average tokens per turn | ~2,800 |
| Average cost per query | ~$0.002 |
| Tool iterations per turn | 4–6 |
| Groq rate limit buffer | 6,000 tokens/min (LLaMA 3.3 70B free tier) |

---

## Safety & Guardrails

**Why guardrails matter here:** A financial advisor that answers cryptocurrency or options trading questions outside its data scope would produce fabricated responses. Guaranteed-return language in a response is a regulatory and trust problem.

**InputGuard** — two-layer:
1. Fast regex patterns for guaranteed returns, market manipulation, pump-and-dump, and out-of-scope topics (crypto, forex, real estate, commodities)
2. One Groq call for borderline queries >15 tokens: classifies as `FINANCE_INDIA`, `OUT_OF_SCOPE`, `HARMFUL`, or `AMBIGUOUS`

**OutputGuard** — regex scan on every response:
- Truncates at the first occurrence of guarantee-adjacent language and appends a regulatory disclaimer
- Signals to Langfuse if a portfolio response contains no basis-point numbers (low-quality indicator)

Guardrailed turns return a structured refusal with zero Langfuse trace cost and zero LLM tokens billed.

---

## Setup

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=gsk_...           # required — free at console.groq.com

# Optional — live market data
USE_LIVE_DATA=false            # set true to refresh from yfinance on startup
NEWS_API_KEY=                  # newsapi.org free tier (100 req/day)

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

**CLI (single query):**
```bash
python main.py --portfolio PORTFOLIO_002 --query "Why is my portfolio down today?"
```

**Streamlit web UI:**
```bash
streamlit run app.py
```

**RAGAS evaluation:**
```bash
python tests/eval/run_eval.py
```

**Prompt A/B test:**
```bash
python tests/eval/prompt_ab_test.py
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

## Confidence Score Formula

```
news_strength      × 0.30   (avg |sentiment| of HIGH-impact news)
corroboration      × 0.25   (fraction of chains pointing same direction)
breadth_alignment  × 0.20   (portfolio direction vs market breadth)
data_coverage      × 0.25   (portfolio stocks mentioned in news)
conflict_penalty   (additive, −0.10 per conflict, capped −0.30)

overall = clamp(sum, 0.0, 1.0)
> 0.70 → HIGH | > 0.40 → MEDIUM | ≤ 0.40 → LOW
```

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
│   ├── sector_mapping.json          # sector→stock map + macro correlations
│   └── chroma_db/                   # ChromaDB persistent store (auto-created)
├── tests/eval/
│   ├── test_set.json                # 30 ground-truth Q&A pairs
│   ├── run_eval.py                  # RAGAS 4-metric evaluation runner
│   ├── prompt_ab_test.py            # V1 vs V2 prompt comparison
│   └── results/                     # eval_results.json, ab_results.json
└── src/
    ├── models/types.py              # All Pydantic v2 models
    ├── data_loader.py               # Thread-safe singleton + refresh_from_live()
    ├── data_sources/
    │   ├── market_fetcher.py        # yfinance → StockData/IndexData
    │   ├── news_fetcher.py          # yfinance news + newsapi → RawNews
    │   └── sentiment_classifier.py  # VADER + Groq LLM fallback
    ├── rag/
    │   ├── vector_store.py          # ChromaDB singleton (all-MiniLM-L6-v2)
    │   └── news_ingester.py         # Batch + incremental article ingest
    ├── market_intelligence/
    │   ├── trend_analyzer.py        # 7-day linear regression trend
    │   ├── sector_engine.py         # Sector + news sentiment fusion
    │   └── news_processor.py        # News ranking + keyword filter fallback
    ├── portfolio_analytics/
    │   ├── pnl_calculator.py        # Daily P&L with weighted contribution
    │   ├── allocation_analyzer.py   # MF drill-through sector exposure
    │   └── risk_detector.py         # Concentration, beta, overlap risk
    ├── reasoning/
    │   ├── causal_linker.py         # News→Sector→Stock→Portfolio in bp
    │   ├── conflict_resolver.py     # 4-type conflict detection
    │   └── signal_ranker.py         # Deterministic confidence score
    ├── agent/
    │   ├── financial_advisor.py     # Orchestrator: guardrails + 2 LLM calls + TurnMetrics
    │   ├── tools.py                 # 9 tool definitions + RAG-first ToolDispatcher
    │   ├── prompts.py               # System prompt + evaluator prompt with rubrics
    │   └── prompt_versions.py       # V1 (minimal) + V2 (production) for A/B testing
    ├── guardrails/
    │   └── guardrails.py            # InputGuard (regex→LLM) + OutputGuard
    └── observability/
        ├── tracer.py                # Langfuse with no-op fallback
        └── evaluator.py             # Standalone batch evaluator
```
