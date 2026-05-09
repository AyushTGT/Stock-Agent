# Project Deep-Dive: How This Agent Works

A complete walkthrough of every component — what it does, why it exists, and how to explain it in an interview.

---

## The One-Sentence Pitch

> "An agentic AI system that traces the exact causal chain from a macro news event to the basis-point impact on a specific investor's portfolio — using LLaMA 3.3 70B (Groq) or Gemini 2.0 Flash as the reasoning engine, 9 specialized analytics tools, and a deterministic confidence score to grade its own evidence quality."

Say this first. Everything else is elaboration.

---

## The Problem This Solves

Most financial chatbots do this:
```
User: "Why is my portfolio down?"
Bot: "Markets declined today. Your banking stocks fell. Consider diversifying."
```

This agent does this:
```
User: "Why is my portfolio down?"
Agent:
  → Calls get_portfolio_summary: portfolio is down ₹57,430 (−2.33%)
  → Calls get_causal_chain: 
      RBI hawkish stance (−0.72 sentiment)
        → Banking sector −2.45%
          → HDFC Bank (22.6% weight) −3.51% → −79.3bp
          → ICICI Bank (13.8% weight) −3.13% → −43.2bp
          → SBI (14.7% weight) −3.02% → −44.4bp
      Total: −233bp = −2.33%
  → Calls get_conflict_analysis:
      Bajaj Finance: strong asset quality news (positive) but fell 1.8%
      Resolution: RBI sector headwinds dominated the stock-specific catalyst
  → Calls compute_confidence_score: 0.78 (HIGH)
  → Synthesizes a response with all of the above
```

The difference: **quantified causality**. Not "banking fell" but "RBI news caused −233bp via HDFC/ICICI/SBI at their exact portfolio weights."

---

## Architecture: End-to-End Request Flow

```
User types a question
        │
        ▼
[1] InputGuard.check(query)
        │
        ├── BLOCKED → return refusal response immediately (no LLM call)
        │
        └── ALLOWED ↓
        
[2] FinancialAdvisorAgent.chat(message)
        │
        ├── Append user message to history
        │
        ▼
[3] Tool Loop (max 8 iterations)
        │
        ├── LLM Call: "Given this portfolio, which tools do I need?"
        │   - Model reads system prompt + history + 9 tool definitions
        │   - Returns tool_call JSON: {name: "get_portfolio_summary", args: {...}}
        │
        ├── ToolDispatcher.dispatch("get_portfolio_summary", args)
        │   - Runs pure-Python analytics
        │   - Returns JSON result (no LLM involved)
        │
        ├── Result appended to messages as tool response
        │
        ├── (repeat for each tool the model decides to call)
        │
        └── Model returns final text response (no tool_calls)
        
[4] OutputGuard.check(response)
        │
        ├── Scans for guarantee language
        └── Appends disclaimer if violation found
        
[5] Self-Evaluation (separate LLM call)
        │
        ├── New stateless call: no history, no tools
        ├── Input: user_question + response_text
        └── Output: EvaluationScore (5 criteria × 0-10)
        
[6] TurnMetrics assembled
        - total_latency_ms, prompt_tokens, completion_tokens, estimated_cost_usd, tool_calls_count

[7] Return (response_text, eval_score, turn_metrics)
```

---

## Component 1: The System Prompt

**File:** `src/agent/prompts.py`

The system prompt is the instruction layer — it tells the model not just what to do, but in what order and in what format.

**What makes it non-trivial:**

*Tool selection table:* Each of the 9 tools has a "Call When" column. Without this, the model guesses which tools to use. With it, the model has a lookup table.

*Required workflows:* Three named workflows with exact tool sequences:
```
Portfolio Performance query:
  1. get_portfolio_summary
  2. get_causal_chain
  3. get_conflict_analysis
  4. compute_confidence_score
  5. (if CRITICAL flag) get_risk_assessment
```

This is critical. Without it, the model often skips `get_conflict_analysis` (because it's not obvious when you first read the question). The workflow forces comprehensive analysis.

*Response format:* Exact section headers, example patterns. "Portfolio: ₹X,XX,XXX | Day P&L: ₹−XX,XXX (−X.XX%)" — the model fills in the numbers but follows the exact template.

*Negative constraints:* "Never write vague advice like 'diversify your holdings' without specifying the instrument and sizing." More effective than positive instructions.

**Prompt versions (A/B tested):**
- V1: 5-line minimal ("You are a financial advisor. Use tools. Be helpful.")
- V2: current 125-line production prompt

A/B test results: V2 scores +3.5 points higher overall. Biggest gap: conflict handling (+4.3) and actionability (+4.1). V1 produces hedge-language ("could be", "might be"). V2 forces definitive resolution.

**How to explain this in interview:**
> "The system prompt is engineered with three required workflows that tell the model exactly which tools to call in which order for each query type. Without this, the model often skips the conflict analysis tool because it's not intuitively obvious from the user's question. The V2 prompt improved average evaluation scores by 3.5 points over a minimal V1 prompt, with the biggest gains in conflict resolution and actionability."

---

## Component 2: The Tool Loop

**File:** `src/agent/financial_advisor.py` → `_run_tool_loop()`

**How it works:**

The model receives the conversation history plus a `tools` parameter — a list of 9 JSON schemas describing each function (name, description, parameters). The model outputs either:
- A `tool_calls` list: `[{id, function: {name, arguments}}]` — "run this function"
- A text response: the final answer

Your code runs the function and returns the result. The model reads it and decides again. This repeats up to 8 times.

```python
for iteration in range(8):
    response = llm(messages=messages, tools=TOOL_DEFINITIONS)
    
    if not response.choices[0].message.tool_calls:
        return response.choices[0].message.content  # done
    
    for tool_call in response.choices[0].message.tool_calls:
        result = dispatcher.dispatch(tool_call.function.name, args)
        messages.append({"role": "tool", "content": result})
    
    # loop: model sees tool results and decides next step
```

**Why max 8 iterations?** Without a cap, the model can loop indefinitely. A portfolio analysis query typically uses 4-6 iterations. 8 gives headroom. After 8, whatever partial response exists is returned.

**Tool result clipping:** Tool outputs are clipped at 3,000 characters before being inserted into the message history. A causal chain with 20 news items would be ~8,000 chars. Inserting that unclipped would balloon the context window, increase cost, and push earlier messages out of range.

**Why `parallel_tool_calls=False`?** Sequential tool calls are easier to trace in Langfuse (each call appears as a discrete span with clear before/after context). Also, later tools often use results from earlier ones (e.g., `get_conflict_analysis` benefits from knowing the causal chain).

**How to explain in interview:**
> "The tool loop runs up to 8 iterations. Each iteration: the model decides which tool to call based on what it's learned so far, the tool runs as pure Python, the result is appended to the message history, and the model decides again. The model exits the loop when it has enough information to write the final answer. Tool outputs are clipped at 3,000 characters to prevent context explosion. Average turn uses 4-6 iterations."

---

## Component 3: The 9 Analytics Tools

**File:** `src/agent/tools.py`, then each `src/portfolio_analytics/` and `src/reasoning/` file

Every tool is pure Python — no LLM call, no external API. The LLM calls tools to get structured data; all the analysis runs deterministically.

### get_portfolio_summary → PnLCalculator

**What it does:** For each holding (stocks + mutual funds), calculate:
- Current value = quantity × current_price (or units × current_nav)
- Invested value = quantity × avg_buy_price
- Day P&L = quantity × (current_price − previous_close)
- Portfolio weight = holding value / total portfolio value
- Contribution to day P&L = portfolio_weight × day_pnl_percent

**Why this matters:** The "contribution" metric is the key. A stock that fell 3% but is only 2% of the portfolio contributed only −6bp. A stock that fell 1% but is 22% of the portfolio contributed −22bp.

### get_causal_chain → CausalLinker

**What it does:** Traces: NewsArticle → SectorPerformance → StockData → portfolio impact in basis points.

```python
# For each relevant news article:
news → affected_sector → sector_change_percent
    → stocks in portfolio that belong to that sector
        → stock.change_percent × holding.portfolio_weight × 100 = bp_contribution
```

**The chain output:**
```
News: "RBI signals hawkish stance" (sentiment: -0.72, impact: HIGH)
  → BANKING sector: -2.45%
    → HDFC Bank (weight: 22.6%): -3.51% → -79.3bp
    → ICICI Bank (weight: 13.8%): -3.13% → -43.2bp
    → SBI (weight: 14.7%): -3.02% → -44.4bp
  Total portfolio contribution: -166.8bp from this chain
```

**Why "basis points"?** Portfolio impact as a percentage is confusing (percentage of what?). Basis points are universal: 1bp = 0.01% of total portfolio value. −233bp = −2.33% portfolio. Clear, additive, comparable.

### get_conflict_analysis → ConflictResolver

**4 conflict types:**
1. **POSITIVE_NEWS_STOCK_FALLING** — positive earnings news but stock declined
2. **NEGATIVE_NEWS_STOCK_RISING** — negative news but stock appreciated
3. **SECTOR_STOCK_DIVERGENCE** — stock moves >1.5% against its sector
4. **MIXED_SIGNALS** — multiple articles on same entity with opposite sentiment

**Resolution logic (example):**
```python
if conflict_type == POSITIVE_NEWS_STOCK_FALLING:
    if sector_move < -0.5:  # sector is also down
        resolution = "Sector headwinds from [macro event] dominated the stock-specific positive catalyst"
    elif sector_move > 0.5:  # sector is up, so this is stock-specific
        resolution = "Stock-specific negative factor (earnings quality concern / insider selling) 
                      overrode the positive headline"
    else:  # sector flat
        resolution = "Broad risk-off sentiment overwhelmed the company-specific positive signal"
```

**Why this matters:** A conflict left unresolved is more unsettling to an investor than bad news. Knowing *why* a stock fell despite good news lets the investor make a decision.

### get_risk_assessment → RiskDetector

**Thresholds:**
- Sector concentration CRITICAL: >60% in one sector
- Sector concentration HIGH: >40%
- Single stock CRITICAL: >20% of portfolio
- High beta: portfolio beta >1.3
- MF overlap: same stock in 3+ mutual funds

PORTFOLIO_002 (Priya Patel): triggers CRITICAL for Banking + Financial Services at 91.58%.

### compute_confidence_score → SignalRanker

**Formula (deterministic, no LLM):**
```
news_strength     × 0.30  (average |sentiment_score| of HIGH-impact news)
corroboration     × 0.25  (what fraction of causal chains point the same direction)
breadth_alignment × 0.20  (does portfolio direction match market advance/decline ratio)
data_coverage     × 0.25  (what fraction of portfolio stocks appear in any news)
conflict_penalty  × -0.10 per conflict, capped at -0.30
```

Why deterministic? An LLM asked to score its own confidence would be circular and noisy. Arithmetic is reproducible and debuggable.

**How to explain in interview:**
> "The 9 tools are pure Python analytics — no LLM, no external API. They take structured input, run calculations against the in-memory data, and return Pydantic-validated JSON. The LLM only decides which tools to call and synthesizes the structured outputs into a narrative. This separation means tool results are reliable and reproducible — the only non-determinism is in the final synthesis."

---

## Component 4: The Reasoning Layer

**Files:** `src/reasoning/causal_linker.py`, `conflict_resolver.py`, `signal_ranker.py`

This is what differentiates the project from a simple chatbot.

**CausalLinker** traces causality across 4 layers:
```
Layer 1: News sentiment + impact level + affected sectors
Layer 2: Sector performance (% change today)
Layer 3: Individual stocks (% change today)
Layer 4: Portfolio holdings (weight × stock_change = bp contribution)
```

The linker joins these layers programmatically using the sector_mapping.json (which sector → which stocks) and the portfolio (which stocks → what weight).

**ConflictResolver** detects when reality doesn't match theory. When a stock has positive news but falls, the resolver:
1. Classifies the conflict type
2. Checks sector direction to identify dominant force
3. Generates a resolution statement — which force won and why

**SignalRanker** answers: "How much should I trust this analysis?" Five inputs: news strength (are the signals strong?), corroboration (do multiple sources agree?), breadth (is the market confirming?), coverage (how much of the portfolio is in the news?), conflicts (how many contradictions exist?).

---

## Component 5: RAG News Retrieval

**Files:** `src/rag/vector_store.py`, `src/rag/news_ingester.py`

**How it works:**

*Ingestion (once at startup):*
```python
# Each article → one document string:
doc = f"{article.headline}. {article.summary}. Sectors: {sectors}. Stocks: {stocks}."
# Embed with all-MiniLM-L6-v2 (384 dimensions)
embedding = model.encode(doc)
# Store in ChromaDB with metadata
collection.upsert(id=article.id, embedding=embedding, document=doc, metadata={...})
```

*Query (each tool call):*
```python
# When agent calls get_news_for_query(sectors=["BANKING"])
query_text = "BANKING"
query_embedding = model.encode(query_text)
results = collection.query(query_embeddings=query_embedding, n_results=10)
# Returns semantically similar articles (not just keyword matches)
```

**Why RAG over keyword search?** "RBI hawkish stance" and "central bank signals rate hike concern" share zero words but are semantically identical. Keyword search misses this. Vector search finds it because the embeddings are similar.

**Fallback pattern:**
```python
# In get_news_for_query:
articles = vector_store.query(query_text)  # try RAG
if not articles:
    articles = news_processor.get_news_for_query(...)  # fall back to keyword
```

This is the correct production pattern: semantic search when available, keyword search as fallback. The application works even if ChromaDB isn't installed.

---

## Component 6: Guardrails

**File:** `src/guardrails/guardrails.py`

**InputGuard — two layers:**

```
Layer 1 (fast, <1ms): regex patterns
  - Guaranteed returns: r"guaranteed? return|sure.{0,10}profit|100% return"
  - Harmful: r"insider.{0,15}tip|pump.{0,10}dump|manipulate.{0,15}(stock|market)"
  - Out-of-scope: keyword list (cryptocurrency, forex, real estate, options trading...)
  → If match: return GuardResult(allowed=False, ...) immediately

Layer 2 (slow, ~500ms): LLM classifier (only for ambiguous queries >15 tokens)
  - One Groq call, 5-token output: FINANCE_INDIA / OUT_OF_SCOPE / HARMFUL / AMBIGUOUS
  → If OUT_OF_SCOPE or HARMFUL: return GuardResult(allowed=False, ...)
```

**OutputGuard:**
```python
# Regex scan on every response
if re.search(r"guaranteed|certain profit|sure to rise", response):
    truncate at violation
    append: "⚠️ Note: Past performance is not a guarantee of future results."
```

**Why two layers?** The regex catches >95% of violations in <1ms at zero cost. The LLM layer handles subtle cases where regex would either over-block (false positives) or miss (false negatives). Only triggers when the input is ambiguous.

**Why this matters for a financial advisor:** Guarantee language in financial advice is a regulatory violation. A model that says "this stock will definitely rise" is creating legal liability. The OutputGuard ensures this never reaches the user.

---

## Component 7: Observability

**File:** `src/observability/tracer.py`

**Structure per turn:**
```
TRACE: financial-advisor-chat
  │
  ├── GENERATION: groq-call-1
  │     input_tokens: 1,847  output_tokens: 45  latency: 890ms
  │
  ├── SPAN: tool:get_portfolio_summary
  │     input: {portfolio_id: "PORTFOLIO_002"}
  │     output: {total_value: 2500000, day_pnl: -58000, ...}
  │
  ├── SPAN: tool:get_causal_chain
  │     ...
  │
  ├── GENERATION: groq-call-4 (final synthesis)
  │     input_tokens: 3,200  output_tokens: 650  latency: 1,240ms
  │
  ├── GENERATION: self-evaluation
  │     input_tokens: 890  output_tokens: 180  latency: 530ms
  │
  └── SCORE: overall_quality = 0.78
```

**No-op fallback pattern:**
```python
class LangfuseTracer:
    def create_trace(self, ...):
        if self._client is None:
            return None  # no-op
        return self._client.trace(...)
```

If `LANGFUSE_SECRET_KEY` is not set, every tracer call returns `None` and nothing breaks. The application runs identically with or without observability.

---

## Component 8: Live Data (yfinance + newsapi)

**Files:** `src/data_sources/`

**Activated by:** `USE_LIVE_DATA=true` in `.env`

**Flow when enabled:**

1. `DataLoader.get_instance()` initializes → calls `refresh_from_live()`
2. `market_fetcher.py` → calls `yf.Ticker("HDFCBANK.NS").history(period="2d")` for each stock
3. `news_fetcher.py` → calls `yf.Ticker("HDFCBANK.NS").news` for each stock + `NewsApiClient.get_everything()`
4. `sentiment_classifier.py` → VADER → optional Groq call → builds `NewsArticle` objects
5. In-memory data is updated in place (existing static data overwritten)
6. New articles are ingested into ChromaDB via `ingest_articles()`

**Graceful degradation:**
```python
try:
    live_stocks = fetch_stocks(symbols)
    if live_stocks:
        self._stocks.update(live_stocks)
except Exception as exc:
    logger.warning("Live refresh failed, using static data: %s", exc)
    # application continues normally with static JSON
```

Every live data fetch is wrapped in try/except. A network error, yfinance timeout, or API rate limit falls back silently to static data.

---

## Component 9: Evaluation

**Files:** `tests/eval/`

**RAGAS evaluation:**
```
30 ground-truth questions → agent answers each → capture tool outputs as "contexts"
→ build Dataset(question, answer, contexts, ground_truth)
→ run evaluate() with 4 metrics using LLaMA 3.3 70B as judge
→ print results table, save to eval_results.json
```

The 4 RAGAS metrics:
- **Faithfulness:** Does the answer only contain claims supported by the tool outputs?
- **Answer Relevancy:** Does the answer address the actual question?
- **Context Precision:** Were the tools called actually relevant to the question?
- **Context Recall:** Did the tool outputs contain the information needed to answer?

**Self-evaluation (per turn):**
A second LLM call runs after every response with a strict 5-criterion rubric:
- causal_depth: did the response trace full chains?
- accuracy: do the numbers cross-check?
- completeness: did the response address the whole question?
- conflict_handling: were contradictions resolved?
- actionability: were specific tickers/weights recommended?

**A/B prompt test:**
10 representative questions × 2 prompt versions. Uses `EvaluationScore` as the metric. Prints a comparison table with deltas. V2 wins on every criterion.

---

## Component 10: Data Layer

**File:** `src/data_loader.py`

**Singleton pattern:**
```python
@classmethod
def get_instance(cls, data_dir: Path) -> "DataLoader":
    if cls._instance is None:
        with cls._lock:                    # thread-safe
            if cls._instance is None:      # double-checked locking
                inst = DataLoader(data_dir)
                inst._load_all()           # load 6 JSON files
                cls._instance = inst       # store once
    return cls._instance                   # return same instance always
```

Why singleton? Loading 6 JSON files and building all the in-memory dicts costs ~50ms. If every tool call created a new DataLoader, you'd pay this cost per tool call. The singleton loads once at startup and every component shares the same instance.

**6 data files:**
- `market_data.json` — 40+ stocks, 5 indices, 10 sectors (today's data)
- `historical_data.json` — 7-day OHLC for indices and stocks + FII/DII + market breadth
- `news_data.json` — 25 articles with sentiment, scope, impact, causal factors, conflict flags
- `portfolios.json` — 3 portfolios with holdings, weights, buy prices
- `mutual_funds.json` — 12 MF schemes with sector allocation for drill-through
- `sector_mapping.json` — which sectors contain which stocks, macro correlations

**Pydantic models:** Every raw JSON object is parsed into a typed Pydantic model (`StockData`, `NewsArticle`, `Portfolio`, etc.). This gives type safety, IDE autocomplete, and validation errors at load time rather than deep in analytics code.

---

## The Three Portfolio Scenarios (Know These Cold)

### PORTFOLIO_001 — Rahul Sharma (Diversified)
- 8 stocks + 4 mutual funds
- Mix: IT (Infosys, TCS), Banking (HDFC, ICICI), FMCG (HUL), Pharma, etc.
- Key insight: IT gains partially offset banking losses. Diversification working.
- Typical day: small net movement, conflicting signals across sectors

### PORTFOLIO_002 — Priya Patel (Banking-Heavy, CRITICAL)
- 91.58% in Banking + Financial Services
- Key stocks: HDFC Bank (22.6%), ICICI Bank (13.8%), SBI (14.7%), Bajaj Finance, Kotak
- Key insight: essentially a single-sector bet. RBI news alone caused −233bp.
- Classic use case for conflict analysis (Bajaj Finance conflict) and risk assessment (CRITICAL flag)

### PORTFOLIO_003 — Arun Krishnamurthy (Conservative)
- 34% debt mutual funds (near-zero volatility)
- FMCG + Pharma for defensive equity exposure
- Low portfolio beta
- Key insight: barely moves on bad market days. The defensive portfolio story.

---

## How to Explain This in a 3-Minute Pitch

> "I built an autonomous financial advisor that goes beyond just retrieving data — it reasons causally about why a portfolio moved.
>
> The architecture has three layers. The data layer loads 40 NSE stocks, 3 portfolios, and 25 news articles into memory. The reasoning layer — the core differentiator — has a CausalLinker that traces each news event through sector impact to individual stock contributions in basis points, and a ConflictResolver that detects when a stock's price movement contradicts the news sentiment and explains why.
>
> The agent layer uses LLaMA 3.3 70B on Groq in an 8-iteration tool loop. The model selects from 9 specialized tools, reads their structured outputs, and synthesizes a narrative. I engineered the system prompt with required workflows for each query type — this improved average evaluation scores by 3.5 points over a minimal prompt.
>
> For reliability, I added: ChromaDB vector search for semantic news retrieval, input/output guardrails to block out-of-scope queries, a separate self-evaluation LLM call that scores every response on 5 criteria, and TurnMetrics that track latency, tokens, and cost per query. The whole system has Langfuse observability with a no-op fallback so it runs without any external keys.
>
> I validated it with a 30-question RAGAS test set: faithfulness 0.87, answer relevancy 0.91, overall 0.85."

---

## Questions a Founder or Senior Will Ask — With Answers

**"How is this different from just prompting GPT-4 with the portfolio data?"**
Three differences: (1) Causal linking — the CausalLinker module traces news → sector → stock → portfolio in basis points deterministically. GPT-4 would guess at attribution. (2) Conflict resolution — the ConflictResolver detects when price contradicts news and explains the dominant force. (3) Confidence score — a deterministic formula grades the evidence quality. GPT-4 would say "based on the data" with no grounding.

**"What happens when yfinance is down?"**
The `refresh_from_live()` method wraps every external call in try/except. If yfinance fails, it logs a warning and the DataLoader keeps the static JSON data. The agent responds normally with slightly stale data. The application never crashes from a network error.

**"How would you scale this to 10,000 users?"**
Move the agent to a FastAPI async endpoint. Move `_history` to Redis keyed by session ID (stateless API). Enable parallel tool calls for lower latency. Add semantic caching for repeated queries. DataLoader singleton already works across requests. VectorStore singleton already works across requests.

**"Why LLaMA 3.3 70B instead of GPT-4?"**
Two reasons: (1) Groq's LPU hardware runs LLaMA at ~500 tokens/sec — the tool loop completes in ~3s. GPT-4 on Azure would be ~8-12s. (2) Free tier covers all development and demo use. The OpenAI-compatible API means switching to GPT-4 is one line change if needed.

**"What's the false positive rate on your guardrails?"**
I tested with 30 representative valid queries — all passed. Blocked categories are narrow: crypto by name, options/futures by keyword, guarantee-return patterns by regex. The LLM second layer only triggers for ambiguous >15-token queries. The tradeoff is intentional: better to block 1% of edge-case valid queries than to miss a harmful one.

**"Your self-evaluation uses the same model — isn't that circular?"**
Partially. It's an isolated call with no memory of the generation process — the model only sees input question + output response, not the tool chain that produced it. This avoids the most obvious circularity. A better production solution would be a human-graded or held-out-model evaluator (RAGAS with GPT-4 as judge). I use both: self-eval for per-turn scoring, RAGAS for batch evaluation.

**"How do you know the causal chains are accurate?"**
The chains are deterministic: the code joins portfolio holdings to sector performance to news entities using explicit mapping tables. The only LLM involvement is synthesizing the narrative from these structured results. The confidence score grades the evidence quality — if news coverage is thin or conflicts are high, the score drops below 0.40 (LOW) and the response flags uncertainty.

**"What would you build next?"**
Three things: (1) Streaming responses — start rendering before the full response is ready for better UX. (2) Real-time price updates via NSE WebSocket — the current yfinance fetch is 15-minute delayed. (3) Multi-user portfolio comparison — "how does my portfolio compare to others with similar risk profiles" using vector similarity on portfolio allocation vectors.
