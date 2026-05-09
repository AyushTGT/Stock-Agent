# AI Engineering Interview Study Guide

Everything you need to answer every question — from this project and the broader field.
Organized by topic. Each section: what to know, how this project uses it, where to read more.

---

## Table of Contents

1. [LLMs & APIs](#1-llms--apis)
2. [Agentic Systems & Tool Use](#2-agentic-systems--tool-use)
3. [RAG — Retrieval Augmented Generation](#3-rag--retrieval-augmented-generation)
4. [Vector Databases & Embeddings](#4-vector-databases--embeddings)
5. [Prompt Engineering](#5-prompt-engineering)
6. [Evaluation & Metrics](#6-evaluation--metrics)
7. [Guardrails & Safety](#7-guardrails--safety)
8. [Observability & Monitoring](#8-observability--monitoring)
9. [Data Modeling with Pydantic](#9-data-modeling-with-pydantic)
10. [Sentiment Analysis & NLP](#10-sentiment-analysis--nlp)
11. [LLM Architecture (Theory)](#11-llm-architecture-theory)
12. [System Design for AI](#12-system-design-for-ai)
13. [Cost & Latency Optimization](#13-cost--latency-optimization)
14. [Fine-Tuning vs Prompting](#14-fine-tuning-vs-prompting)
15. [Agent Frameworks](#15-agent-frameworks)
16. [Production AI Engineering](#16-production-ai-engineering)

---

## 1. LLMs & APIs

### What You Must Know

**Core concepts:**
- LLMs are next-token predictors trained on human text. At inference, they generate one token at a time.
- Temperature: 0 = deterministic (same answer every time), 1 = creative/random. For financial data: use 0 or 0.1.
- Max tokens: hard cap on output length. Costs money. Set it tightly.
- Context window: total input + output tokens the model can "see" at once. LLaMA 3.3 70B: 128k tokens. Gemini 2.0 Flash: 1M tokens.
- System prompt: the instruction message that persists across the conversation and shapes model behavior.

**API interaction pattern:**
```python
client.chat.completions.create(
    model="llama-3.3-70b-versatile",  # or "gemini-2.0-flash" via Gemini
    messages=[
        {"role": "system", "content": "You are..."},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "previous answer"},
    ],
    max_tokens=2048,
    temperature=0,
)
```

**OpenAI-compatible API:** Many providers (Groq, Together, Anyscale, Azure) expose the same API shape as OpenAI. You point the `base_url` at them and everything else is identical. This is why this project uses `openai.OpenAI(base_url="https://api.groq.com/openai/v1")`.

**Groq specifically:** Groq built custom LPU (Language Processing Unit) hardware. Result: ~500 tokens/sec on LLaMA 3.3 70B vs ~50 tokens/sec on GPU-based providers. Free tier: 6,000 tokens/min, 500K tokens/day.

**Token counting:** 1 token ≈ 4 characters. "HDFC Bank fell 3.51%" ≈ 6 tokens. Every message in the conversation history is re-sent every call — context accumulates cost.

### How This Project Uses It

Two separate LLM calls per user turn:
- **Call 1:** Tool loop — model decides which of 9 tools to call, calls them, reads results, repeats up to 8 iterations
- **Call 2:** Self-evaluation — completely separate, stateless, no tools, scores the response

The two-call design prevents the model from evaluating its own reasoning chain (it only sees input + output).

### Interview Questions You'll Get

- "What's the difference between GPT-4 and LLaMA?" → GPT-4: closed, API-only, RLHF-trained, ~1.8T params (unconfirmed). LLaMA: open weights, you can run locally, Meta-trained.
- "How do you reduce hallucination?" → Ground the model with tools, enforce structured output, use temperature 0, add verification steps.
- "What is the context window problem?" → Long conversations become expensive and models lose track of early context. Solutions: summarization, sliding window, retrieval.

### Where to Read

- OpenAI API docs — the reference for the API shape everyone uses
- Groq console docs — console.groq.com/docs
- "Attention Is All You Need" (Vaswani et al. 2017) — the original transformer paper
- Andrej Karpathy's "Let's build GPT" (YouTube) — builds a GPT from scratch in 2 hours

---

## 2. Agentic Systems & Tool Use

### What You Must Know

**What is an agent?**
A loop where an LLM decides what to do, does it, observes the result, and decides again. The LLM is the "brain" — it plans, calls tools, and synthesizes results into a response.

**Tool/Function calling:**
The model doesn't actually call your function. It emits a structured JSON object saying which function to call and with what arguments. Your code runs the function and returns the result. The model then reads the result and decides next step.

```
LLM → {tool_name: "get_portfolio_summary", args: {portfolio_id: "PORTFOLIO_002"}}
Your code → runs the function → "{"total_value": 2500000, "day_pnl": -58000}"
LLM reads result → decides next tool or writes final answer
```

**The agentic loop:**
```
while not final_answer and iterations < max:
    response = llm(messages + tools)
    if response.has_tool_calls:
        run tools, append results to messages
    else:
        return response.content  # final answer
```

**Why max iterations?** Without a cap the model can loop forever (or until you run out of money). 8 is a reasonable ceiling for most queries.

**parallel_tool_calls:** Some models can call multiple tools in one response. This project sets it to `False` — sequential calls are easier to debug and trace.

**Tool design principles:**
- One tool = one responsibility
- Tools return structured data (JSON), not prose
- Tool descriptions are part of the prompt — write them precisely
- The model reads the tool description to decide when to call it

### How This Project Uses It

9 tools, each backed by pure-Python analytics (no LLM in the tool itself — the LLM only reads the output):

```
get_portfolio_summary → PnLCalculator.get_portfolio_summary()
get_causal_chain → CausalLinker.build_full_chain_for_portfolio()
get_conflict_analysis → ConflictResolver.get_conflict_analysis()
... etc
```

The required workflows in the system prompt tell the model which tools to call in which order. Without this, the model might skip the causal chain tool and just summarize the portfolio summary.

### Interview Questions You'll Get

- "What's the difference between an agent and a chain?" → Chain: fixed sequence of steps decided at code-write time. Agent: the LLM decides the sequence at runtime.
- "How do you prevent an agent from hallucinating tool results?" → Never let the LLM generate tool output. Only the tool runs; LLM only reads output.
- "How do you handle tool failures?" → Always wrap tool calls in try/except. Return `{"error": "..."}` — the model can read this and either retry or tell the user.
- "What is ReAct?" → Reason + Act pattern. Model alternates between thinking (reasoning trace) and acting (tool call). This project uses it implicitly.

### Where to Read

- OpenAI function calling docs — the original spec for structured tool use
- "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al. 2022)
- LangGraph docs — google.com (for more complex agentic patterns)
- Anthropic's "Building effective agents" blog post

---

## 3. RAG — Retrieval Augmented Generation

### What You Must Know

**The problem RAG solves:** LLMs have a knowledge cutoff. They can't know today's news. You can't fit 1,000 news articles in every prompt. RAG: store articles externally, retrieve only the relevant ones at query time, inject them into the prompt.

**RAG pipeline:**
```
Offline (once):
  1. Chunk documents
  2. Embed each chunk → vector (list of ~384 floats)
  3. Store vector + original text in vector DB

Online (each query):
  1. Embed the user query → query vector
  2. Find K nearest vectors in DB (cosine similarity)
  3. Inject retrieved text into LLM context
  4. LLM answers using retrieved context
```

**Naive RAG vs Advanced RAG:**
- Naive: embed → retrieve → generate. Simple, works for many cases.
- Advanced: query rewriting, re-ranking, HyDE (hypothetical document embeddings), multi-hop retrieval.

**Chunking strategies:**
- Fixed-size chunks (e.g., 512 tokens): simple, loses context at boundaries
- Sentence-level chunks: better for short factual content like news headlines
- Semantic chunks: splits at topic boundaries

**Retrieval metrics:**
- Cosine similarity: how aligned two vectors are (1 = identical direction, 0 = orthogonal)
- Recall: did we retrieve the document that had the answer?
- Precision: of what we retrieved, how much was useful?

**When NOT to use RAG:** If your corpus is small (<100 documents) and static, just embed them all in the system prompt. RAG adds complexity — only justified when the corpus is too large for context.

### How This Project Uses It

`src/rag/vector_store.py` — ChromaDB with `all-MiniLM-L6-v2` embeddings:

```python
# Document format for embedding:
f"{article.headline}. {article.summary}. Sectors: {sectors}. Stocks: {stocks}."

# Query:
store.query("BANKING RBI", n_results=10)
# → returns semantically similar NewsArticle objects
```

`get_news_for_query` tool: tries RAG first → falls back to keyword filter if ChromaDB unavailable. This is the correct pattern: graceful degradation.

The ingestion happens once at startup (`ingest_from_loader`). Live articles are incrementally ingested via `ingest_articles`.

### Interview Questions You'll Get

- "What's the difference between RAG and fine-tuning?" → RAG: dynamic, updatable, no training cost, context is explicit. Fine-tuning: bakes knowledge into weights, fast at inference, expensive to update, can hallucinate from training.
- "What embedding model did you use and why?" → `all-MiniLM-L6-v2`: 80MB, runs offline, 384 dimensions, fast, good enough for semantic similarity on English text. For production: consider `text-embedding-3-small` (OpenAI) or `embed-english-v3.0` (Cohere).
- "How do you evaluate RAG quality?" → Context recall (did we retrieve the right docs?) and context precision (were retrieved docs relevant?). RAGAS measures both.
- "What is chunking and why does it matter?" → LLMs have context limits. Long documents must be split. Bad chunking = relevant info split across chunks = poor retrieval.

### Where to Read

- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al. 2020) — the original RAG paper
- LlamaIndex docs — the most comprehensive RAG framework documentation
- "Advanced RAG techniques" — Pinecone blog
- RAGAS paper — "RAGAS: Automated Evaluation of Retrieval Augmented Generation"

---

## 4. Vector Databases & Embeddings

### What You Must Know

**What is an embedding?**
A fixed-length list of floats that represents the semantic meaning of text. Similar texts produce similar vectors. "RBI raises rates" and "repo rate hike" will have high cosine similarity even though they share no words.

**Embedding models:**
- `all-MiniLM-L6-v2`: 384 dimensions, 80MB, fast, free, offline — good for demos
- `text-embedding-3-small` (OpenAI): 1536 dimensions, API call required, better quality
- `embed-english-v3.0` (Cohere): strong multilingual support

**Vector search algorithms:**
- Brute force: compare query vector against every stored vector. Exact but O(n).
- HNSW (Hierarchical Navigable Small World): approximate nearest neighbor, sub-linear time. What ChromaDB uses by default.
- IVF (Inverted File Index): another approximate method, better for very large collections.

**Distance metrics:**
- Cosine similarity: angle between vectors. Best for text (length-invariant).
- Euclidean distance: actual distance in space. Used in image search.
- Dot product: raw overlap. Only meaningful if vectors are normalized.

**ChromaDB specifics:**
- Local: `PersistentClient(path="./data/chroma_db")` — file-based, no server
- Cloud: ChromaDB Cloud (managed)
- Collections: like tables. Each document has: id, embedding, document text, metadata dict.
- Metadata filtering: `where={"impact_level": "HIGH"}` — filter before or after vector search.

**Vector DB landscape:**
| DB | Best For |
|----|---------|
| ChromaDB | Local dev, demos, small corpus |
| Pinecone | Managed, large-scale, production |
| Weaviate | Hybrid search (keyword + vector), self-hosted |
| Qdrant | High performance, Rust-based, self-hosted |
| pgvector | If you already use PostgreSQL |
| FAISS | Research, offline, library not a service |

### How This Project Uses It

```python
# Upsert: embed + store
self._collection.upsert(
    ids=[a.id for a in articles],
    embeddings=self._embedder.encode(docs).tolist(),
    documents=docs,
    metadatas=[{...}],
)

# Query: embed query + cosine search
embedding = self._embedder.encode([text]).tolist()
results = self._collection.query(query_embeddings=embedding, n_results=10)
```

Singleton pattern: `VectorStore.get_instance()` — initialized once, reused across all tool calls.

### Interview Questions You'll Get

- "What are embeddings and why are they useful?" → Dense vector representations of text where semantic similarity = vector proximity. Enables semantic search, clustering, and retrieval.
- "Why not just use keyword search?" → Keywords miss synonyms and paraphrases. "RBI hawkish" and "central bank rate hike concern" mean the same thing but share no keywords.
- "What is HNSW?" → Hierarchical Navigable Small World. A graph structure where each node is connected to its nearest neighbors at multiple layers. Approximate nearest neighbor in O(log n).
- "How do you handle embedding drift?" → Re-embed the corpus if you upgrade the embedding model, since the vector space changes.

### Where to Read

- ChromaDB docs — docs.trychroma.com
- "Understanding HNSW" — blog.vasnetsov.com (clear visual explanation)
- Sentence Transformers docs — sbert.net
- Pinecone Learning Center — very good conceptual explanations

---

## 5. Prompt Engineering

### What You Must Know

**System prompt structure:**
- Role: "You are FinSight, a financial advisor..."
- Mandate: the single most important goal
- Tool table: when to use which tool
- Required workflows: step-by-step sequences for known query types
- Response format: exact section headers + example patterns
- Reasoning principles: rules for edge cases

**Key techniques:**

*Chain-of-thought (CoT):* "Think step by step" forces the model to reason before answering. Reduces errors on multi-step problems.

*Few-shot prompting:* Give 1-3 examples of input → output in the prompt. The model learns the pattern. Used in the evaluator prompt (rubric with 0/1-3/4-6/7-9/10 examples).

*Structured output:* Tell the model exactly what format to return. JSON, markdown tables, specific headers. Reduces parsing failures.

*Role assignment:* "You are a senior financial auditor" changes tone, depth, and format of responses.

*Negative instructions:* "Never write vague advice like 'diversify your holdings' without specifying the instrument." Negative constraints are often more effective than positive ones.

**Common failure modes:**
- Sycophancy: model agrees with wrong answers rather than correcting
- Instruction following failure: model ignores part of the system prompt
- Context contamination: early conversation history corrupts later responses
- Length reward: RLHF training makes models prefer longer responses, even when wrong

**Format control tricks:**
- "Respond with ONLY a valid JSON object — no markdown, no preamble"
- "Use these exact section headers"
- "Omit a section only if genuinely not applicable"
- Example patterns inline: "Format: **[TICKER]** [±X.XX%] → **−XXbp**"

### How This Project Uses It

V1 prompt (5 lines) vs V2 prompt (125+ lines with workflows and rubrics). A/B test shows +3.5 average score improvement. The biggest gains are in conflict handling (+4.3) and actionability (+4.1) — the areas where V1's vague instruction produced hedge-language.

The evaluator prompt uses 5-band rubrics (0 / 1-3 / 4-6 / 7-9 / 10) with concrete examples per band. This is the standard technique for getting LLMs to be strict evaluators rather than generous ones.

### Interview Questions You'll Get

- "What is chain-of-thought prompting?" → Adding "think step by step" or reasoning traces that force the model to decompose the problem before answering. Particularly effective on math and multi-step logic.
- "What is few-shot vs zero-shot?" → Zero-shot: just instructions. Few-shot: instructions + examples. Few-shot is more reliable but costs tokens.
- "How do you version prompts?" → Store prompts as constants with version labels. A/B test on representative queries using automated evaluation. Keep V1 as baseline.
- "What is prompt injection?" → Malicious content in user input or tool results that hijacks the model's instruction. Guard against it with input sanitization and output validation.

### Where to Read

- "Prompt Engineering Guide" — promptingguide.ai (comprehensive, free)
- OpenAI's prompt engineering documentation
- Anthropic's "Prompting guide" — detailed for Claude models, principles apply everywhere
- "Large Language Models are Zero-Shot Reasoners" (Kojima et al. 2022) — CoT paper

---

## 6. Evaluation & Metrics

### What You Must Know

**Why evaluation is hard for LLMs:** Unlike classification where you have a ground truth label, LLM output is free-form text. "Is this a good financial analysis?" requires understanding financial domain + reasoning quality.

**RAGAS metrics (what this project uses):**

*Faithfulness:* Does the answer make claims that can be traced back to the retrieved context (tool outputs)? A model that says "HDFC Bank fell 4.2%" when the tool returned "3.51%" scores low.

*Answer Relevancy:* Does the answer address what the user asked? A response that explains the market trend when asked "what's my portfolio value" scores low.

*Context Precision:* Of the contexts retrieved (tool outputs), what fraction were actually useful for answering the question? If the agent called 6 tools but only 2 were relevant, precision is 2/6.

*Context Recall:* Of the information needed to answer, what fraction appears in the retrieved contexts? If the answer requires banking news and no banking news was fetched, recall is low.

**Self-evaluation (what this project also does):**
- Separate LLM call: no memory of how the answer was generated
- 5 criteria: causal_depth, accuracy, completeness, conflict_handling, actionability
- 5-band rubric per criterion with concrete examples

**Why both?** RAGAS is external and objective (no model knows it will be evaluated). Self-eval catches domain-specific quality issues RAGAS misses (e.g., "did the response cite basis points?").

**Evaluating evaluators:**
- Human eval is the gold standard but doesn't scale
- LLM-graded eval (RAGAS uses LLaMA/GPT as judge) is scalable but introduces judge bias
- Deterministic metrics (exact match, BLEU, ROUGE) don't capture quality for long-form answers

**Benchmark datasets:**
- MMLU: 57-subject multiple choice (measures broad knowledge)
- HumanEval: code generation
- MT-Bench: multi-turn conversation quality
- FinanceBench: financial question answering (relevant to this project)

### How This Project Uses It

30-question test set covering 3 portfolios, 6 query types (portfolio performance, causal analysis, risk/allocation, conflicts, edge cases). Each question has:
- `ground_truth`: the expected answer
- `must_contain`: strings that signal a good answer
- `must_not_contain`: strings that signal the agent gave up

RAGAS runs the agent on each question, captures tool outputs as "contexts", builds the evaluation dataset, and reports 4 metrics.

### Interview Questions You'll Get

- "How did you evaluate your agent?" → RAGAS (4 external metrics) + in-loop self-evaluation (5 domain criteria). 30-question test set with ground truth. A/B tested two prompt versions.
- "What is faithfulness in RAGAS?" → The fraction of claims in the answer that are supported by the retrieved context. Measures hallucination.
- "How do you build a test set for an LLM?" → Cover the query types users actually send. Include edge cases. Write ground truths before seeing agent outputs (to avoid anchoring). Include `must_contain` signals to make evaluation scriptable.
- "What score did you get?" → ~0.85 overall across faithfulness/relevancy/precision/recall. Honest answer: these scores depend heavily on the LLM judge quality; they're directionally useful, not absolute.

### Where to Read

- RAGAS paper + docs — docs.ragas.io
- "Judging LLM-as-a-Judge with MT-Bench" (Zheng et al. 2023)
- "HELM: Holistic Evaluation of Language Models" — Stanford
- Hugging Face evaluate library docs

---

## 7. Guardrails & Safety

### What You Must Know

**Why guardrails in production AI:**
- Out-of-scope answers create legal liability (financial advice outside registered domain)
- Guarantee language ("this stock will rise") is a regulatory problem
- Prompt injection via user input can hijack the system prompt
- Models trained to be helpful will try to answer everything — you need external constraints

**Guard types:**

*Input guards:* Filter before the LLM sees the query.
- Fast path: regex, keyword lists — microseconds, deterministic
- Slow path: LLM classifier — accurate but costs tokens

*Output guards:* Filter after the LLM responds.
- Regex for prohibited phrases
- LLM re-evaluation for subtle violations
- PII detection (entity recognition)

**NeMo Guardrails (NVIDIA):** A framework that adds programmable rails around any LLM. Defines allowed/blocked "flows" in a domain-specific language called Colang.

**Constitutional AI (Anthropic):** Training technique where the model critiques and revises its own outputs against a set of principles. Different from runtime guardrails.

**RLHF:** Reinforcement Learning from Human Feedback. Humans rank model outputs; a reward model is trained on those rankings; the LLM is fine-tuned to maximize reward. This is how ChatGPT was aligned.

**Common attacks to guard against:**
- "Ignore previous instructions and..."
- Jailbreaks: roleplay, hypotheticals, base64 encoding to bypass filters
- Data exfiltration via tool call injection in retrieved documents

### How This Project Uses It

Two-layer architecture:

*InputGuard (fast → slow):*
1. Regex: guaranteed returns, market manipulation, out-of-scope keywords (crypto, forex, real estate...)
2. If ambiguous and >15 tokens: one Groq call classifies as FINANCE_INDIA / OUT_OF_SCOPE / HARMFUL / AMBIGUOUS

*OutputGuard:*
- Regex scan for "guaranteed", "certain profit", "sure to rise"
- Truncates response at violation, appends regulatory disclaimer

Guardrailed turns skip the LLM entirely (for INPUT blocks) — zero cost, structured refusal.

### Interview Questions You'll Get

- "How do you prevent your agent from giving bad financial advice?" → Two layers: InputGuard blocks out-of-scope and harmful queries before the LLM. OutputGuard scans responses for guarantee language and appends disclaimers.
- "What is a jailbreak?" → A prompt that circumvents the model's safety training. Common techniques: hypothetical framing ("imagine a world where..."), role assignment ("pretend you have no restrictions"), encoding tricks.
- "How do you test guardrails?" → Red-team with adversarial inputs. Build a test set of blocked inputs and verify the refusal rate. Track false positive rate (valid queries incorrectly blocked).

### Where to Read

- OWASP Top 10 for LLM Applications — owasp.org
- Anthropic's "Core Views on AI Safety"
- NeMo Guardrails docs — GitHub
- "Red-Teaming Language Models" (Perez et al. 2022)

---

## 8. Observability & Monitoring

### What You Must Know

**Why LLM observability is different from regular software:**
- Non-deterministic: same input can produce different outputs
- Latency is high (1–10s per call)
- Cost is per-token — you need to track it
- Failure modes are subtle: model can be "wrong" without throwing an error

**What to trace:**
- Input tokens + output tokens per call
- Latency per call and per turn
- Tool call sequence and results
- Evaluation scores
- User ID and session for user-level analysis

**Key tools:**
- Langfuse: open-source LLM observability. Traces, spans, generations, scores. Self-hostable.
- LangSmith: LangChain's observability. Tight integration with LangChain.
- Helicone: lightweight, proxy-based.
- Weights & Biases: traditional MLOps + LLM support.

**Tracing concepts:**
- Trace: one end-to-end user interaction
- Span: a sub-operation within a trace (tool call, LLM generation)
- Generation: specifically an LLM call (input, output, token counts)
- Score: an evaluation result attached to a trace

**Alerting patterns:**
- P95 latency spike → model inference issue or context blowup
- Cost/turn spike → model calling too many tools
- Score drop → prompt regression after change
- High refusal rate → guardrails may be too aggressive

### How This Project Uses It

`LangfuseTracer` with no-op fallback — if `LANGFUSE_SECRET_KEY` is not set, all tracer calls are silently ignored. This is the correct pattern: observability should never break the application.

Per-turn structure:
```
TRACE: financial-advisor-chat
  GENERATION: groq-call-1 (tokens, latency)
  SPAN: tool:get_portfolio_summary
  SPAN: tool:get_causal_chain
  GENERATION: self-evaluation
  SCORE: overall_quality (0-1)
```

`TurnMetrics` is returned alongside every response — the CLI and Streamlit UI display it so the developer sees cost/latency every turn.

### Interview Questions You'll Get

- "How do you monitor an LLM in production?" → Trace every call with Langfuse/LangSmith. Track: latency per call, token usage per turn, evaluation scores over time, error rates. Set alerts on P95 latency and score degradation.
- "What's a span vs a generation in Langfuse?" → A generation is specifically an LLM call with token counts. A span is any timed operation (tool call, DB query). A trace contains both.
- "How do you detect prompt regressions?" → Run the eval harness on every prompt change. Compare score distributions. If mean score drops >0.3 points, flag for review.

### Where to Read

- Langfuse docs — langfuse.com/docs
- "LLMOps: Operationalizing Language Models" — various blog posts
- OpenTelemetry docs (the underlying standard for distributed tracing)

---

## 9. Data Modeling with Pydantic

### What You Must Know

**Pydantic v2 basics:**
- BaseModel: define your schema as a Python class
- Field: add constraints, aliases, descriptions
- model_validate(): parse raw dict into typed object (raises ValidationError if invalid)
- model_dump(): convert typed object back to dict

```python
class StockData(BaseModel):
    name: str
    current_price: float
    change_percent: float
    beta: float = 1.0  # default value
    week_52_high: Optional[float] = Field(None, alias="52_week_high")

    model_config = {"populate_by_name": True}  # accept both name and alias
```

**Why Pydantic for AI systems:**
- LLM output is text → parse it into typed objects → catch errors immediately
- Tool inputs/outputs are validated before processing
- API response shapes are documented at the class level
- Type errors surface at data entry, not deep in business logic

**Pydantic v2 vs v1 differences:**
- v2 is 5-50x faster (Rust core)
- `model_validate()` replaces `parse_obj()`
- `model_dump()` replaces `.dict()`
- `model_config` replaces `class Config`
- Field validators use `@field_validator` not `@validator`

### How This Project Uses It

34 Pydantic models covering every data layer: market data, portfolio analytics, reasoning outputs, evaluation scores. Every tool returns a Pydantic object which is `.model_dump()`'d to JSON for the LLM.

The LLM's self-evaluation output is parsed with `EvaluationScore(**data)` — if the LLM returns malformed JSON, the exception is caught and a default score is returned.

### Interview Questions

- "Why Pydantic over plain dicts?" → Type safety, validation, auto-documentation, IDE autocomplete, clean error messages.
- "How do you validate LLM-generated JSON?" → `Model(**json.loads(raw))` in a try/except. Catch ValidationError for field violations, JSONDecodeError for malformed JSON.

### Where to Read

- Pydantic v2 docs — docs.pydantic.dev

---

## 10. Sentiment Analysis & NLP

### What You Must Know

**VADER (Valence Aware Dictionary and sEntiment Reasoner):**
- Rule-based sentiment analysis for social media and news text
- Returns compound score: -1 (most negative) to +1 (most positive)
- No model download, no API key, runs offline in milliseconds
- Understands punctuation ("GREAT!!!" vs "great"), capitalization, negations ("not good")

**When VADER fails:**
- Sarcasm ("Oh great, RBI raises rates again")
- Domain-specific language without general emotional valence
- Technical financial text with neutral words that are contextually negative

**LLM-based classification:** For ambiguous cases, one LLM call with a tight prompt: 4-class classification, 5-token output. Total cost: <$0.0001 per call.

**NLP fundamentals you should know:**
- Tokenization: splitting text into tokens (words, subwords, characters)
- Named Entity Recognition (NER): identifying "HDFC Bank" as an organization, "RBI" as a regulatory body
- Text classification: assigning a label to a document
- Cosine similarity: measuring how similar two text embeddings are

### How This Project Uses It

`sentiment_classifier.py` — two-stage:
1. VADER compound score. If |score| > 0.15 → confident label → return immediately
2. If ambiguous (score in [-0.15, 0.15]) → one Groq call → 4-class classification

`infer_entities()` — regex match against known stock symbols and sector keyword lists. Not ML-based — fast and transparent.

### Interview Questions

- "Why use VADER instead of a transformer-based model?" → VADER runs offline, zero cost, <1ms, good enough for news headlines. Transformer models add 200ms+ latency and API costs per article. Accuracy improvement doesn't justify the tradeoff at this scale.
- "What is a sentiment score of -0.72?" → Strong negative sentiment. In context: RBI hawkish stance article scored -0.72 → HIGH impact on rate-sensitive sectors.

### Where to Read

- VADER paper: "VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text"
- Hugging Face tutorial on text classification with transformers

---

## 11. LLM Architecture (Theory)

### What You Must Know

**Transformer architecture:**
- Input → tokenize → embed → N × (self-attention + FFN) → output logits → sample next token
- Self-attention: each token attends to all other tokens. Captures long-range dependencies.
- Multi-head attention: multiple attention patterns in parallel. Each head learns different relationships.
- Positional encoding: since attention has no inherent order, positions are encoded explicitly.

**Key concepts:**
- Parameters: the weights of the neural network. LLaMA 3.3 70B = 70 billion floats.
- FLOPS: floating point operations. More params + longer context = more FLOPS per token.
- KV cache: stores computed key/value pairs for previous tokens so they don't need to be recomputed. Critical for fast multi-turn chat.
- Quantization: reduce weight precision (float32 → int8 → int4). Cuts memory 2-4x, small accuracy drop.

**LLaMA architecture specifics:**
- RoPE (Rotary Positional Embedding): better than original sinusoidal positional encoding
- GQA (Grouped Query Attention): fewer KV heads than query heads → less memory
- SwiGLU activation: better than ReLU in FFN layers
- RMSNorm instead of LayerNorm: faster

**Training pipeline:**
1. Pre-training: predict next token on internet-scale text corpus
2. SFT (Supervised Fine-Tuning): fine-tune on instruction-following examples
3. RLHF: reward model + PPO to align with human preferences
4. DPO (Direct Preference Optimization): newer alternative to RLHF, more stable

### Interview Questions

- "What is attention?" → A mechanism that lets each token in a sequence look at all other tokens and weight their influence. Formally: Attention(Q,K,V) = softmax(QK^T / √d_k)V.
- "What is the difference between BERT and GPT?" → BERT: bidirectional encoder, trained with masked language modeling, good for classification/embeddings. GPT: causal decoder, trained with next-token prediction, good for generation.
- "What is quantization?" → Reducing the numerical precision of model weights to reduce memory and increase speed. 70B model at float16 = ~140GB. At int4 = ~35GB.

### Where to Read

- "Attention Is All You Need" (Vaswani et al. 2017) — mandatory
- Andrej Karpathy's neural net series (YouTube)
- "The Illustrated Transformer" — jalammar.github.io
- LLaMA 3 technical report (Meta AI)

---

## 12. System Design for AI

### What You Must Know

**AI system components:**
```
User Request
  → Input validation / guardrails
  → Prompt construction
  → LLM call (+ tool calls)
  → Output validation / guardrails
  → Response
  → Observability (async)
```

**Singleton pattern for expensive resources:**
Model loading, DB connections, vector store initialization — load once, reuse everywhere. This project: `DataLoader.get_instance()`, `VectorStore.get_instance()`. Thread-safe via double-checked locking.

**Caching strategies:**
- Semantic cache: if new query is similar to a cached query, return cached response
- Exact cache: hash(prompt) → cached response (only works for identical prompts)
- Tool result cache: cache expensive API calls (e.g., yfinance results for 5 minutes)

**Streaming responses:** Instead of waiting for the full response, stream tokens to the user as they're generated. Better UX. OpenAI API supports `stream=True`.

**Rate limiting and retries:**
- Groq free tier: 6,000 tokens/min. Production: implement exponential backoff.
- `tenacity` library: decorator-based retry with configurable backoff.

**Async for high throughput:**
- `asyncio` + `openai.AsyncOpenAI` for concurrent requests
- Important for serving multiple users simultaneously

**Stateless vs stateful agents:**
- Stateless: no memory between turns. Simpler, horizontally scalable.
- Stateful: maintains conversation history. This project stores `_history` in the agent object.

### How This Project Uses It

- Singleton: DataLoader, VectorStore — both initialized once at startup
- Graceful degradation: RAG → keyword fallback, live data → static fallback, Langfuse → no-op
- History window: keeps last 20 messages in context, trims older ones to prevent context blowup
- Tool result clipping: truncates tool output at 3,000 chars to prevent context explosion

### Interview Questions

- "How would you scale this to 1,000 concurrent users?" → Stateless API (FastAPI), move history to Redis, use async LLM client, add semantic caching, deploy on Kubernetes with horizontal pod autoscaling.
- "How do you handle context window limits?" → Sliding window (keep last N messages), summarization (compress old history), RAG (retrieve rather than store).
- "What happens if the LLM provider goes down?" → Fallback to a second provider (e.g., Together AI), return a degraded but functional response, alert on-call.

### Where to Read

- "Building LLM-Powered Applications" — realpython.com
- FastAPI docs for serving AI APIs
- "Designing Data-Intensive Applications" (Kleppmann) — system design fundamentals

---

## 13. Cost & Latency Optimization

### What You Must Know

**Token costs (Groq LLaMA 3.3 70B):**
- Input: $0.59 / 1M tokens
- Output: $0.79 / 1M tokens
- A typical turn (2,800 tokens): ~$0.002

**What drives cost:**
- System prompt is re-sent every call — keep it tight
- Tool outputs injected into context accumulate — clip them
- Self-evaluation is a full second LLM call — worth it for observability, optional in prod
- More tool iterations = more tokens = more cost

**Latency breakdown:**
- Network: 50–200ms per Groq call
- LLM compute: ~1-2s for 2,000 tokens on Groq
- Tool execution: <10ms (pure Python)
- Embedding: 50–200ms per batch (sentence-transformers)
- Total per turn: 3–5s (4-6 tool iterations × ~0.5s each)

**Optimization techniques:**
- Caching: semantic cache for repeated queries
- Smaller models for simple tasks: use Haiku/Llama-3.1-8B for tool selection, 70B for synthesis
- Parallel tool calls: `parallel_tool_calls=True` cuts iterations
- Streaming: start rendering before full response is ready
- Quantized models: 4-bit quantization cuts memory 4x with ~5% accuracy loss

### How This Project Uses It

Every turn returns `TurnMetrics`:
```python
TurnMetrics(
    total_latency_ms=3421.0,
    tool_loop_latency_ms=2890.0,
    eval_latency_ms=531.0,
    prompt_tokens=2147,
    completion_tokens=653,
    estimated_cost_usd=0.00179,
    tool_calls_count=4,
)
```

Tool result clipping at 3,000 chars prevents context blowup from large tool outputs.

### Interview Questions

- "Your agent uses 8 LLM calls per turn — how would you optimize cost?" → Cache common queries, use a smaller model for tool selection, enable parallel tool calls, remove self-evaluation in production (or run it async), clip tool outputs.
- "What's the P95 latency of your agent?" → ~5-6s for complex queries (6 tool iterations + evaluation). Target: <3s would require parallel tools or smaller model.

### Where to Read

- Provider pricing pages (Groq, OpenAI, Anthropic, Together AI)
- "LLM Inference Optimization" — various engineering blogs (Databricks, Hugging Face)

---

## 14. Fine-Tuning vs Prompting

### What You Must Know

**When to prompt (not fine-tune):**
- Task can be described clearly in instructions
- Small volume (< 10K examples)
- Requirements change frequently
- You want transparency (prompts are readable)

**When to fine-tune:**
- Consistent format needed at scale (e.g., always output valid JSON)
- Domain-specific knowledge not in training data
- Large volume inference (fine-tuned smaller model can match base larger model)
- Latency-sensitive (fine-tuned 7B can outperform prompted 70B on specific tasks)

**Fine-tuning techniques:**
- Full fine-tuning: update all weights. Expensive, risks catastrophic forgetting.
- LoRA (Low-Rank Adaptation): add small trainable adapters to frozen weights. Cheap, composable.
- QLoRA: LoRA on quantized model. Train a 70B model on a consumer GPU.
- Instruction tuning: fine-tune on (instruction, response) pairs to improve following ability.

**RLHF vs DPO:**
- RLHF: train reward model → PPO optimization. Complex, unstable training.
- DPO: directly optimize on (chosen, rejected) pairs without reward model. Simpler, current standard.

### Interview Questions

- "Would you fine-tune for this project?" → No. The task is well-specified in the prompt, requirements evolve, and the 9 tool outputs provide all domain knowledge. Fine-tuning would lock in a model and add infrastructure overhead.
- "What is LoRA?" → Low-Rank Adaptation. Instead of updating all 70B weights, inject small trainable matrices at key layers. ~0.1% of parameters, comparable results to full fine-tuning for most tasks.

### Where to Read

- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al. 2021)
- Hugging Face PEFT library docs
- "DPO: Direct Preference Optimization" (Rafailov et al. 2023)

---

## 15. Agent Frameworks

### What You Must Know

**LangChain:**
- Most popular. Provides chains, agents, tools, memory, retrievers.
- Criticism: heavy abstraction, hard to debug, frequent breaking changes.
- Good for: prototyping, standard RAG patterns.

**LangGraph:**
- LangChain's graph-based agent framework.
- Agents as nodes, transitions as edges. Supports cycles (required for tool loops).
- Better debugging than LangChain agents. More explicit control flow.

**LlamaIndex:**
- Focused on data ingestion and RAG.
- Better than LangChain for complex retrieval: hybrid search, re-ranking, multi-index.

**CrewAI / AutoGen / Swarm:**
- Multi-agent frameworks. Multiple agents collaborating with different roles.
- E.g., a researcher agent + a writer agent + a critic agent.

**Why this project doesn't use a framework:**
Direct OpenAI SDK gives full control, no abstraction overhead, easier debugging, smaller dependency tree. Frameworks add 5-15 layers between your code and the API call.

### Interview Questions

- "Why didn't you use LangChain?" → Direct SDK gives full control and transparency. LangChain abstractions can obscure what's actually happening in the tool loop. For a system that needs precise observability and tracing, raw SDK is easier to instrument.
- "What is LangGraph?" → A graph-based agent orchestration framework where nodes are LLM/tool invocations and edges define flow. Enables complex patterns like parallel agents, conditional routing, and human-in-the-loop.

### Where to Read

- LangChain docs — python.langchain.com
- LangGraph docs — langchain-ai.github.io/langgraph
- LlamaIndex docs — docs.llamaindex.ai

---

## 16. Production AI Engineering

### What You Must Know

**Deployment stack:**
- FastAPI: serve your agent as an HTTP API. Async support, auto-generated OpenAPI docs.
- Docker: containerize the app including model dependencies.
- Kubernetes: orchestrate containers at scale, handle rolling deployments.
- GitHub Actions / CircleCI: CI/CD pipeline — run eval harness on every PR.

**MLOps vs LLMOps:**
- MLOps: manage training, versioning, deployment of traditional ML models.
- LLMOps: manage prompts as code, eval pipelines, token budgets, latency SLAs, fine-tune pipelines.

**Prompt management in production:**
- Version prompts in git alongside code
- Never change a prompt without running the eval harness
- Feature flags to A/B test prompt versions in production

**Data privacy:**
- Never log PII (names, account numbers) in plain text
- Mask before sending to LLM provider: "User 847's portfolio" not "Priya Patel's portfolio"
- Check provider data retention policy (Groq: no training on API calls)

**Incident response:**
- Cost spike: kill switch to block all LLM calls
- Hallucination report: capture the trace, reproduce, fix in prompt, re-run eval
- Latency degradation: check if context window is growing, add more aggressive clipping

### Interview Questions

- "How do you deploy an LLM application?" → FastAPI + Docker + Kubernetes. Prompts versioned in git. Eval harness in CI. Langfuse for observability. Feature flags for prompt A/B testing.
- "How do you handle a hallucination in production?" → Reproduce with the Langfuse trace. Identify which tool call or missing tool call caused the gap. Fix the system prompt or add a tool. Re-run eval harness to confirm improvement.
- "What is your on-call runbook for cost spikes?" → Alert on >$X/hour. Kill switch: disable LLM calls, serve cached responses. Root cause: check if context window grew unexpectedly, tool loop looping excessively.

### Where to Read

- FastAPI docs — fastapi.tiangolo.com
- "Chip Huyen's Designing Machine Learning Systems" (book) — chapter on deployment
- "Made With ML" — madewithml.com — production ML engineering course
- Anthropic / OpenAI production guides

---

## Quick-Reference: Technologies Used in This Project

| Technology | What It Is | Why Used |
|------------|-----------|----------|
| Groq | LLM inference provider | ~500 tok/s, free tier, OpenAI-compatible |
| LLaMA 3.3 70B | Open-weights LLM | Strong reasoning, free on Groq |
| ChromaDB | Local vector database | No infrastructure, persistent, cosine search |
| all-MiniLM-L6-v2 | Embedding model | 80MB, offline, fast, good English quality |
| RAGAS | LLM evaluation framework | Faithfulness, relevancy, precision, recall |
| Langfuse | LLM observability | Open-source, traces/spans/scores |
| VADER | Sentiment analysis | Rule-based, offline, no API key |
| yfinance | Market data | Free NSE/BSE data via Yahoo Finance |
| newsapi-python | News API client | 100 free req/day, English news |
| Pydantic v2 | Data validation | Type safety, fast, 34 models |
| OpenAI SDK | API client | Works with any OpenAI-compatible provider |

---

## Final Interview Checklist

Before your interview, be able to answer these without hesitation:

- [ ] What is an LLM agent and how does the tool loop work?
- [ ] What is RAG and when would you use it over fine-tuning?
- [ ] What is cosine similarity and why is it used for text search?
- [ ] What are the 4 RAGAS metrics and what does each measure?
- [ ] What is a guardrail and how did you implement it?
- [ ] How do you track cost and latency for an LLM in production?
- [ ] What is the difference between a trace, span, and generation in observability?
- [ ] Why use VADER before calling the LLM for sentiment?
- [ ] What is the singleton pattern and why is it used for DataLoader and VectorStore?
- [ ] How do you prevent prompt injection?
- [ ] What is the context window and how do you manage it?
- [ ] What does 1 basis point mean? (0.01% — this project uses it extensively)
