from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(
    page_title="Financial Advisor Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

_PORTFOLIOS = {
    "PORTFOLIO_001": "Rahul Sharma — Diversified",
    "PORTFOLIO_002": "Priya Patel — Banking-Heavy (CRITICAL risk)",
    "PORTFOLIO_003": "Arun Krishnamurthy — Conservative",
}

_STARTER_QUESTIONS = [
    "Why is my portfolio down today? Give me the full causal breakdown.",
    "Which news events had the biggest impact on my holdings?",
    "Are there any conflicting signals I should know about?",
    "What is my sector exposure and where am I most concentrated?",
    "What's the overall market trend and how does it affect me?",
    "What are the key risks in my portfolio right now?",
]

_GEMINI_MODELS = {
    "gemini-2.0-flash-lite": "gemini-2.0-flash-lite — best free tier (30 RPM, 1500 RPD)",
    "gemini-1.5-flash-8b": "gemini-1.5-flash-8b — generous free tier",
    "gemini-1.5-flash-002": "gemini-1.5-flash-002 — stable versioned release",
    "gemini-2.0-flash": "gemini-2.0-flash — most capable (strict free limits)",
}


@st.cache_resource(show_spinner="Loading financial data...")
def get_agent(portfolio_id: str, api_key: str, provider: str, model: str):
    from src.agent.financial_advisor import FinancialAdvisorAgent  # noqa: PLC0415
    return FinancialAdvisorAgent(portfolio_id=portfolio_id, data_dir=DATA_DIR, api_key=api_key, provider=provider, model=model)


def _render_eval_scores(eval_score) -> None:
    with st.expander("Analysis Quality Scores", expanded=False):
        cols = st.columns(5)
        metrics = [
            ("Causal Depth", eval_score.causal_depth),
            ("Accuracy", eval_score.accuracy),
            ("Completeness", eval_score.completeness),
            ("Conflict Handling", eval_score.conflict_handling),
            ("Actionability", eval_score.actionability),
        ]
        for col, (label, value) in zip(cols, metrics):
            col.metric(label, f"{value:.1f}/10")
        overall_color = "green" if eval_score.overall >= 7 else ("orange" if eval_score.overall >= 5 else "red")
        st.markdown(
            f"**Overall: :{overall_color}[{eval_score.overall:.1f}/10]** — {eval_score.justification}"
        )


def _render_turn_metrics(turn_metrics) -> None:
    latency_s = turn_metrics.total_latency_ms / 1000
    total_tokens = turn_metrics.prompt_tokens + turn_metrics.completion_tokens
    st.caption(
        f"⏱ {latency_s:.1f}s  ·  "
        f"🔤 {total_tokens:,} tokens  ·  "
        f"💰 ${turn_metrics.estimated_cost_usd:.4f}  ·  "
        f"🔧 {turn_metrics.tool_calls_count} tool calls"
    )


if "custom_groq_key" not in st.session_state:
    st.session_state["custom_groq_key"] = ""
if "custom_gemini_key" not in st.session_state:
    st.session_state["custom_gemini_key"] = ""

with st.sidebar:
    st.title("📈 Financial Advisor")
    st.divider()

    with st.expander("LLM Provider & API Key", expanded=False):
        _provider = st.radio(
            "Provider",
            options=["groq", "gemini"],
            format_func=lambda x: "Groq (LLaMA 3.3 70B)" if x == "groq" else "Gemini",
            horizontal=True,
        )

        if _provider == "gemini":
            _model = st.selectbox(
                "Gemini Model",
                options=list(_GEMINI_MODELS.keys()),
                format_func=lambda m: _GEMINI_MODELS[m],
                index=0,
            )
        else:
            _model = "llama-3.3-70b-versatile"

        _env_key = os.environ.get("GROQ_API_KEY" if _provider == "groq" else "GEMINI_API_KEY", "")
        _saved_custom = (
            st.session_state["custom_groq_key"]
            if _provider == "groq"
            else st.session_state["custom_gemini_key"]
        )
        _api_key = _saved_custom or _env_key

        if _saved_custom:
            st.success("Your key is active")
        elif _env_key:
            st.success("Default key loaded")
        else:
            st.warning("No API key — add one below")

        _provider_label = "Groq" if _provider == "groq" else "Gemini"
        _placeholder = "gsk_..." if _provider == "groq" else "AIza..."
        _key_url = "console.groq.com → API Keys" if _provider == "groq" else "aistudio.google.com → Get API Key"

        st.markdown(
            f"**For better rate limits, use your own free key:**\n"
            f"* Go to **{_key_url}**\n"
            f"* Click **Create API Key**\n"
            f"* Paste it below and click **Apply**\n"
            f"* Your key is session-only and never saved"
        )
        with st.form(f"{_provider}_key_form", clear_on_submit=False):
            _input_key = st.text_input(
                f"{_provider_label} API Key",
                type="password",
                placeholder=_placeholder,
            )
            _col1, _col2 = st.columns(2)
            _apply = _col1.form_submit_button("Apply", use_container_width=True)
            _clear = _col2.form_submit_button("Clear", use_container_width=True)

        if _apply and _input_key.strip():
            if _provider == "groq":
                st.session_state["custom_groq_key"] = _input_key.strip()
            else:
                st.session_state["custom_gemini_key"] = _input_key.strip()
            st.rerun()
        if _clear:
            st.session_state["custom_groq_key"] = ""
            st.session_state["custom_gemini_key"] = ""
            st.rerun()

    st.divider()

    selected_pid = st.selectbox(
        "Select Portfolio",
        options=list(_PORTFOLIOS.keys()),
        format_func=lambda k: f"{k} — {_PORTFOLIOS[k].split('—')[1].strip()}",
    )
    st.caption(_PORTFOLIOS[selected_pid])
    st.divider()

    st.subheader("Starter Questions")
    for q in _STARTER_QUESTIONS:
        if st.button(q, use_container_width=True, key=q):
            st.session_state["pending_question"] = q

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["eval_scores"] = []
        st.session_state["turn_metrics"] = []
        st.rerun()

    st.caption("Langfuse observability: " + ("Enabled" if os.environ.get("LANGFUSE_SECRET_KEY") else "Disabled (no key)"))


if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "eval_scores" not in st.session_state:
    st.session_state["eval_scores"] = []
if "turn_metrics" not in st.session_state:
    st.session_state["turn_metrics"] = []
if "last_portfolio" not in st.session_state:
    st.session_state["last_portfolio"] = selected_pid

if st.session_state["last_portfolio"] != selected_pid:
    st.session_state["messages"] = []
    st.session_state["eval_scores"] = []
    st.session_state["turn_metrics"] = []
    st.session_state["last_portfolio"] = selected_pid

st.title("Autonomous Financial Advisor")
st.caption(f"Analysing: **{_PORTFOLIOS[selected_pid]}**")

for i, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            score_idx = i // 2
            if score_idx < len(st.session_state["eval_scores"]):
                _render_eval_scores(st.session_state["eval_scores"][score_idx])
            if score_idx < len(st.session_state["turn_metrics"]):
                _render_turn_metrics(st.session_state["turn_metrics"][score_idx])

if "pending_question" in st.session_state:
    prompt = st.session_state.pop("pending_question")
else:
    prompt = st.chat_input("Ask about your portfolio...")

if prompt:
    if not _api_key:
        st.error("Please add an API key in the 'LLM Provider & API Key' section above.")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analysing your portfolio..."):
            agent = get_agent(selected_pid, _api_key, _provider, _model)
            response_text, eval_score, turn_metrics = agent.chat(prompt)

        st.markdown(response_text)
        _render_eval_scores(eval_score)
        _render_turn_metrics(turn_metrics)

    st.session_state["messages"].append({"role": "assistant", "content": response_text})
    st.session_state["eval_scores"].append(eval_score)
    st.session_state["turn_metrics"].append(turn_metrics)
