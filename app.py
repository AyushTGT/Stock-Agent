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


@st.cache_resource(show_spinner="Loading financial data...")
def get_agent(portfolio_id: str, api_key: str):
    from src.agent.financial_advisor import FinancialAdvisorAgent  # noqa: PLC0415
    return FinancialAdvisorAgent(portfolio_id=portfolio_id, data_dir=DATA_DIR, api_key=api_key)


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


with st.sidebar:
    st.title("📈 Financial Advisor")
    # st.caption("Powered by LLaMA 3.3 70B via Groq")
    st.divider()

    entered_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Your key is used only for this session and is never saved.",
    )
    _groq_api_key = entered_key.strip() or os.environ.get("GROQ_API_KEY", "")
    if not _groq_api_key:
        st.warning("Enter a Groq API key to start.")
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
        st.rerun()

    # st.caption("Langfuse observability: " + ("Enabled" if os.environ.get("LANGFUSE_SECRET_KEY") else "Disabled (no key)"))


if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "eval_scores" not in st.session_state:
    st.session_state["eval_scores"] = []
if "last_portfolio" not in st.session_state:
    st.session_state["last_portfolio"] = selected_pid

if st.session_state["last_portfolio"] != selected_pid:
    st.session_state["messages"] = []
    st.session_state["eval_scores"] = []
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

if "pending_question" in st.session_state:
    prompt = st.session_state.pop("pending_question")
else:
    prompt = st.chat_input("Ask about your portfolio...")

if prompt:
    if not _groq_api_key:
        st.error("Please enter your Groq API key in the sidebar to continue.")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analysing your portfolio..."):
            agent = get_agent(selected_pid, _groq_api_key)
            response_text, eval_score = agent.chat(prompt)

        st.markdown(response_text)
        _render_eval_scores(eval_score)

    st.session_state["messages"].append({"role": "assistant", "content": response_text})
    st.session_state["eval_scores"].append(eval_score)
