from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
console = Console()

_PORTFOLIO_CHOICES = {
    "1": ("PORTFOLIO_001", "Rahul Sharma — Diversified (8 stocks + 4 MFs)"),
    "2": ("PORTFOLIO_002", "Priya Patel — Banking-heavy CRITICAL concentration"),
    "3": ("PORTFOLIO_003", "Arun Krishnamurthy — Conservative (34% debt MFs)"),
}


def pick_api_key(cli_key: str | None, provider: str) -> str:
    env_var = "GEMINI_API_KEY" if provider == "gemini" else "GROQ_API_KEY"
    key = cli_key or os.environ.get(env_var, "")
    if key:
        return key
    console.print(Panel(
        f"[yellow]No {env_var} found in environment.[/yellow]\n"
        "Your key will be used only for this session and will [bold]not[/bold] be saved.",
        title="API Key Required",
        expand=False,
    ))
    label = "Enter your Gemini API key" if provider == "gemini" else "Enter your Groq API key"
    key = Prompt.ask(label, password=True)
    if not key.strip():
        console.print("[red]API key is required. Exiting.[/red]")
        sys.exit(1)
    return key.strip()


def pick_portfolio(cli_id: str | None) -> str:
    if cli_id:
        valid_ids = {v[0] for v in _PORTFOLIO_CHOICES.values()}
        if cli_id in valid_ids:
            return cli_id
        console.print(f"[red]Unknown portfolio ID '{cli_id}'. Valid: {', '.join(sorted(valid_ids))}[/red]")
        sys.exit(1)

    console.print(Panel("[bold cyan]Autonomous Financial Advisor[/bold cyan]\nPowered by LLaMA 3.3 70B (Groq) or Gemini 2.0 Flash", expand=False))
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", width=3)
    table.add_column("Portfolio ID", width=16)
    table.add_column("Description")
    for key, (pid, desc) in _PORTFOLIO_CHOICES.items():
        table.add_row(key, pid, desc)
    console.print(table)

    choice = Prompt.ask("Select portfolio", choices=list(_PORTFOLIO_CHOICES.keys()), default="1")
    return _PORTFOLIO_CHOICES[choice][0]


def print_eval_scores(eval_score) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Metric", style="dim")
    table.add_column("Score", style="bold")
    table.add_row("Causal Depth", f"{eval_score.causal_depth:.1f}/10")
    table.add_row("Accuracy", f"{eval_score.accuracy:.1f}/10")
    table.add_row("Completeness", f"{eval_score.completeness:.1f}/10")
    table.add_row("Conflict Handling", f"{eval_score.conflict_handling:.1f}/10")
    table.add_row("Actionability", f"{eval_score.actionability:.1f}/10")
    overall_style = "green" if eval_score.overall >= 7 else ("yellow" if eval_score.overall >= 5 else "red")
    console.print(
        Panel(
            table,
            title=f"[{overall_style}]Analysis Quality: {eval_score.overall:.1f}/10[/{overall_style}]",
            subtitle=f"[dim]{eval_score.justification}[/dim]",
            expand=False,
        )
    )


def run_chat(portfolio_id: str, api_key: str, provider: str, model: str | None, single_query: str | None = None) -> None:
    from src.agent.financial_advisor import FinancialAdvisorAgent

    model_label = model or ("gemini-2.0-flash-lite" if provider == "gemini" else "llama-3.3-70b-versatile")
    console.print(f"\n[bold green]Loading agent for {portfolio_id} ({provider} / {model_label})...[/bold green]")
    agent = FinancialAdvisorAgent(portfolio_id=portfolio_id, data_dir=DATA_DIR, api_key=api_key, provider=provider, model=model)
    console.print("[green]Ready. Type your question (or 'exit' to quit).[/green]\n")

    queries = [single_query] if single_query else None

    while True:
        if queries is not None:
            if not queries:
                break
            user_input = queries.pop(0)
            console.print(f"[bold blue]You:[/bold blue] {user_input}")
        else:
            try:
                user_input = Prompt.ask("[bold blue]You[/bold blue]")
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye.[/dim]")
                break

        if user_input.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        if not user_input.strip():
            continue

        with console.status("[bold yellow]Thinking...[/bold yellow]", spinner="dots"):
            response_text, eval_score = agent.chat(user_input)

        console.print("\n[bold green]Advisor:[/bold green]")
        console.print(Markdown(response_text))
        console.print()
        print_eval_scores(eval_score)
        console.print()

        if queries is not None and not queries:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Financial Advisor CLI")
    parser.add_argument("--portfolio", "-p", help="Portfolio ID (PORTFOLIO_001 / 002 / 003)")
    parser.add_argument("--query", "-q", help="Single query — runs once and exits")
    parser.add_argument("--api-key", "-k", help="API key (session only, not saved)")
    parser.add_argument(
        "--provider",
        choices=["groq", "gemini"],
        default=os.environ.get("LLM_PROVIDER", "groq"),
        help="LLM provider: groq (default) or gemini",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model override (e.g. gemini-1.5-flash-8b). Defaults to provider's default.",
    )
    args = parser.parse_args()

    api_key = pick_api_key(args.api_key, args.provider)
    portfolio_id = pick_portfolio(args.portfolio)
    run_chat(portfolio_id, api_key=api_key, provider=args.provider, model=args.model, single_query=args.query)


if __name__ == "__main__":
    main()
