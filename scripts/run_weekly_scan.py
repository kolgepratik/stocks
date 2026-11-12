#!/usr/bin/env python3
"""
run_weekly_scan.py – CLI entry point to trigger the full trading pipeline.

Usage
-----
    # From project root with uv:
    uv run python scripts/run_weekly_scan.py

    # With custom universe override:
    uv run python scripts/run_weekly_scan.py --symbols NVDA TSLA AMD

    # Dry run (scanner only, no DB save or notifications):
    uv run python scripts/run_weekly_scan.py --dry-run

    # Scheduled via cron (Sunday 6pm ET):
    # 0 18 * * 0  cd /path/to/trading_agents && uv run python scripts/run_weekly_scan.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Multi-Agent Options Trading Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        metavar="TICKER",
        help="Override scan universe (e.g. --symbols NVDA TSLA AMD)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Fixed maximum investment budget for the week in USD (e.g. --budget 500)",
    )
    parser.add_argument(
        "--max-trade-cost",
        type=float,
        default=None,
        help="Maximum dollar cost allowed per single position (defaults to max weekly budget e.g. 500)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run scanner + structurer only; skip DB save and Telegram notifications",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Suppress Telegram notifications but still save to DB",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="LangGraph thread ID for checkpointing (auto-generated if omitted)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity",
    )
    return parser.parse_args()


def _print_summary(state: dict) -> None:
    """Print a rich formatted summary of pipeline results."""
    console.print("\n[bold green]═══ Pipeline Complete ═══[/bold green]")

    # Print US Political & Macro Economic Context
    macro_ctx = state.get("macro_political_context") or {}
    pol_summary = macro_ctx.get("us_political_macro_summary")
    if pol_summary:
        console.print(f"\n[bold underline yellow]🏛️ US Political, Economic & Policy Intelligence Brief[/bold underline yellow]")
        console.print(f"[white]{pol_summary}[/white]")
        headlines = macro_ctx.get("macro_headlines", [])
        if headlines:
            console.print("\n[dim]Top National Headlines Ingested & Analyzed:[/dim]")
            for h in headlines[:4]:
                console.print(f"  [dim]• [{h.get('source', 'Media')}] {h.get('title')}[/dim]")
        
        pol_catalysts = macro_ctx.get("political_catalysts", [])
        if pol_catalysts:
            console.print("\n[bold yellow]⚡ Recent & Expected US Political & Policy Catalysts:[/bold yellow]")
            for pc in pol_catalysts[:4]:
                console.print(f"  [yellow]• {pc}[/yellow]")
        console.print("")

    # Print Sector Subagent Research Dossiers
    dossiers = macro_ctx.get("sector_dossiers") or []
    if dossiers:
        console.print(f"[bold underline cyan]🌐 Specialized Sector Subagent Research Dossiers (7-Day & Upcoming News)[/bold underline cyan]\n")
        for d in dossiers:
            chg = d.get("sector_7d_change_pct")
            chg_str = f"{chg:+.2f}%" if chg is not None else "N/A"
            console.print(f"[bold magenta]• {d.get('sector_name')} ({d.get('benchmark_etf')} 7d: {chg_str} | Range: {d.get('sector_7d_range', 'N/A')})[/bold magenta]")
            news = d.get("news_7d_headlines", [])
            if news:
                console.print(f"  [dim]Past 7-Day News: {news[0]}[/dim]")
            upc = d.get("upcoming_announcements", [])
            if upc:
                console.print(f"  [yellow]Upcoming Expected Catalyst: {upc[0]}[/yellow]")
            console.print("")

    recs = state.get("final_recommendations") or []
    if not recs:
        console.print("[yellow]No final recommendations generated.[/yellow]")
        return

    table = Table(title=f"[bold]Option Recommendations ({len(recs)} trades)[/bold]")
    table.add_column("Symbol", style="cyan bold")
    table.add_column("Structure", style="green")
    table.add_column("Strike")
    table.add_column("Expiry")
    table.add_column("Entry $")
    table.add_column("Est Profit $", style="bold green")
    table.add_column("Stop $")
    table.add_column("Total Cost $", style="yellow bold")
    table.add_column("Confidence", style="magenta")
    table.add_column("Risk")

    total_cost = 0.0
    for rec in recs:
        risk_color = {
            "LOW": "green", "MODERATE": "yellow", "HIGH": "red", "EXTREME": "red bold"
        }.get(rec.get("risk_rating", ""), "white")
        cost = rec.get("max_cost") or (rec.get("entry_price", 0) * 100 * (rec.get("max_contracts") or 1))
        total_cost += cost
        est_p = rec.get("estimated_profit_usd") or ((rec.get("target_exit_price", 0) - rec.get("entry_price", 0)) * 100)
        conf = f"{rec.get('confidence_rating', 'MED')} ({rec.get('confidence_score_pct', 70):.0f}%)"
        strike_str = (
            f"${rec.get('strike', 0):g}/${rec.get('short_strike'):g}"
            if rec.get("short_strike")
            else f"${rec.get('strike', 0):.2f}"
        )
        table.add_row(
            rec.get("symbol", "?"),
            rec.get("structure") or rec.get("option_type", "?"),
            strike_str,
            rec.get("expiration", "?"),
            f"${rec.get('entry_price', 0):.2f}",
            f"+${est_p:.2f} (+{rec.get('estimated_profit_pct', 50):.0f}%)",
            f"${rec.get('stop_loss_price', 0):.2f}",
            f"${cost:,.2f}",
            conf,
            f"[{risk_color}]{rec.get('risk_rating', 'N/A')}[/{risk_color}]",
        )

    console.print(table)
    budget = state.get("weekly_budget_usd", 500.0)
    console.print(
        f"\n[bold green]💡 Advisory Selection:[/bold green] [white]All approved recommendations below are sized independently within your ${budget:,.2f} budget cap. "
        "Trades are ranked by conviction and likelihood of profits so you can choose your preferred setup.[/white]\n"
    )

    # Print Full Actionable Advisory Breakdown
    console.print("[bold underline cyan]Detailed Advisory & Trade Execution Plans (Ranked by Profit Likelihood)[/bold underline cyan]\n")
    for i, rec in enumerate(recs, 1):
        sym = rec.get("symbol")
        structure = rec.get("structure", rec.get("option_type"))
        strike_str = (
            f"${rec.get('strike', 0):g}/${rec.get('short_strike'):g}"
            if rec.get("short_strike")
            else f"${rec.get('strike', 0):.2f}"
        )
        cat = rec.get("sector_category") or rec.get("sector_name") or "Market"
        rank_badge = f"[bold yellow]★ RANK #{i} (Top Profit Confidence)[/bold yellow]" if i == 1 else f"[bold cyan]Rank #{i}[/bold cyan]"
        console.print(f"[bold yellow]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold yellow]")
        console.print(f"{rank_badge} — [bold white]{sym} [{cat}] {structure} ({strike_str})[/bold white]")
        console.print(f"[bold yellow]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold yellow]")
        
        # Why Selected
        why = rec.get("why_selected") or rec.get("thesis", "N/A")
        news_cat = rec.get("news_catalyst", "Company news confirms technical setup")
        pol_imp = rec.get("political_macro_impact", "Neutral macro policy exposure")
        sec_name = rec.get("sector_name") or "Broad Market"
        sec_etf = rec.get("sector_etf") or "SPY"
        sec_7d = rec.get("sector_7d_change_pct")
        sec_range = rec.get("sector_7d_range", "N/A")
        rs_7d = rec.get("relative_strength_7d_pct")

        console.print(f"[bold]1. Why Selected & Catalyst Breakdown:[/bold]\n   {why}")
        if sec_7d is not None:
            console.print(f"   • [bold cyan]Sector 7-Day Market Data ({sec_name} - {sec_etf}):[/bold cyan] {sec_7d:+.2f}% [Range: {sec_range}] | Relative Strength: {f'{rs_7d:+.2f}%' if rs_7d is not None else 'N/A'}")
        
        pol_cat = rec.get("political_catalysts")
        earn_ctx = rec.get("earnings_context")
        comp_news = rec.get("company_news_context")
        if pol_cat:
            console.print(f"   • [bold yellow]🏛️ Political / Regulatory Catalysts:[/bold yellow] {pol_cat}")
        if earn_ctx:
            console.print(f"   • [bold magenta]📅 Earnings Intelligence:[/bold magenta] {earn_ctx}")
        if comp_news:
            console.print(f"   • [bold green]📢 Earnings Call / Corporate Developments:[/bold green] {comp_news}")
        console.print(f"   • [dim]Company News Catalyst:[/dim] {news_cat}")
        console.print(f"   • [dim]US Political & Policy Sensitivity:[/dim] {pol_imp}\n")
        
        # Estimated Profit & Loss
        entry = rec.get("entry_price", 0)
        target = rec.get("target_exit_price", 0)
        stop = rec.get("stop_loss_price", 0)
        est_p = rec.get("estimated_profit_usd") or ((target - entry) * 100)
        est_pct = rec.get("estimated_profit_pct", 50.0)
        contracts = rec.get("max_contracts", 1)
        cost = rec.get("max_cost", entry * 100 * contracts)
        console.print(
            f"[bold]2. Estimated Profit & Risk Profile:[/bold]\n"
            f"   • Entry Price:      ${entry:.2f}/sh (${round(entry*100, 2):.2f}/contract)\n"
            f"   • Total Investment: {contracts} contract{'s' if contracts != 1 else ''} = ${cost:,.2f}\n"
            f"   • Target Exit:      ${target:.2f} -> [bold green]+${est_p:.2f} profit (+{est_pct:.0f}% gain)[/bold green]\n"
            f"   • Stop Loss:        ${stop:.2f} -> [bold red]-25% loss limit[/bold red]\n"
            f"   • Delta: {rec.get('delta', 0):.2f} | IV: {rec.get('implied_volatility', 0):.1f}%\n"
        )
        
        # When to Buy
        when_buy = rec.get("when_to_buy", "Enter near market open using limit order at or below mid-price.")
        console.print(f"[bold]3. When to Buy (Entry Trigger):[/bold]\n   {when_buy}\n")
        
        # When to Sell
        when_sell = rec.get("sell_conditions", "Take profit at target or cut at stop loss.")
        console.print(f"[bold]4. When / Under What Conditions to Sell (Exit Rules):[/bold]\n   {when_sell}\n")
        
        # Early Warning Signs
        warnings = rec.get("early_warning_signs", [])
        if isinstance(warnings, list) and warnings:
            warn_str = "\n   • " + "\n   • ".join(warnings)
        else:
            warn_str = f"   • {warnings or 'Monitor volume and moving average support'}"
        console.print(f"[bold]5. Potential Early Warning Signs of Increasing Risk:[/bold]{warn_str}\n")
        
        # Confidence
        conf_rating = rec.get("confidence_rating", "MEDIUM")
        conf_pct = rec.get("confidence_score_pct", 70.0)
        console.print(
            f"[bold]6. Confidence & Conviction in Achieving Estimated Profits:[/bold]\n"
            f"   • Rating: [bold magenta]{conf_rating}[/bold magenta] ({conf_pct:.0f}% probability estimate)\n"
            f"   • Risk Sentinel Review: {rec.get('risk_rating', 'LOW')} risk\n"
        )


def main() -> None:
    args = _parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level=args.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )
    logger.add("logs/scan_{time:YYYY-MM-DD}.log", level="DEBUG", rotation="1 day", retention="7 days")

    from pathlib import Path
    Path("logs").mkdir(exist_ok=True)

    console.print("[bold blue]🚀 Multi-Agent Options Trading System[/bold blue]")
    console.print(f"Mode: {'[yellow]DRY RUN[/yellow]' if args.dry_run else '[green]LIVE[/green]'}")
    if args.budget:
        console.print(f"User Budget Override: [bold yellow]${args.budget:,.2f}[/bold yellow]")

    if args.dry_run:
        # Patch persistence_node to skip DB + notifications
        import src.workflow.graph as graph_module
        def _noop_persistence(state):
            logger.info("[DRY RUN] Skipping DB save and notifications")
            return state
        graph_module.persistence_node = _noop_persistence

    if args.no_notify:
        import src.notifications.telegram_bot as tg_module
        async def _noop_notify(*a, **kw):
            logger.info("[--no-notify] Telegram notifications suppressed")
        tg_module.send_trade_notifications = _noop_notify

    from src.workflow.graph import run_pipeline
    final_state = run_pipeline(
        symbols=args.symbols,
        thread_id=args.thread_id,
        weekly_budget_usd=args.budget,
        max_cost_per_trade_usd=args.max_trade_cost,
    )

    _print_summary(final_state)

    if final_state.get("workflow_error"):
        console.print(f"[red]⚠️  Workflow error: {final_state['workflow_error']}[/red]")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
