"""
build_static.py
===============
Build a single-file offline dashboard from generated score, backtest, and
ML benchmark CSVs.

Usage:
    python -m src.dashboard.build_static
    python -m src.dashboard.build_static --symbols AAPL,BTC
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import pandas as pd

from src.data.utils import PROJECT_ROOT, ensure_dir, get_logger, load_asset_config

logger = get_logger("dashboard.build_static")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SCORES_DIR = PROCESSED_DIR / "scores"
BACKTEST_DIR = PROCESSED_DIR / "backtests"
EQUITY_DIR = BACKTEST_DIR / "equity_curves"
ML_DIR = PROCESSED_DIR / "ml"
DASHBOARD_DIR = PROCESSED_DIR / "dashboard"
DEFAULT_OUTPUT = DASHBOARD_DIR / "index.html"


def fmt_pct(value: object, digits: int = 1) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.{digits}f}%"


def fmt_num(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def fmt_money(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    return f"${float(value):,.0f}"


def css_class_for_score(score: object, inverse: bool = False) -> str:
    if pd.isna(score):
        return "muted"
    value = float(score)
    if inverse:
        if value <= 35:
            return "good"
        if value <= 65:
            return "watch"
        return "bad"
    if value >= 65:
        return "good"
    if value >= 45:
        return "watch"
    return "bad"


def load_symbols(symbol_override: str | None) -> list[str]:
    cfg = load_asset_config()
    symbols = [s["ticker"] for s in cfg.get("stocks", [])] + [c["symbol"] for c in cfg.get("crypto", [])]
    if symbol_override:
        wanted = {s.strip().upper() for s in symbol_override.split(",") if s.strip()}
        symbols = [s for s in symbols if s in wanted]
    return symbols


def load_latest_scores(symbols: list[str]) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        path = SCORES_DIR / f"{symbol}.csv"
        if not path.exists():
            logger.warning("Missing score file for %s; run src.scoring.run_scores first.", symbol)
            continue
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        if df.empty:
            continue
        latest = df.iloc[-1].copy()
        latest["price"] = latest.get("adj_close", latest.get("price_usd", latest.get("close")))
        rows.append(latest)
    return pd.DataFrame(rows)


def load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning("Optional dashboard input missing: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path)


def sparkline_svg(path: Path, width: int = 300, height: int = 72) -> str:
    if not path.exists():
        return '<span class="muted">No equity curve</span>'
    df = pd.read_csv(path)
    if df.empty or "equity" not in df.columns:
        return '<span class="muted">No equity curve</span>'

    values = pd.to_numeric(df["equity"], errors="coerce").dropna()
    if values.empty:
        return '<span class="muted">No equity curve</span>'
    if len(values) > 120:
        values = values.iloc[:: max(1, len(values) // 120)]

    lo = float(values.min())
    hi = float(values.max())
    span = hi - lo or 1.0
    step = width / max(1, len(values) - 1)
    points = []
    for idx, value in enumerate(values):
        x = idx * step
        y = height - ((float(value) - lo) / span * (height - 8)) - 4
        points.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" role="img" aria-label="Equity curve">'
        f'<polyline points="{" ".join(points)}"></polyline></svg>'
    )


def render_score_cards(latest_scores: pd.DataFrame, backtests: pd.DataFrame, ml_summary: pd.DataFrame) -> str:
    cards = []
    for _, row in latest_scores.sort_values("symbol").iterrows():
        symbol = str(row["symbol"])
        asset_backtests = backtests[backtests["symbol"] == symbol] if not backtests.empty else pd.DataFrame()
        best = None
        if not asset_backtests.empty:
            best = asset_backtests.sort_values("sharpe", ascending=False).iloc[0]
        ml_row = None
        if not ml_summary.empty:
            matches = ml_summary[ml_summary["symbol"] == symbol]
            if not matches.empty:
                ml_row = matches.iloc[0]

        timing_class = css_class_for_score(row.get("timing_score"))
        value_class = css_class_for_score(row.get("value_score"))
        risk_class = css_class_for_score(row.get("risk_score"), inverse=True)
        final_class = css_class_for_score(row.get("final_score"))
        cards.append(
            f"""
            <section class="asset">
              <div class="asset-head">
                <div>
                  <h2>{html.escape(symbol)}</h2>
                  <p>{pd.to_datetime(row["date"]).date().isoformat()} · {fmt_money(row.get("price"))}</p>
                </div>
                <strong class="score {final_class}">{fmt_num(row.get("final_score"), 0)}</strong>
              </div>
              <div class="score-grid">
                <div><span>Timing</span><strong class="{timing_class}">{fmt_num(row.get("timing_score"), 0)}</strong><em>{html.escape(str(row.get("timing_label", "n/a")))}</em></div>
                <div><span>Value</span><strong class="{value_class}">{fmt_num(row.get("value_score"), 0)}</strong><em>{html.escape(str(row.get("value_label", "n/a")))}</em></div>
                <div><span>Risk</span><strong class="{risk_class}">{fmt_num(row.get("risk_score"), 0)}</strong><em>{html.escape(str(row.get("risk_label", "n/a")))}</em></div>
              </div>
              <div class="mini-table">
                <div><span>Suggested size</span><b>{fmt_pct(row.get("suggested_pct_of_account"))}</b></div>
                <div><span>Stop distance</span><b>{fmt_pct(row.get("stop_distance_pct"))}</b></div>
                <div><span>Best backtest</span><b>{html.escape(str(best["strategy"])) if best is not None else "n/a"}</b></div>
                <div><span>Best Sharpe</span><b>{fmt_num(best["sharpe"]) if best is not None else "n/a"}</b></div>
                <div><span>ML AUC</span><b>{fmt_num(ml_row["roc_auc"]) if ml_row is not None else "n/a"}</b></div>
                <div><span>ML return</span><b>{fmt_pct(ml_row["strategy_total_return"]) if ml_row is not None else "n/a"}</b></div>
              </div>
            </section>
            """
        )
    return "\n".join(cards)


def render_backtest_table(backtests: pd.DataFrame) -> str:
    if backtests.empty:
        return '<p class="muted">No backtest summary found.</p>'

    rows = []
    for _, row in backtests.sort_values(["symbol", "sharpe"], ascending=[True, False]).iterrows():
        symbol = str(row["symbol"])
        strategy = str(row["strategy"])
        curve_path = EQUITY_DIR / f"{symbol}_{strategy}.csv"
        rows.append(
            f"""
            <tr>
              <td>{html.escape(symbol)}</td>
              <td>{html.escape(strategy)}</td>
              <td>{fmt_pct(row.get("total_return"))}</td>
              <td>{fmt_pct(row.get("cagr"))}</td>
              <td>{fmt_num(row.get("sharpe"))}</td>
              <td>{fmt_pct(row.get("max_drawdown"))}</td>
              <td>{fmt_pct(row.get("average_exposure"))}</td>
              <td>{sparkline_svg(curve_path)}</td>
            </tr>
            """
        )
    return f"""
    <table>
      <thead>
        <tr>
          <th>Asset</th><th>Strategy</th><th>Total Return</th><th>CAGR</th>
          <th>Sharpe</th><th>Max DD</th><th>Exposure</th><th>Equity</th>
        </tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    """


def render_ml_table(ml_summary: pd.DataFrame) -> str:
    if ml_summary.empty:
        return '<p class="muted">No ML summary found.</p>'
    rows = []
    for _, row in ml_summary.sort_values("roc_auc", ascending=False).iterrows():
        rows.append(
            f"""
            <tr>
              <td>{html.escape(str(row["symbol"]))}</td>
              <td>{fmt_num(row.get("samples"), 0)}</td>
              <td>{fmt_pct(row.get("accuracy"))}</td>
              <td>{fmt_num(row.get("roc_auc"))}</td>
              <td>{fmt_num(row.get("brier_score"), 3)}</td>
              <td>{fmt_pct(row.get("strategy_total_return"))}</td>
              <td>{fmt_num(row.get("strategy_sharpe"))}</td>
              <td>{fmt_pct(row.get("strategy_max_drawdown"))}</td>
            </tr>
            """
        )
    return f"""
    <table>
      <thead>
        <tr>
          <th>Asset</th><th>Samples</th><th>Accuracy</th><th>AUC</th>
          <th>Brier</th><th>Strategy Return</th><th>Sharpe</th><th>Max DD</th>
        </tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    """


def render_html(latest_scores: pd.DataFrame, backtests: pd.DataFrame, ml_summary: pd.DataFrame) -> str:
    generated_at = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M %Z")
    if latest_scores.empty:
        raise ValueError("No score files found; run python -m src.scoring.run_scores first.")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>crypto-stock-signal-lab dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #18212f;
      --muted: #657083;
      --line: #d9dee8;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --good: #107f5d;
      --watch: #9a6700;
      --bad: #bd3636;
      --accent: #3457d5;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }}
    header {{ padding: 28px 32px 18px; border-bottom: 1px solid var(--line); background: var(--panel); }}
    main {{ padding: 24px 32px 40px; max-width: 1480px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0; font-size: 21px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }}
    p {{ margin: 0; color: var(--muted); }}
    .asset-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; margin-bottom: 28px; }}
    .asset {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }}
    .asset-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 16px; }}
    .score {{ min-width: 58px; min-height: 44px; display: inline-flex; align-items: center; justify-content: center; border-radius: 6px; color: #fff; font-size: 24px; }}
    .score-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-bottom: 14px; }}
    .score-grid div {{ border: 1px solid var(--line); border-radius: 6px; padding: 10px; min-height: 90px; }}
    span {{ color: var(--muted); font-size: 12px; display: block; }}
    strong {{ font-size: 24px; display: block; margin-top: 4px; }}
    em {{ color: var(--muted); font-style: normal; font-size: 12px; display: block; margin-top: 4px; }}
    .mini-table {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); border-top: 1px solid var(--line); }}
    .mini-table div {{ padding: 10px 0; border-bottom: 1px solid var(--line); }}
    .mini-table b {{ font-size: 14px; }}
    .good {{ color: var(--good); }} .watch {{ color: var(--watch); }} .bad {{ color: var(--bad); }} .muted {{ color: var(--muted); }}
    .score.good {{ background: var(--good); color: #fff; }} .score.watch {{ background: var(--watch); color: #fff; }} .score.bad {{ background: var(--bad); color: #fff; }}
    section.block {{ margin-top: 22px; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; min-width: 860px; }}
    th {{ text-align: left; color: var(--muted); font-weight: 600; border-bottom: 1px solid var(--line); padding: 9px 8px; }}
    td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; vertical-align: middle; }}
    .spark {{ width: 220px; height: 54px; display: block; }}
    .spark polyline {{ fill: none; stroke: var(--accent); stroke-width: 2.4; stroke-linecap: round; stroke-linejoin: round; }}
    @media (max-width: 760px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .asset-grid {{ grid-template-columns: 1fr; }}
      .score-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>crypto-stock-signal-lab dashboard</h1>
    <p>Generated {html.escape(generated_at)} · educational research only, not financial advice</p>
  </header>
  <main>
    <div class="asset-grid">
      {render_score_cards(latest_scores, backtests, ml_summary)}
    </div>
    <section class="block">
      <h3>Backtest Comparison</h3>
      {render_backtest_table(backtests)}
    </section>
    <section class="block">
      <h3>Walk-Forward ML Benchmark</h3>
      {render_ml_table(ml_summary)}
    </section>
  </main>
</body>
</html>
"""


def build_dashboard(symbols: list[str], output_path: Path) -> Path:
    latest_scores = load_latest_scores(symbols)
    backtests = load_optional_csv(BACKTEST_DIR / "summary.csv")
    ml_summary = load_optional_csv(ML_DIR / "walkforward_summary.csv")
    html_text = render_html(latest_scores, backtests, ml_summary)
    ensure_dir(output_path.parent)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline HTML dashboard from processed CSV outputs.")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbol override, e.g. AAPL,BTC.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output HTML path.")
    args = parser.parse_args()

    symbols = load_symbols(args.symbols)
    if not symbols:
        logger.error("No matching symbols found. Check --symbols or config/assets.yaml.")
        sys.exit(1)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    try:
        built = build_dashboard(symbols=symbols, output_path=output_path)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info("Saved dashboard -> %s", built)
    print(f"Dashboard written to {built}")


if __name__ == "__main__":
    main()

