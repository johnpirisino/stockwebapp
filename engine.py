# engine.py
import os
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

import matplotlib
matplotlib.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Frame, PageTemplate, Image
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch

# =========================================
# Environment / config
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FD_API_KEY = os.getenv("FINANCIAL_DATASETS_API_KEY")
FD_BASE_URL = "https://api.financialdatasets.ai"

DEBUG = os.getenv("DEBUG_ENGINE", "N").upper() == "Y"


def dbg(msg: str) -> None:
    if DEBUG:
        print(msg, flush=True)


# =========================================
# Helpers
# =========================================

def fmt_number(value, decimals=2):
    try:
        if value is None:
            return "N/A"
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return "N/A"


def fmt_int(value):
    try:
        if value is None:
            return "N/A"
        return f"{int(value):,}"
    except Exception:
        return "N/A"


def fd_headers() -> Dict[str, str]:
    if not FD_API_KEY:
        return {}
    # Header name is case-insensitive; this matches FD docs
    return {"X-API-Key": FD_API_KEY}


def fd_get_json(path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Optional[str]]:
    """
    Generic helper to call FinancialDatasets.ai and return (json, error_str).
    Never prints huge JSON; only status + short error snippet if DEBUG on.
    """
    if not FD_API_KEY:
        return None, "FINANCIAL_DATASETS_API_KEY missing."

    url = f"{FD_BASE_URL}{path}"
    try:
        r = requests.get(url, headers=fd_headers(), params=params, timeout=20)
        dbg(f"FD GET {path} status={r.status_code}")
        if r.status_code != 200:
            snippet = (r.text or "")[:200]
            dbg(f"FD ERROR {path}: {snippet}")
            return None, f"{r.status_code}: {snippet}"
        data = r.json()
        return data, None
    except Exception as e:
        dbg(f"FD EXCEPTION {path}: {e}")
        return None, str(e)


# =========================================
# FinancialDatasets.ai wrappers
# =========================================

def fetch_price_history_df(symbol: str, days: int = 365) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Use FD /prices to get daily OHLCV, return as DataFrame indexed by datetime.

    Response shape (per FD docs and your logs) is:
      { "ticker": "AAPL", "prices": [ { "time": "...", "close": ..., "volume": ... }, ... ] }
    """
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    params = {
        "ticker": symbol.upper(),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "interval": "day",
        "interval_multiplier": 1,
    }
    data, err = fd_get_json("/prices", params)
    if err or not data:
        return None, err or "Empty price response."

    prices = data.get("prices") if isinstance(data, dict) else None
    if not prices:
        return None, "No 'prices' field in price response."

    df = pd.DataFrame(prices)

    # Time column: "time" preferred, fall back to "date"
    if "time" in df.columns:
        df["time"] = df["time"].astype(str)
        # strip timezone words if present
        df["time"] = df["time"].str.replace(" EDT", "", regex=False).str.replace(" EST", "", regex=False)
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df = df.dropna(subset=["time"])
        df = df.set_index("time")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df = df.dropna(subset=["date"])
        df = df.set_index("date")
    else:
        return None, "Price response missing 'time'/'date' field."

    # Close & volume columns: handle multiple possible names
    if "close" not in df.columns:
        if "adj_close" in df.columns:
            df["close"] = df["adj_close"]
        elif "c" in df.columns:
            df["close"] = df["c"]
        else:
            return None, "Price response missing 'close' column."

    if "volume" not in df.columns:
        if "v" in df.columns:
            df["volume"] = df["v"]
        else:
            df["volume"] = 0.0

    df = df.sort_index()
    return df, None


def fetch_financial_metrics_snapshot(symbol: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    /financial-metrics/snapshot
    According to your schema, response is a single dict:

    {
      "ticker": "AAPL",
      "market_cap": ...,
      "enterprise_value": ...,
      ...
    }
    """
    params = {"ticker": symbol.upper()}
    data, err = fd_get_json("/financial-metrics/snapshot", params)
    if err or not data:
        return None, err or "No snapshot metrics."
    if not isinstance(data, dict):
        return None, "Unexpected snapshot format (expected dict)."
    return data, None


def fetch_financial_metrics_history(symbol: str, period: str = "annual", limit: int = 10) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    /financial-metrics
    Returns a list of metric objects (per your fm_history schema).
    """
    params = {"ticker": symbol.upper(), "period": period, "limit": limit}
    data, err = fd_get_json("/financial-metrics", params)
    if err or not data:
        return None, err or "No metrics history."

    # Response might be list or dict with a list inside; normalize to list.
    if isinstance(data, list):
        return data, None
    if isinstance(data, dict):
        # Prefer key names that FD uses; fall back to treating dict as single item.
        for key in ("metrics", "financial_metrics", "results"):
            if isinstance(data.get(key), list):
                return data[key], None
        return [data], None
    return None, "Unexpected financial-metrics format."


def fetch_fd_analyst_estimates(symbol: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    /analyst-estimates
    Expecting a dict with 'analyst_estimates': [ ... ].
    """
    params = {"ticker": symbol.upper(), "period": "annual"}
    data, err = fd_get_json("/analyst-estimates", params)
    if err or not data:
        return None, err or "No analyst estimates."

    if isinstance(data, dict):
        est = data.get("analyst_estimates")
        if isinstance(est, list):
            return est, None
    return None, "Unexpected analyst-estimates format."


def fetch_company_facts(symbol: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    /company/facts
    Per your logs, this is:

    { "company_facts": { "ticker": "AAPL", "name": "Apple Inc", "sector": "...", "industry": "...", ... } }
    """
    params = {"ticker": symbol.upper()}
    data, err = fd_get_json("/company/facts", params)
    if err or not data:
        return None, err or "No company facts."

    if isinstance(data, dict):
        facts = data.get("company_facts")
        if isinstance(facts, dict):
            return facts, None
    return None, "Unexpected company-facts format."


def fetch_news(symbol: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    /news  → { "news": [ ... ] }
    """
    params = {"ticker": symbol.upper(), "limit": 5}
    data, err = fd_get_json("/news", params)
    if err or not data:
        return [], err or "No news."

    if isinstance(data, dict):
        items = data.get("news") or []
        if isinstance(items, list):
            return items, None
    return [], "Unexpected news format."


def fetch_insider_trades(symbol: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    /insider-transactions  → { "insider_transactions": [ ... ] }
    Your log shows rich insider data, so we parse that.
    """
    params = {"ticker": symbol.upper(), "limit": 20}
    data, err = fd_get_json("/insider-transactions", params)
    if err or not data:
        return [], err or "No insider transactions."

    if isinstance(data, dict):
        trades = data.get("insider_transactions") or data.get("insider_trades") or []
        if isinstance(trades, list):
            return trades, None
    return [], "Unexpected insider-transactions format."


def fetch_institutional(symbol: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    /institutional-ownership → { "institutional_ownership": [ ... ] }
    """
    params = {"ticker": symbol.upper(), "limit": 200}
    data, err = fd_get_json("/institutional-ownership", params)
    if err or not data:
        return [], err or "No institutional ownership."

    if isinstance(data, dict):
        holders = data.get("institutional_ownership") or []
        if isinstance(holders, list):
            return holders, None
    return [], "Unexpected institutional-ownership format."


def fetch_financials(symbol: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Use the income-statement view of /financials to build multi-year fundamentals.
    Your schema shows:

      {
        "income_statements": [
          {
            "ticker": "AAPL",
            "report_period": "2023-12-25",
            "period": "annual",
            "revenue": ...,
            "gross_profit": ...,
            "operating_income": ...,
            "net_income": ...,
            "earnings_per_share": ...,
            ...
          }
        ]
      }
    """
    params = {"ticker": symbol.upper(), "period": "annual", "limit": 10}
    data, err = fd_get_json("/financials", params)
    if err or not data:
        return None, err or "No financials."

    # Expect dict with "income_statements" list; if not, normalize.
    if isinstance(data, dict):
        if isinstance(data.get("income_statements"), list):
            return data, None
        # Some variants might wrap inside "financials"
        if isinstance(data.get("financials"), dict) and isinstance(data["financials"].get("income_statements"), list):
            return data["financials"], None
    return None, "Unexpected financials format."


# =========================================
# Stock snapshot built from FD only
# =========================================

def build_stock_snapshot(symbol: str) -> Dict[str, Any]:
    """
    Build the snapshot dict used throughout the report.

    Keys:
      symbol, long_name, sector, industry, website,
      current_price, day_change_pct, day_change_dollar,
      year_low, year_high, change_1y_pct
    """
    symbol = symbol.upper()
    snapshot: Dict[str, Any] = {
        "symbol": symbol,
        "long_name": symbol,
        "sector": "N/A",
        "industry": "N/A",
        "website": "N/A",
        "current_price": None,
        "day_change_pct": None,
        "day_change_dollar": None,
        "year_low": None,
        "year_high": None,
        "change_1y_pct": None,
    }

    # Company facts
    cf, cf_err = fetch_company_facts(symbol)
    if cf:
        snapshot["long_name"] = cf.get("name") or cf.get("company_name") or symbol
        snapshot["sector"] = cf.get("sector", "N/A")
        snapshot["industry"] = cf.get("industry", "N/A")
        snapshot["website"] = cf.get("website_url") or cf.get("website") or cf.get("homepage") or "N/A"
    else:
        dbg(f"{symbol}: company facts missing: {cf_err}")

    # Price history (use last ~400 days to be safe)
    df, price_err = fetch_price_history_df(symbol, days=400)
    if df is None or df.empty:
        dbg(f"{symbol}: price history missing: {price_err}")
        return snapshot

    close = df["close"].astype(float)
    snapshot["current_price"] = float(close.iloc[-1])

    # Day change using last two closes
    if len(close) > 1:
        prev = float(close.iloc[-2])
        if prev != 0:
            day_change = snapshot["current_price"] - prev
            snapshot["day_change_dollar"] = day_change
            snapshot["day_change_pct"] = (day_change / prev) * 100.0

    # 52-week high/low & 1Y change from same series
    snapshot["year_low"] = float(close.min())
    snapshot["year_high"] = float(close.max())
    first_price = float(close.iloc[0])
    if first_price not in (0, None):
        snapshot["change_1y_pct"] = ((snapshot["current_price"] - first_price) / first_price) * 100.0

    return snapshot


# =========================================
# Multi-year Fundamentals Table (from FD financials)
# =========================================

def build_fundamentals_table(financials: Dict[str, Any]) -> List[str]:
    lines: List[str] = []

    inc_list = financials.get("income_statements") or []
    if not inc_list:
        lines.append("MULTI-YEAR FUNDAMENTALS (Annual)")
        lines.append("-" * 70)
        lines.append("No income statement data returned.")
        lines.append("")
        return lines

    rows: List[Dict[str, Any]] = []
    for item in inc_list:
        if item.get("period") != "annual":
            continue
        report_period = item.get("report_period") or item.get("fiscal_period")
        if not report_period:
            continue
        year = report_period[:4]

        sales = item.get("revenue")
        gp = item.get("gross_profit")
        op = item.get("operating_income") or item.get("ebit")
        ni = item.get("net_income")
        eps = item.get("earnings_per_share")

        rows.append({
            "year": year,
            "sales": sales,
            "gp": gp,
            "op": op,
            "ni": ni,
            "eps": eps,
        })

    if not rows:
        lines.append("MULTI-YEAR FUNDAMENTALS (Annual)")
        lines.append("-" * 70)
        lines.append("No annual income statements found.")
        lines.append("")
        return lines

    rows.sort(key=lambda r: r["year"])

    def pct(num, den):
        try:
            if num is None or den in (None, 0):
                return None
            return (float(num) / float(den)) * 100.0
        except Exception:
            return None

    def fmt_pct(value, decimals=1):
        if value is None:
            return "   N/A"
        return f"{value:6.{decimals}f}%"

    prev_sales = None
    prev_eps = None
    for r in rows:
        s = r["sales"]
        gp = r["gp"]
        op = r["op"]
        ni = r["ni"]
        eps = r["eps"]

        r["gp_margin"] = pct(gp, s)
        r["op_margin"] = pct(op, s)
        r["ni_margin"] = pct(ni, s)

        if prev_sales not in (None, 0) and s not in (None, 0):
            r["sales_yoy"] = pct(s - prev_sales, prev_sales)
        else:
            r["sales_yoy"] = None

        if prev_eps not in (None, 0) and eps not in (None, 0):
            r["eps_yoy"] = pct(eps - prev_eps, prev_eps)
        else:
            r["eps_yoy"] = None

        prev_sales = s
        prev_eps = eps

    # Last 5 years only
    if len(rows) > 5:
        rows = rows[-5:]

    lines.append("MULTI-YEAR FUNDAMENTALS (Annual)")
    lines.append("-" * 70)
    header = (
        "Year   "
        "Sales        GP           OpInc        NetInc       EPS    "
        "GP%    OP%    NI%   SlsYoY  EPSYoY"
    )
    lines.append(header)
    lines.append("-" * 70)

    for r in rows:
        year = r["year"]
        s = fmt_int(r["sales"])
        gp = fmt_int(r["gp"])
        op = fmt_int(r["op"])
        ni = fmt_int(r["ni"])
        eps = fmt_number(r["eps"], 2)

        gp_m = fmt_pct(r["gp_margin"])
        op_m = fmt_pct(r["op_margin"])
        ni_m = fmt_pct(r["ni_margin"])
        sy = fmt_pct(r["sales_yoy"])
        ey = fmt_pct(r["eps_yoy"])

        line = (
            f"{year:<6}"
            f"{s:>12} {gp:>12} {op:>12} {ni:>12} {eps:>7}  "
            f"{gp_m:>6} {op_m:>6} {ni_m:>6} {sy:>7} {ey:>7}"
        )
        lines.append(line)

    lines.append("")
    return lines


# =========================================
# OpenAI – AI Fundamental + AI Freelancing (single & pair)
# =========================================

def build_ai_single_prompt(symbol, snapshot, fm_snapshot, analyst_estimates, company_facts):
    cik = (company_facts or {}).get("cik", "")
    return f"""
You are a senior equity analyst evaluating {symbol}.
Suppress all ### and ** formatting.

Write a structured investment summary with these sections:

1. Company Overview
2. Stock Performance (1Y, day change, key levels)
3. Valuation (use P/E, P/B, P/S, EV/EBITDA, EV/Sales where available)
4. Growth & Profitability (margins, ROE, etc. if present)
5. Analyst Estimates & Expectations
6. Key Risks
7. Final Verdict (Buy / Hold / Avoid — not advice)

Useful Links:
- Google News: https://news.google.com/search?q={symbol}+stock
- Yahoo Finance: https://finance.yahoo.com/quote/{symbol}
- SEC Filings: https://www.sec.gov/edgar/browse/?CIK={cik}
- MarketWatch: https://www.marketwatch.com/investing/stock/{symbol}
- Company Website: {(company_facts or {}).get('website_url', 'N/A')}

Snapshot:
{json.dumps(snapshot, indent=2)}

Financial Metrics Snapshot:
{json.dumps(fm_snapshot, indent=2)}

Analyst Estimates:
{json.dumps(analyst_estimates, indent=2)}

Company Facts:
{json.dumps(company_facts, indent=2)}
"""


def generate_ai_fundamental_single(symbol, snapshot, fm_snapshot, analyst_estimates, company_facts):
    if not OPENAI_API_KEY:
        return "OpenAI key missing; cannot generate AI fundamental summary."

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = build_ai_single_prompt(symbol, snapshot, fm_snapshot, analyst_estimates, company_facts)
    try:
        res = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You are a financial analyst. No ### or **."},
                {"role": "user", "content": prompt},
            ],
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}\n{traceback.format_exc()}"


def generate_ai_freelancing_single(symbol: str):
    if not OPENAI_API_KEY:
        return "OpenAI key missing; cannot generate AI freelancing view."

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        f"Tell me all I should know about {symbol}. "
        "Include business model, strategy, valuation, competition, risks, catalysts, "
        "and long-term outlook. No ### or **."
    )
    try:
        res = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.4,
            messages=[
                {"role": "system", "content": "You are an expert financial analyst."},
                {"role": "user", "content": prompt},
            ],
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Freelancing AI Error: {e}"


def generate_ai_combined_pair(
    s1: str,
    d1: Dict[str, Any],
    fm1: Optional[Dict[str, Any]],
    analyst1: Optional[List[Dict[str, Any]]],
    facts1: Optional[Dict[str, Any]],
    s2: str,
    d2: Dict[str, Any],
    fm2: Optional[Dict[str, Any]],
    analyst2: Optional[List[Dict[str, Any]]],
    facts2: Optional[Dict[str, Any]],
) -> str:
    if not OPENAI_API_KEY:
        return "OpenAI key missing; cannot generate AI comparison."

    client = OpenAI(api_key=OPENAI_API_KEY)
    payload = {
        "snapshot_1": d1,
        "metrics_1": fm1,
        "analyst_1": analyst1,
        "facts_1": facts1,
        "snapshot_2": d2,
        "metrics_2": fm2,
        "analyst_2": analyst2,
        "facts_2": facts2,
    }
    prompt = f"""
You are a senior equity analyst comparing two stocks: {s1} and {s2}.
Do NOT use ### or ** markdown.

Provide BOTH:

1) AI FUNDAMENTAL COMPARISON SUMMARY
   - Compare their business models, competitive positions, growth, margins,
     balance sheet quality, valuation, risks, and long-term outlook.
   - Explicitly note where one appears stronger/weaker vs the other.

2) AI FREELANCING VIEW
   - "Tell me all I should know" about each stock for an informed investor:
     strategy, key products/segments, major secular trends, management,
     capital allocation, catalysts, red flags, and scenario analysis.
   - Weave this into the comparison, but make it clear which points
     apply to {s1} and which to {s2}.

Required sections in your answer:
1. Business Overview & Competitive Position
2. Stock Performance & Momentum
3. Valuation Comparison (P/E, P/B, P/S, EV/EBITDA, EV/Sales where available)
4. Growth, Profitability & Balance Sheet Quality
5. Key Risks & Downside Scenarios
6. Key Catalysts & Upside Scenarios
7. Overall Assessment – which looks more attractive right now and why
   (clearly labeled as NOT investment advice).

Data:
{json.dumps(payload, indent=2)}
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You are a senior equity analyst. No ### or **."},
                {"role": "user", "content": prompt},
            ],
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}\n{traceback.format_exc()}"


# =========================================
# Chart builders (FD prices → pandas → matplotlib)
# =========================================

def build_single_charts(symbol: str, fm_history: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    df, err = fetch_price_history_df(symbol, days=365)
    if df is None or df.empty:
        dbg(f"{symbol}: no price data for charts: {err}")
        return None

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    # Moving averages
    hist = pd.DataFrame({"Close": close, "Volume": volume})
    hist["MA20"] = hist["Close"].rolling(window=20).mean()
    hist["MA50"] = hist["Close"].rolling(window=50).mean()
    hist["MA200"] = hist["Close"].rolling(window=200).mean()

    # RSI
    delta = hist["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # MACD
    ema12 = hist["Close"].ewm(span=12, adjust=False).mean()
    ema26 = hist["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    fig, axs = plt.subplots(5, 1, figsize=(8.5, 11), sharex=False)
    fig.subplots_adjust(hspace=0.4)

    dates = hist.index

    # Price + MAs
    ax_price = axs[0]
    ax_price.plot(dates, hist["Close"], label="Close")
    ax_price.plot(dates, hist["MA20"], label="MA20", linewidth=0.8)
    ax_price.plot(dates, hist["MA50"], label="MA50", linewidth=0.8)
    ax_price.plot(dates, hist["MA200"], label="MA200", linewidth=0.8)
    ax_price.set_title(f"{symbol} Price + Moving Averages")
    ax_price.legend(loc="upper left", fontsize=7)

    # Volume
    ax_vol = axs[1]
    ax_vol.bar(dates, hist["Volume"] / 1_000_000.0, width=1.0)
    ax_vol.set_title("Daily Volume (M)")
    ax_vol.set_ylabel("Shares (M)")

    # RSI
    ax_rsi = axs[2]
    ax_rsi.plot(dates, rsi)
    ax_rsi.axhline(70, color="red", linestyle="--", linewidth=0.8)
    ax_rsi.axhline(30, color="green", linestyle="--", linewidth=0.8)
    ax_rsi.set_title("RSI (14)")
    ax_rsi.set_ylim(0, 100)

    # MACD
    ax_macd = axs[3]
    ax_macd.plot(dates, macd, label="MACD")
    ax_macd.plot(dates, signal, label="Signal", linestyle="--")
    ax_macd.axhline(0, color="black", linewidth=0.8)
    ax_macd.legend(fontsize=7)
    ax_macd.set_title("MACD (12/26/9)")

    # Valuation history from FM history
    ax_val = axs[4]
    if fm_history:
        records = []
        for item in fm_history:
            lbl = item.get("report_period") or item.get("fiscal_period") or item.get("date")
            pe = item.get("price_to_earnings_ratio")
            ev_ebitda = item.get("enterprise_value_to_ebitda_ratio")
            ev_sales = item.get("enterprise_value_to_revenue_ratio")
            if lbl and (pe is not None or ev_ebitda is not None or ev_sales is not None):
                records.append((lbl, pe, ev_ebitda, ev_sales))
        if records:
            records.sort(key=lambda x: x[0])
            labels = [r[0] for r in records]
            pe_vals = [r[1] for r in records]
            ev_ebitda_vals = [r[2] for r in records]
            ev_sales_vals = [r[3] for r in records]
            x_idx = range(len(labels))
            if any(v is not None for v in pe_vals):
                ax_val.plot(list(x_idx), pe_vals, marker="o", label="P/E")
            if any(v is not None for v in ev_ebitda_vals):
                ax_val.plot(list(x_idx), ev_ebitda_vals, marker="o", label="EV/EBITDA")
            if any(v is not None for v in ev_sales_vals):
                ax_val.plot(list(x_idx), ev_sales_vals, marker="o", label="EV/Sales")
            ax_val.set_title("Valuation Over Time")
            ax_val.set_xticks(list(x_idx))
            ax_val.set_xticklabels(labels, rotation=45, fontsize=7)
            ax_val.legend(fontsize=7)
        else:
            ax_val.text(0.5, 0.5, "No valuation history available.", ha="center", va="center")
            ax_val.axis("off")
    else:
        ax_val.text(0.5, 0.5, "No valuation history available.", ha="center", va="center")
        ax_val.axis("off")

    tmpdir = os.path.join(BASE_DIR, "generated_reports")
    os.makedirs(tmpdir, exist_ok=True)
    img_path = os.path.join(tmpdir, f"single_{symbol}_charts.png")
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return img_path


def build_compare_charts(s1: str, s2: str) -> Optional[str]:
    df1, err1 = fetch_price_history_df(s1, days=365)
    df2, err2 = fetch_price_history_df(s2, days=365)

    if df1 is None or df1.empty or df2 is None or df2.empty:
        dbg(f"Compare charts: missing price data for {s1} or {s2}: {err1}, {err2}")
        return None

    close1 = df1["close"].astype(float)
    close2 = df2["close"].astype(float)

    fig, axs = plt.subplots(3, 1, figsize=(8.5, 11), sharex=False)
    fig.subplots_adjust(hspace=0.35)

    # Price history
    ax1 = axs[0]
    ax1.plot(close1.index, close1, label=f"{s1} Close")
    ax1.plot(close2.index, close2, label=f"{s2} Close")
    ax1.set_title("Price History (1Y)")
    ax1.legend(fontsize=8)

    # Indexed performance
    ax2 = axs[1]
    base1 = close1.iloc[0]
    base2 = close2.iloc[0]
    norm1 = close1 / base1 * 100.0
    norm2 = close2 / base2 * 100.0
    ax2.plot(norm1.index, norm1, label=f"{s1} (Indexed to 100)")
    ax2.plot(norm2.index, norm2, label=f"{s2} (Indexed to 100)")
    ax2.set_title("Indexed Performance (Start = 100)")
    ax2.legend(fontsize=8)

    # P/E history from financial-metrics
    fm_hist1, _ = fetch_financial_metrics_history(s1, "annual", 10)
    fm_hist2, _ = fetch_financial_metrics_history(s2, "annual", 10)

    ax3 = axs[2]
    recs1: List[Tuple[str, Optional[float]]] = []
    recs2: List[Tuple[str, Optional[float]]] = []

    if fm_hist1:
        for item in fm_hist1:
            lbl = item.get("report_period") or item.get("fiscal_period") or item.get("date")
            pe = item.get("price_to_earnings_ratio")
            if lbl and pe is not None:
                recs1.append((lbl, pe))
    if fm_hist2:
        for item in fm_hist2:
            lbl = item.get("report_period") or item.get("fiscal_period") or item.get("date")
            pe = item.get("price_to_earnings_ratio")
            if lbl and pe is not None:
                recs2.append((lbl, pe))

    if recs1 or recs2:
        if recs1:
            recs1.sort(key=lambda x: x[0])
            x1 = range(len(recs1))
            y1 = [r[1] for r in recs1]
            ax3.plot(list(x1), y1, marker="o", label=f"{s1} P/E")
        if recs2:
            recs2.sort(key=lambda x: x[0])
            x2 = range(len(recs2))
            y2 = [r[1] for r in recs2]
            ax3.plot(list(x2), y2, marker="o", label=f"{s2} P/E")
        ax3.set_title("P/E Over Time (Annual)")
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No P/E history from financial-metrics.", ha="center", va="center")
        ax3.axis("off")

    tmpdir = os.path.join(BASE_DIR, "generated_reports")
    os.makedirs(tmpdir, exist_ok=True)
    img_path = os.path.join(tmpdir, f"compare_{s1}_{s2}_charts.png")
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return img_path


# =========================================
# PDF export (shared)
# =========================================

def is_real_section_header(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    if any(ch.isdigit() for ch in s):
        return False
    if "," in s or "%" in s:
        return False
    if len(s) > 35:
        return False
    alpha = "".join(ch for ch in s if ch.isalpha())
    if not alpha:
        return False
    return alpha.isupper()


def export_pdf(text: str, title_line: str, chart_path: Optional[str], output_path: str):
    PAGE_WIDTH, PAGE_HEIGHT = letter

    def header(canvas, doc):
        if canvas.getPageNumber() == 1:
            return
        canvas.setFont("Times-Bold", 11)
        canvas.drawString(50, PAGE_HEIGHT - 40, title_line)
        canvas.setFont("Times-Roman", 9)
        canvas.drawRightString(PAGE_WIDTH - 50, 40, f"Page {canvas.getPageNumber()}")

    frame = Frame(50, 50, PAGE_WIDTH - 100, PAGE_HEIGHT - 120)
    template = PageTemplate(id="Report", frames=frame, onPage=header)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=70,
        bottomMargin=50,
    )
    doc.addPageTemplates([template])

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name="Title",
        fontName="Times-Bold",
        fontSize=24,
        alignment=TA_CENTER,
        spaceAfter=30,
    )

    descriptor_style = ParagraphStyle(
        name="Descriptor",
        fontName="Times-Italic",
        fontSize=12,
        alignment=TA_CENTER,
        spaceAfter=40,
    )

    header_style = ParagraphStyle(
        name="Header",
        fontName="Times-Bold",
        fontSize=14,
        spaceBefore=20,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        name="Body",
        fontName="Times-Roman",
        fontSize=10,
        leading=14,
    )

    link_style = ParagraphStyle(
        name="Link",
        fontName="Times-Roman",
        fontSize=10,
        leading=14,
        textColor="blue",
        underline=True,
    )

    story = []

    # Title page
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph("John Pirisino's Stock Analyzer", title_style))
    story.append(Paragraph(title_line, descriptor_style))
    story.append(PageBreak())

    import re
    url_regex = r"(https?://[^\s]+)"

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped == "":
            story.append(Spacer(1, 10))
            continue

        safe = (
            line.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        m = re.search(url_regex, safe)
        if m:
            url = m.group(1)
            link = f'<link href="{url}"><u>{url}</u></link>'
            story.append(Paragraph(link, link_style))
            continue

        if is_real_section_header(stripped):
            story.append(Paragraph(f"<b>{safe}</b>", header_style))
        else:
            story.append(Paragraph(safe, body_style))

    if chart_path and os.path.exists(chart_path):
        story.append(PageBreak())
        story.append(Paragraph("Charts & Visuals", header_style))
        story.append(Spacer(1, 12))
        img = Image(chart_path)
        img._restrictSize(6.5 * inch, 9 * inch)
        story.append(img)

    doc.build(story)


# =========================================
# PUBLIC ENTRY POINTS (used by Flask app)
# =========================================

def run_single_to_pdf(symbol: str, out_dir: str) -> str:
    symbol = symbol.upper()
    os.makedirs(out_dir, exist_ok=True)

    snapshot = build_stock_snapshot(symbol)
    fm_snapshot, fm_err = fetch_financial_metrics_snapshot(symbol)
    fm_history, fm_hist_err = fetch_financial_metrics_history(symbol, "annual", 10)
    analyst, analyst_err = fetch_fd_analyst_estimates(symbol)
    facts, facts_err = fetch_company_facts(symbol)
    financials, fin_err = fetch_financials(symbol)
    news, news_err = fetch_news(symbol)
    insider, insider_err = fetch_insider_trades(symbol)
    inst, inst_err = fetch_institutional(symbol)

    lines: List[str] = []

    # Snapshot
    lines.append("=" * 72)
    lines.append(f"STOCK SNAPSHOT: {symbol}")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Name     : {snapshot['long_name']}")
    lines.append(f"Sector   : {snapshot['sector']}")
    lines.append(f"Industry : {snapshot['industry']}")
    lines.append(f"Website  : {snapshot['website']}")
    lines.append("")
    lines.append("PRICE")
    lines.append("-" * 72)
    lines.append(f"Current Price      : {fmt_number(snapshot['current_price'])}")
    lines.append(f"Day Change (%)     : {fmt_number(snapshot['day_change_pct'])}%")
    lines.append(f"Day Change ($)     : {fmt_number(snapshot['day_change_dollar'])}")
    lines.append(f"52W Low            : {fmt_number(snapshot['year_low'])}")
    lines.append(f"52W High           : {fmt_number(snapshot['year_high'])}")
    lines.append(f"1Y Change (%)      : {fmt_number(snapshot['change_1y_pct'])}%")
    lines.append("")

    # AI Fundamental
    lines.append("=" * 72)
    lines.append("AI FUNDAMENTAL SUMMARY")
    lines.append("=" * 72)
    lines.append("")
    lines.append(
        generate_ai_fundamental_single(symbol, snapshot, fm_snapshot, analyst, facts)
    )
    lines.append("")

    # AI Freelancing
    lines.append("=" * 72)
    lines.append("AI FREELANCING SUMMARY")
    lines.append("=" * 72)
    lines.append("")
    lines.append(generate_ai_freelancing_single(symbol))
    lines.append("")

    # Company facts
    lines.append("COMPANY FACTS")
    lines.append("-" * 72)
    if facts:
        for k, v in facts.items():
            if isinstance(v, list):
                continue
            lines.append(f"{k:25}: {v}")
    else:
        lines.append("No company facts available." + (f" ({facts_err})" if facts_err else ""))
    lines.append("")

    # Multi-year fundamentals
    if financials:
        lines.extend(build_fundamentals_table(financials))
    else:
        lines.append("MULTI-YEAR FUNDAMENTALS (Annual)")
        lines.append("-" * 70)
        lines.append("No financials available for multi-year view." + (f" ({fin_err})" if fin_err else ""))
        lines.append("")

    # Financial metrics snapshot
    lines.append("FINANCIAL METRICS SNAPSHOT (FinancialDatasets.ai)")
    lines.append("-" * 72)
    if fm_snapshot:
        fm = fm_snapshot
        lines.append(f"Market Cap              : {fmt_int(fm.get('market_cap'))}")
        lines.append(f"Enterprise Value        : {fmt_int(fm.get('enterprise_value'))}")
        lines.append(f"P/E Ratio               : {fmt_number(fm.get('price_to_earnings_ratio'),4)}")
        lines.append(f"P/B Ratio               : {fmt_number(fm.get('price_to_book_ratio'),4)}")
        lines.append(f"P/S Ratio               : {fmt_number(fm.get('price_to_sales_ratio'),4)}")
        lines.append(f"EV/EBITDA               : {fmt_number(fm.get('enterprise_value_to_ebitda_ratio'),4)}")
        lines.append(f"EV/Sales                : {fmt_number(fm.get('enterprise_value_to_revenue_ratio'),4)}")
        lines.append(f"Free Cash Flow Yield    : {fmt_number(fm.get('free_cash_flow_yield'),4)}")
        lines.append(f"PEG Ratio               : {fmt_number(fm.get('peg_ratio'),4)}")
        lines.append(f"Gross Margin            : {fmt_number(fm.get('gross_margin'),4)}")
        lines.append(f"Operating Margin        : {fmt_number(fm.get('operating_margin'),4)}")
        lines.append(f"Net Margin              : {fmt_number(fm.get('net_margin'),4)}")
        lines.append(f"Return on Equity        : {fmt_number(fm.get('return_on_equity'),4)}")
        lines.append(f"Return on Assets        : {fmt_number(fm.get('return_on_assets'),4)}")
        lines.append(f"Return on Invested Cap. : {fmt_number(fm.get('return_on_invested_capital'),4)}")
        lines.append(f"Current Ratio           : {fmt_number(fm.get('current_ratio'),4)}")
        lines.append(f"Quick Ratio             : {fmt_number(fm.get('quick_ratio'),4)}")
        lines.append(f"Cash Ratio              : {fmt_number(fm.get('cash_ratio'),4)}")
        lines.append(f"Debt to Equity          : {fmt_number(fm.get('debt_to_equity'),4)}")
        lines.append(f"Debt to Assets          : {fmt_number(fm.get('debt_to_assets'),4)}")
        lines.append(f"Interest Coverage       : {fmt_number(fm.get('interest_coverage'),4)}")
        lines.append(f"Revenue Growth          : {fmt_number(fm.get('revenue_growth'),4)}")
        lines.append(f"Earnings Growth         : {fmt_number(fm.get('earnings_growth'),4)}")
        lines.append(f"Book Value Growth       : {fmt_number(fm.get('book_value_growth'),4)}")
        lines.append(f"EPS Growth              : {fmt_number(fm.get('earnings_per_share_growth'),4)}")
        lines.append(f"Free Cash Flow Growth   : {fmt_number(fm.get('free_cash_flow_growth'),4)}")
        lines.append(f"Operating Income Growth : {fmt_number(fm.get('operating_income_growth'),4)}")
        lines.append(f"EBITDA Growth           : {fmt_number(fm.get('ebitda_growth'),4)}")
        lines.append(f"Payout Ratio            : {fmt_number(fm.get('payout_ratio'),4)}")
        lines.append(f"Earnings Per Share      : {fmt_number(fm.get('earnings_per_share'),4)}")
        lines.append(f"Book Value Per Share    : {fmt_number(fm.get('book_value_per_share'),4)}")
        lines.append(f"Free Cash Flow Per Share: {fmt_number(fm.get('free_cash_flow_per_share'),4)}")
    else:
        lines.append("No snapshot metrics." + (f" ({fm_err})" if fm_err else ""))
    lines.append("")

    # Analyst estimates
    lines.append("ANALYST ESTIMATES (Annual)")
    lines.append("-" * 72)
    if analyst:
        for e in analyst:
            lines.append(f"Fiscal Period : {e.get('fiscal_period','N/A')}")
            lines.append(f"Period        : {e.get('period','N/A')}")
            lines.append(f"EPS Estimate  : {fmt_number(e.get('earnings_per_share'),4)}")
            lines.append("")
    else:
        msg = "No analyst estimates."
        if analyst_err:
            msg += f" ({analyst_err})"
        lines.append(msg)
    lines.append("")

    # Insider trades (top 5)
    lines.append("INSIDER TRADES (Recent)")
    lines.append("-" * 72)
    if insider:
        for t in insider[:5]:
            lines.append(
                f"{t.get('transaction_date','N/A')} | "
                f"{t.get('name','N/A')} | "
                f"Shares: {fmt_int(t.get('transaction_shares'))}"
            )
    else:
        msg = "No insider trades."
        if insider_err:
            msg += f" ({insider_err})"
        lines.append(msg)
    lines.append("")

    # Institutional (top 10, simple)
    lines.append("INSTITUTIONAL OWNERSHIP (Top 10)")
    lines.append("-" * 72)
    if inst:
        sorted_inst = sorted(inst, key=lambda x: x.get("shares") or 0, reverse=True)[:10]
        for h in sorted_inst:
            lines.append(
                f"{h.get('investor','N/A')[:40]:40} {fmt_int(h.get('shares')):>12}"
            )
    else:
        msg = "No institutional ownership data."
        if inst_err:
            msg += f" ({inst_err})"
        lines.append(msg)
    lines.append("")

    # News
    lines.append("LATEST NEWS")
    lines.append("-" * 72)
    if news:
        for n in news:
            lines.append(f"{n.get('date','N/A')} - {n.get('title','N/A')}")
            lines.append(f"URL: {n.get('url','N/A')}")
            lines.append("")
    else:
        msg = "No news available."
        if news_err:
            msg += f" ({news_err})"
        lines.append(msg)
    lines.append("")

    # Charts
    chart_path = build_single_charts(symbol, fm_history)

    # Output path
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(out_dir, f"{symbol}_{ts}.pdf")
    title_line = f"{symbol} – {snapshot['long_name']}"
    export_pdf("\n".join(lines), title_line, chart_path, out_file)
    return out_file


def run_compare_to_pdf(s1: str, s2: str, out_dir: str) -> str:
    s1 = s1.upper()
    s2 = s2.upper()
    os.makedirs(out_dir, exist_ok=True)

    d1 = build_stock_snapshot(s1)
    d2 = build_stock_snapshot(s2)

    fm1, fm1_err = fetch_financial_metrics_snapshot(s1)
    fm2, fm2_err = fetch_financial_metrics_snapshot(s2)

    analyst1, analyst1_err = fetch_fd_analyst_estimates(s1)
    analyst2, analyst2_err = fetch_fd_analyst_estimates(s2)

    facts1, facts1_err = fetch_company_facts(s1)
    facts2, facts2_err = fetch_company_facts(s2)

    news1, news1_err = fetch_news(s1)
    news2, news2_err = fetch_news(s2)

    inst1, inst1_err = fetch_institutional(s1)
    inst2, inst2_err = fetch_institutional(s2)

    insider1, insider1_err = fetch_insider_trades(s1)
    insider2, insider2_err = fetch_insider_trades(s2)

    lines: List[str] = []

    def s2_num_line(label: str, v1, v2, decimals=2):
        left = f"{label:<18}{fmt_number(v1, decimals):>12}"
        right = f"{label:<18}{fmt_number(v2, decimals):>12}"
        lines.append(f"{left}    {right}")

    def s2_text_line(label: str, t1: str, t2: str):
        left = f"{label:<12}{(t1 or '')[:28]:<28}"
        right = f"{label:<12}{(t2 or '')[:28]:<28}"
        lines.append(f"{left}    {right}")

    # Header
    lines.append("=" * 72)
    lines.append(f"COMPARISON: {s1} vs {s2}")
    lines.append("=" * 72)
    lines.append("")

    # Basic info
    lines.append("BASIC INFO")
    lines.append("-" * 72)
    lines.append(f"{s1:<40}{s2:<40}")
    lines.append("")
    s2_text_line("Name", d1["long_name"], d2["long_name"])
    s2_text_line("Sector", d1["sector"], d2["sector"])
    s2_text_line("Industry", d1["industry"], d2["industry"])
    s2_text_line("Website", d1["website"], d2["website"])
    lines.append("")

    # Price & performance
    lines.append("PRICE & PERFORMANCE")
    lines.append("-" * 72)
    lines.append(f"{s1:<40}{s2:<40}")
    lines.append("")
    s2_num_line("Current Price", d1["current_price"], d2["current_price"])
    s2_num_line("Day Change (%)", d1["day_change_pct"], d2["day_change_pct"])
    s2_num_line("Day Change ($)", d1["day_change_dollar"], d2["day_change_dollar"])
    s2_num_line("52W Low", d1["year_low"], d2["year_low"])
    s2_num_line("52W High", d1["year_high"], d2["year_high"])
    s2_num_line("1Y Change (%)", d1["change_1y_pct"], d2["change_1y_pct"])
    lines.append("")

    # AI fundamentals + freelancing combined
    lines.append("=" * 72)
    lines.append("AI FUNDAMENTAL & FREELANCING COMPARISON SUMMARY")
    lines.append("=" * 72)
    lines.append("")
    lines.append(
        generate_ai_combined_pair(
            s1, d1, fm1, analyst1, facts1,
            s2, d2, fm2, analyst2, facts2
        )
    )
    lines.append("")

    # Metrics snapshot
    lines.append("FINANCIAL METRICS SNAPSHOT")
    lines.append("-" * 72)
    lines.append(f"{s1:<40}{s2:<40}")
    lines.append("")
    if fm1 and fm2:
        s2_num_line("Market Cap", (fm1 or {}).get("market_cap"), (fm2 or {}).get("market_cap"), 0)
        s2_num_line("Enterprise Val", (fm1 or {}).get("enterprise_value"), (fm2 or {}).get("enterprise_value"), 0)
        s2_num_line("P/E Ratio", (fm1 or {}).get("price_to_earnings_ratio"), (fm2 or {}).get("price_to_earnings_ratio"))
        s2_num_line("P/B Ratio", (fm1 or {}).get("price_to_book_ratio"), (fm2 or {}).get("price_to_book_ratio"))
        s2_num_line("P/S Ratio", (fm1 or {}).get("price_to_sales_ratio"), (fm2 or {}).get("price_to_sales_ratio"))
        s2_num_line("EV/EBITDA", (fm1 or {}).get("enterprise_value_to_ebitda_ratio"),
                    (fm2 or {}).get("enterprise_value_to_ebitda_ratio"))
        s2_num_line("EV/Sales", (fm1 or {}).get("enterprise_value_to_revenue_ratio"),
                    (fm2 or {}).get("enterprise_value_to_revenue_ratio"))
        s2_num_line("Gross Margin", (fm1 or {}).get("gross_margin"), (fm2 or {}).get("gross_margin"))
        s2_num_line("Op Margin", (fm1 or {}).get("operating_margin"), (fm2 or {}).get("operating_margin"))
        s2_num_line("Net Margin", (fm1 or {}).get("net_margin"), (fm2 or {}).get("net_margin"))
    else:
        msg = "Metrics missing for one or both tickers."
        if fm1_err or fm2_err:
            msg += f" ({fm1_err or ''} {fm2_err or ''})"
        lines.append(msg)
    lines.append("")

    # Analyst estimates
    lines.append("ANALYST ESTIMATES (Annual)")
    lines.append("-" * 72)
    lines.append(f"{s1:<40}{s2:<40}")
    lines.append("")
    max_rows = max(len(analyst1 or []), len(analyst2 or []))
    if max_rows == 0:
        msg = "No analyst estimates for either ticker."
        if analyst1_err or analyst2_err:
            msg += f" ({analyst1_err or ''} {analyst2_err or ''})"
        lines.append(msg)
    else:
        for i in range(max_rows):
            left = analyst1[i] if analyst1 and i < len(analyst1) else None
            right = analyst2[i] if analyst2 and i < len(analyst2) else None
            lp = left.get("fiscal_period") if left else "N/A"
            rp = right.get("fiscal_period") if right else "N/A"
            leps = left.get("earnings_per_share") if left else None
            reps = right.get("earnings_per_share") if right else None
            s2_text_line("Year", lp, rp)
            s2_num_line("EPS Est", leps, reps, 4)
            lines.append("")
    lines.append("")

    # Institutional
    lines.append("INSTITUTIONAL OWNERSHIP (Top 5 by Shares)")
    lines.append("-" * 72)
    lines.append(f"{s1:<40}{s2:<40}")
    lines.append("")
    top1: List[Dict[str, Any]] = []
    top2: List[Dict[str, Any]] = []
    if inst1:
        top1 = sorted(inst1, key=lambda x: x.get("shares") or 0, reverse=True)[:5]
    if inst2:
        top2 = sorted(inst2, key=lambda x: x.get("shares") or 0, reverse=True)[:5]

    max_len = max(len(top1), len(top2))
    if max_len == 0:
        msg = "No institutional ownership data."
        if inst1_err or inst2_err:
            msg += f" ({inst1_err or ''} {inst2_err or ''})"
        lines.append(msg)
    else:
        for i in range(max_len):
            left = top1[i] if i < len(top1) else None
            right = top2[i] if i < len(top2) else None
            lname = (left or {}).get("investor", "N/A")
            rname = (right or {}).get("investor", "N/A")
            lsh = (left or {}).get("shares")
            rsh = (right or {}).get("shares")
            left_txt = f"{lname[:30]:30} {fmt_int(lsh):>10}"
            right_txt = f"{rname[:30]:30} {fmt_int(rsh):>10}"
            lines.append(f"{left_txt}    {right_txt}")
    lines.append("")

    # Insider trades
    lines.append("INSIDER TRADES (Recent)")
    lines.append("-" * 72)
    lines.append(f"{s1:<40}{s2:<40}")
    lines.append("")
    max_len = max(len(insider1), len(insider2))
    if max_len == 0:
        msg = "No insider trades for either ticker."
        if insider1_err or insider2_err:
            msg += f" ({insider1_err or ''} {insider2_err or ''})"
        lines.append(msg)
    else:
        for i in range(min(max_len, 5)):
            left = insider1[i] if i < len(insider1) else None
            right = insider2[i] if i < len(insider2) else None
            ltxt = "N/A"
            rtxt = "N/A"
            if left:
                ltxt = f"{left.get('transaction_date','N/A')} {left.get('name','N/A')[:20]:20} Sh:{fmt_int(left.get('transaction_shares'))}"
            if right:
                rtxt = f"{right.get('transaction_date','N/A')} {right.get('name','N/A')[:20]:20} Sh:{fmt_int(right.get('transaction_shares'))}"
            lines.append(f"{ltxt:<40}    {rtxt:<40}")
    lines.append("")

    # News
    lines.append("LATEST NEWS")
    lines.append("-" * 72)
    lines.append(f"{s1:<60}{s2:<60}")
    lines.append("")
    max_len = max(len(news1), len(news2))
    if max_len == 0:
        msg = "No news for either ticker."
        if news1_err or news2_err:
            msg += f" ({news1_err or ''} {news2_err or ''})"
        lines.append(msg)
    else:
        for i in range(max_len):
            left = news1[i] if i < len(news1) else None
            right = news2[i] if i < len(news2) else None
            if left:
                ltitle = f"{left.get('date','N/A')} - {left.get('title','N/A')}"
                lurl = left.get("url", "")
            else:
                ltitle = ""
                lurl = ""
            if right:
                rtitle = f"{right.get('date','N/A')} - {right.get('title','N/A')}"
                rurl = right.get("url", "")
            else:
                rtitle = ""
                rurl = ""
            line_title = f"{ltitle[:55]:55}    {rtitle[:55]:55}".rstrip()
            lines.append(line_title)
            if lurl or rurl:
                line_url = f"{('URL: ' + lurl)[:55]:55}    {('URL: ' + rurl)[:55]:55}".rstrip()
                lines.append(line_url)
            lines.append("")
    lines.append("")

    chart_path = build_compare_charts(s1, s2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(out_dir, f"{s1}_{s2}_{ts}.pdf")
    title_line = f"{s1} vs {s2}"
    export_pdf("\n".join(lines), title_line, chart_path, out_file)
    return out_file
