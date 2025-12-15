# engine.py (v2 - full rewrite, production-robust, Railway-safe)
# -------------------------------------------------------------------
# Goals:
# - Never crash due to missing helper functions (clean module layout)
# - Use FinancialDatasets.ai (FD) as primary source for fundamentals + snapshots
# - Hybrid pricing:
#     1) FD /prices (primary; handles redirects)
#     2) yfinance (fallback; defensive against 429 / parsing failures)
# - PDF output with modern "grid" tables (ReportLab Table) for:
#     - Multi-Year Fundamentals
#     - Financial Metrics Snapshot
#     - Analyst Estimates
#     - Insider Trades
#     - Institutional Ownership
# - AI prompts:
#     - Pass BOTH ticker + company name to AI to disambiguate
#     - Force "analysis only": no greetings, no small talk, no preamble
# - Controlled debugging:
#     - DEBUG_ENGINE=Y enables concise logs (no giant JSON dumps)
#
# Public entry points (imported by app.py):
#   - run_single_to_pdf(symbol: str, out_dir: str) -> str
#   - run_compare_to_pdf(s1: str, s2: str, out_dir: str) -> str
# -------------------------------------------------------------------

import os
import json
import time
import math
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd
from dotenv import load_dotenv

# OpenAI (optional)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# yfinance (optional fallback)
try:
    import yfinance as yf
except Exception:
    yf = None  # type: ignore

# matplotlib (charts)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ReportLab (PDF)
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Frame, PageTemplate,
    Image as RLImage, Table, TableStyle, KeepTogether
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import inch
from reportlab.lib import colors


# ==========================================================
# ENV / CONFIG
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FD_API_KEY = os.getenv("FINANCIAL_DATASETS_API_KEY")
FD_BASE_URL = os.getenv("FD_BASE_URL", "https://api.financialdatasets.ai").rstrip("/")

DEBUG = os.getenv("DEBUG_ENGINE", "N").upper() == "Y"
REQUEST_TIMEOUT = int(os.getenv("FD_TIMEOUT_SECONDS", "30"))  # FD calls
YF_TIMEOUT = int(os.getenv("YF_TIMEOUT_SECONDS", "20"))

# Keep logs sane (Railway drops logs if too noisy)
def dbg(msg: str) -> None:
    if DEBUG:
        print(f"[ENGINE DEBUG] {msg}", flush=True)


# ==========================================================
# FORMAT HELPERS
# ==========================================================

def _is_number(x: Any) -> bool:
    try:
        if x is None:
            return False
        float(x)
        return True
    except Exception:
        return False


def fmt_int(value: Any) -> str:
    try:
        if value is None:
            return "N/A"
        if isinstance(value, bool):
            return "N/A"
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return "N/A"
        return f"{int(round(v)):,}"
    except Exception:
        return "N/A"


def fmt_number(value: Any, decimals: int = 2) -> str:
    try:
        if value is None:
            return "N/A"
        if isinstance(value, bool):
            return "N/A"
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return "N/A"
        return f"{v:,.{decimals}f}"
    except Exception:
        return "N/A"


def fmt_pct(value: Any, decimals: int = 1) -> str:
    """
    Accepts either 0.1234 or 12.34 (unknown source).
    FD margins are typically 0.xx, YoY growth often 0.xx.
    We'll treat abs(value) <= 2 as ratio; else already percent.
    """
    try:
        if value is None:
            return "N/A"
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return "N/A"
        if abs(v) <= 2.0:
            v = v * 100.0
        return f"{v:.{decimals}f}%"
    except Exception:
        return "N/A"


def safe_str(x: Any, fallback: str = "N/A") -> str:
    s = ("" if x is None else str(x)).strip()
    return s if s else fallback


# ==========================================================
# FINANCIALDATASETS.AI (FD) HTTP HELPERS
# ==========================================================

def fd_headers() -> Dict[str, str]:
    if not FD_API_KEY:
        return {}
    # Their docs show X-API-KEY, but header names are case-insensitive.
    return {"X-API-KEY": FD_API_KEY}


def fd_get_json(path: str, params: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None) -> Tuple[Optional[Any], Optional[str], Optional[int], str]:
    """
    Returns: (json_data, error_str, status_code, final_url)
    - allow_redirects=True is required because FD endpoints often 301 to trailing slash.
    - This function never raises; it returns errors as strings.
    """
    if not FD_API_KEY:
        return None, "FINANCIAL_DATASETS_API_KEY missing.", None, ""

    url = f"{FD_BASE_URL}{path}"
    try:
        r = requests.get(
            url,
            headers=fd_headers(),
            params=params,
            timeout=timeout or REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        final_url = r.url
        status = r.status_code

        if status != 200:
            snippet = (r.text or "")[:250]
            dbg(f"FD GET {path} status={status} url={final_url} err={snippet}")
            return None, f"{status}: {snippet}", status, final_url

        # Sometimes upstream returns empty body
        if not r.text or not r.text.strip():
            dbg(f"FD GET {path} status=200 but empty body url={final_url}")
            return None, "200: empty response body", status, final_url

        data = r.json()
        dbg(f"FD GET {path} ok status=200 url={final_url}")
        return data, None, status, final_url

    except Exception as e:
        dbg(f"FD GET {path} exception: {e}")
        return None, str(e), None, url


# ==========================================================
# FD WRAPPERS (normalized)
# ==========================================================

def fetch_financial_metrics_snapshot(symbol: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    data, err, _, _ = fd_get_json("/financial-metrics/snapshot", {"ticker": symbol.upper()})
    if err or not data:
        return None, err or "No data"

    # Expected: { "snapshot": {...} }
    if isinstance(data, dict) and isinstance(data.get("snapshot"), dict):
        return data["snapshot"], None

    # Some variants might return snapshot object directly
    if isinstance(data, dict) and "ticker" in data:
        return data, None

    return None, "Unexpected snapshot format"


def fetch_financial_metrics_history(symbol: str, period: str = "annual", limit: int = 10) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    data, err, _, _ = fd_get_json("/financial-metrics", {"ticker": symbol.upper(), "period": period, "limit": limit})
    if err or not data:
        return None, err or "No data"

    if isinstance(data, list):
        return data, None

    if isinstance(data, dict):
        for key in ("metrics", "financial_metrics", "results"):
            if isinstance(data.get(key), list):
                return data[key], None
        # fallback single dict
        return [data], None

    return None, "Unexpected financial-metrics format"


def fetch_company_facts(symbol: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    data, err, _, _ = fd_get_json("/company/facts", {"ticker": symbol.upper()})
    if err or not data:
        return None, err or "No data"

    # Expected: { "company_facts": {...} }
    if isinstance(data, dict) and isinstance(data.get("company_facts"), dict):
        return data["company_facts"], None

    # fallback
    if isinstance(data, dict) and "ticker" in data:
        return data, None

    return None, "Unexpected company facts format"


def fetch_analyst_estimates(symbol: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    data, err, _, _ = fd_get_json("/analyst-estimates", {"ticker": symbol.upper(), "period": "annual"})
    if err or not data:
        return None, err or "No data"

    if isinstance(data, dict) and isinstance(data.get("analyst_estimates"), list):
        return data["analyst_estimates"], None

    return None, "Unexpected analyst estimates format"


def fetch_financials(symbol: str, period: str = "annual", limit: int = 10) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    data, err, _, _ = fd_get_json("/financials", {"ticker": symbol.upper(), "period": period, "limit": limit})
    if err or not data:
        return None, err or "No data"

    if isinstance(data, dict):
        # Expected: { "income_statements": [...] } possibly more keys
        if isinstance(data.get("income_statements"), list):
            return data, None
        if isinstance(data.get("financials"), dict) and isinstance(data["financials"].get("income_statements"), list):
            return data["financials"], None

    return None, "Unexpected financials format"


def fetch_news(symbol: str, limit: int = 5) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    data, err, _, _ = fd_get_json("/news", {"ticker": symbol.upper(), "limit": limit})
    if err or not data:
        return [], err or "No data"

    if isinstance(data, dict) and isinstance(data.get("news"), list):
        return data["news"], None

    return [], "Unexpected news format"


def fetch_insider_transactions(symbol: str, limit: int = 20) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    # FD endpoint name varies across accounts; this is the one that worked in your logs
    data, err, _, _ = fd_get_json("/insider-transactions", {"ticker": symbol.upper(), "limit": limit})
    if err or not data:
        # fallback to older endpoint name if needed
        data2, err2, _, _ = fd_get_json("/insider-trades", {"ticker": symbol.upper(), "limit": limit})
        if err2 or not data2:
            return [], err or err2 or "No data"
        if isinstance(data2, dict) and isinstance(data2.get("insider_trades"), list):
            return data2["insider_trades"], None
        return [], "Unexpected insider format (fallback)"

    if isinstance(data, dict):
        if isinstance(data.get("insider_transactions"), list):
            return data["insider_transactions"], None
        if isinstance(data.get("insider_trades"), list):
            return data["insider_trades"], None

    return [], "Unexpected insider format"


def fetch_institutional_ownership(symbol: str, limit: int = 200) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    data, err, _, _ = fd_get_json("/institutional-ownership", {"ticker": symbol.upper(), "limit": limit})
    if err or not data:
        return [], err or "No data"

    if isinstance(data, dict) and isinstance(data.get("institutional_ownership"), list):
        return data["institutional_ownership"], None

    return [], "Unexpected institutional format"


# ==========================================================
# PRICE PIPELINES
# ==========================================================

def _normalize_price_df(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Normalize to columns: close, volume, (optional: open/high/low)
    Index: tz-naive datetime (UTC removed for plotting)
    """
    try:
        if df is None or df.empty:
            return None, "Empty price dataframe"

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            return None, "Price DF missing DatetimeIndex"

        df = df.copy()

        # Column name normalization
        colmap = {c.lower(): c for c in df.columns}
        def pick(*names):
            for n in names:
                if n in colmap:
                    return colmap[n]
            return None

        close_c = pick("close", "adj close", "adj_close", "c")
        vol_c = pick("volume", "v")

        if close_c is None:
            return None, "Price DF missing close column"

        df["close"] = pd.to_numeric(df[close_c], errors="coerce")

        if vol_c is not None:
            df["volume"] = pd.to_numeric(df[vol_c], errors="coerce")
        else:
            df["volume"] = 0.0

        # Clean index tz
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        df = df.dropna(subset=["close"]).sort_index()
        if df.empty:
            return None, "All close values NaN"

        return df[["close", "volume"]], None
    except Exception as e:
        return None, str(e)


def fetch_price_history_fd(symbol: str, days: int = 400) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    FD /prices endpoint (PRIMARY).
    Your curl showed 301 -> /prices/ ... so allow_redirects=True in fd_get_json handles it.
    Expected:
      { "ticker": "FI", "prices": [ { "time": "...", "close": ..., "volume": ... }, ... ] }
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

    data, err, status, final_url = fd_get_json("/prices", params, timeout=REQUEST_TIMEOUT)
    if err or not data:
        dbg(f"FD prices failed for {symbol}: {err} (status={status} url={final_url})")
        return None, err or "No FD prices data"

    prices = None
    if isinstance(data, dict):
        prices = data.get("prices")

    if not isinstance(prices, list) or not prices:
        return None, "FD prices missing/empty 'prices' list"

    df = pd.DataFrame(prices)

    # Time col can be "time" or "date"
    if "time" in df.columns:
        df["time"] = df["time"].astype(str)
        df["time"] = df["time"].str.replace(" EDT", "", regex=False).str.replace(" EST", "", regex=False)
        dt = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df = df.assign(_dt=dt).dropna(subset=["_dt"]).set_index("_dt")
    elif "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df = df.assign(_dt=dt).dropna(subset=["_dt"]).set_index("_dt")
    else:
        return None, "FD prices missing 'time'/'date' field"

    # Standard columns
    # FD commonly uses lower-case: close, volume
    if "close" not in df.columns:
        return None, "FD prices missing 'close' column"
    if "volume" not in df.columns:
        df["volume"] = 0.0

    # normalize
    df.index = df.index.tz_convert("UTC").tz_localize(None)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["close"]).sort_index()
    if df.empty:
        return None, "FD prices: all close values NaN"

    return df[["close", "volume"]], None


def fetch_price_history_yf(symbol: str, days: int = 400) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    yfinance fallback (SECONDARY).
    Defensive against 429 + JSON parse errors.
    """
    if yf is None:
        return None, "yfinance not installed"

    # yfinance uses periods; approximate days
    period = "1y" if days <= 370 else "2y"

    # Light retry/backoff (but don't spam Yahoo)
    last_err = None
    for attempt in range(3):
        try:
            t = yf.Ticker(symbol.upper())
            # Avoid .info (quoteSummary) because it triggers 429 frequently
            hist = t.history(period=period, interval="1d")
            if hist is None or hist.empty:
                return None, "yfinance returned empty history"

            # normalize
            df = hist.copy()
            # yfinance cols: Open High Low Close Volume
            df = df.rename(columns={c: c.lower() for c in df.columns})
            if "adj close" in df.columns and "close" not in df.columns:
                df["close"] = df["adj close"]

            # limit to last N days
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.sort_index()
                if len(df) > days:
                    df = df.iloc[-days:]

            nd, nerr = _normalize_price_df(df)
            if nd is not None:
                return nd, None

            return None, nerr or "yfinance normalize failed"
        except Exception as e:
            last_err = str(e)
            # Exponential backoff (small)
            time.sleep(1.2 * (attempt + 1))

    return None, f"yfinance error: {last_err}"


def fetch_price_history_hybrid(symbol: str, days: int = 400) -> Tuple[Optional[pd.DataFrame], Optional[str], str]:
    """
    Hybrid: FD first, then yfinance fallback.
    Returns (df, err, source) where source is 'FD' or 'YF' or 'NONE'
    """
    df, err = fetch_price_history_fd(symbol, days=days)
    if df is not None and not df.empty:
        return df, None, "FD"

    df2, err2 = fetch_price_history_yf(symbol, days=days)
    if df2 is not None and not df2.empty:
        return df2, None, "YF"

    return None, err2 or err or "No price data available", "NONE"


# ==========================================================
# SNAPSHOT BUILDER (company facts + hybrid price)
# ==========================================================

def build_stock_snapshot(symbol: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (snapshot, meta_debug)
    snapshot keys:
      symbol, long_name, sector, industry, website
      current_price, day_change_pct, day_change_dollar
      year_low, year_high, change_1y_pct
    meta_debug includes price_source + any errors to embed (optional)
    """
    symbol = symbol.upper()
    meta: Dict[str, Any] = {"price_source": None, "errors": []}

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

    # Company facts (primary for name/sector/industry/website)
    facts, facts_err = fetch_company_facts(symbol)
    if facts:
        snapshot["long_name"] = facts.get("name") or facts.get("company_name") or symbol
        snapshot["sector"] = safe_str(facts.get("sector"), "N/A")
        snapshot["industry"] = safe_str(facts.get("industry"), "N/A")
        snapshot["website"] = safe_str(
            facts.get("website_url") or facts.get("website") or facts.get("homepage"),
            "N/A"
        )
    else:
        if facts_err:
            meta["errors"].append(f"company_facts: {facts_err}")

    # Hybrid pricing
    df, perr, source = fetch_price_history_hybrid(symbol, days=400)
    meta["price_source"] = source
    if df is None or df.empty:
        if perr:
            meta["errors"].append(f"prices: {perr}")
        return snapshot, meta

    close = df["close"].astype(float)
    snapshot["current_price"] = float(close.iloc[-1])

    # day change using last 2 closes
    if len(close) >= 2:
        prev = float(close.iloc[-2])
        if prev != 0:
            delta = snapshot["current_price"] - prev
            snapshot["day_change_dollar"] = delta
            snapshot["day_change_pct"] = (delta / prev) * 100.0

    # 52W range + 1y change
    # Use last 252 trading days if available
    close_252 = close.iloc[-252:] if len(close) >= 252 else close
    snapshot["year_low"] = float(close_252.min())
    snapshot["year_high"] = float(close_252.max())

    first = float(close_252.iloc[0])
    if first != 0:
        snapshot["change_1y_pct"] = ((snapshot["current_price"] - first) / first) * 100.0

    return snapshot, meta


# ==========================================================
# AI PROMPTS (analysis-only, include ticker + company name)
# ==========================================================

AI_SYSTEM_NO_GREETING = (
    "You are a senior equity analyst. "
    "Return ONLY the analysis content. "
    "No greetings, no pleasantries, no introductions, no side conversations. "
    "No markdown headings like ### and no **bold**."
)

def _display_ticker_name(symbol: str, company_name: str) -> str:
    # Ensures we pass both identifiers to reduce exchange ambiguity
    company_name = safe_str(company_name, symbol)
    return f"{symbol} ({company_name})"


def build_ai_single_prompt(
    symbol: str,
    snapshot: Dict[str, Any],
    fm_snapshot: Optional[Dict[str, Any]],
    analyst_estimates: Optional[List[Dict[str, Any]]],
    company_facts: Optional[Dict[str, Any]],
) -> str:
    company_name = safe_str(snapshot.get("long_name"), symbol)
    display = _display_ticker_name(symbol, company_name)
    cik = (company_facts or {}).get("cik", "")

    return f"""
Evaluate the stock {display}. Use the provided data. If a field is missing, say "Not available" and continue.

Write a structured investment summary with these sections:
1. Company Overview
2. Stock Performance (1Y, day change, key levels)
3. Valuation (use P/E, P/B, P/S, EV/EBITDA, EV/Sales where available)
4. Growth & Profitability (margins, ROE, etc. if present)
5. Analyst Estimates & Expectations
6. Key Risks
7. Final Verdict (Buy / Hold / Avoid — clearly NOT investment advice)

Useful Links:
- Google News: https://news.google.com/search?q={symbol}+stock
- Yahoo Finance: https://finance.yahoo.com/quote/{symbol}
- SEC Filings: https://www.sec.gov/edgar/browse/?CIK={cik}
- MarketWatch: https://www.marketwatch.com/investing/stock/{symbol}

Data:
SNAPSHOT:
{json.dumps(snapshot, indent=2)}

FINANCIAL METRICS SNAPSHOT:
{json.dumps(fm_snapshot, indent=2)}

ANALYST ESTIMATES:
{json.dumps(analyst_estimates, indent=2)}

COMPANY FACTS:
{json.dumps(company_facts, indent=2)}
""".strip()


def call_openai_chat(prompt: str, temperature: float = 0.3) -> str:
    if not OPENAI_API_KEY or OpenAI is None:
        return "OpenAI key missing or OpenAI library not installed; AI section unavailable."

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        res = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            temperature=temperature,
            messages=[
                {"role": "system", "content": AI_SYSTEM_NO_GREETING},
                {"role": "user", "content": prompt},
            ],
        )
        return res.choices[0].message.content or ""
    except Exception as e:
        return f"AI Error: {e}\n{traceback.format_exc()}"


def generate_ai_fundamental_single(
    symbol: str,
    snapshot: Dict[str, Any],
    fm_snapshot: Optional[Dict[str, Any]],
    analyst_estimates: Optional[List[Dict[str, Any]]],
    company_facts: Optional[Dict[str, Any]],
) -> str:
    prompt = build_ai_single_prompt(symbol, snapshot, fm_snapshot, analyst_estimates, company_facts)
    return call_openai_chat(prompt, temperature=0.3)


def generate_ai_freelancing_single(symbol: str, company_name: str) -> str:
    display = _display_ticker_name(symbol, company_name)
    prompt = (
        f"Tell me all I should know about {display} as an informed long-term investor. "
        "Cover business model, segments, strategy, competitive landscape, valuation, risks, catalysts, "
        "and scenario analysis. If uncertain, say so. Return analysis only."
    )
    return call_openai_chat(prompt, temperature=0.4)


def generate_ai_combined_pair(
    s1: str, snap1: Dict[str, Any], fm1: Optional[Dict[str, Any]], analyst1: Optional[List[Dict[str, Any]]], facts1: Optional[Dict[str, Any]],
    s2: str, snap2: Dict[str, Any], fm2: Optional[Dict[str, Any]], analyst2: Optional[List[Dict[str, Any]]], facts2: Optional[Dict[str, Any]],
) -> str:
    display1 = _display_ticker_name(s1, safe_str(snap1.get("long_name"), s1))
    display2 = _display_ticker_name(s2, safe_str(snap2.get("long_name"), s2))

    payload = {
        "stock_1": {"display": display1, "snapshot": snap1, "metrics_snapshot": fm1, "analyst_estimates": analyst1, "company_facts": facts1},
        "stock_2": {"display": display2, "snapshot": snap2, "metrics_snapshot": fm2, "analyst_estimates": analyst2, "company_facts": facts2},
    }

    prompt = f"""
Compare two stocks: {display1} and {display2}.
Return analysis only. No greetings. No markdown headings (###) and no bold (**).

Required sections:
1. Business Overview & Competitive Position
2. Stock Performance & Momentum
3. Valuation Comparison (P/E, P/B, P/S, EV/EBITDA, EV/Sales where available)
4. Growth, Profitability & Balance Sheet Quality
5. Key Risks & Downside Scenarios
6. Key Catalysts & Upside Scenarios
7. Overall Assessment (clearly labeled as NOT investment advice)

Data:
{json.dumps(payload, indent=2)}
""".strip()

    return call_openai_chat(prompt, temperature=0.3)


# ==========================================================
# TABLE DATA BUILDERS
# ==========================================================

def build_multi_year_fundamentals_rows(financials: Optional[Dict[str, Any]]) -> Tuple[List[List[Any]], Optional[str]]:
    """
    Returns table rows including header row.
    """
    header = ["Year", "Sales", "GP", "OpInc", "NetInc", "EPS", "GP%", "OP%", "NI%", "SlsYoY", "EPSYoY"]

    if not financials:
        return [header, ["N/A"] * len(header)], "No financials available."

    inc_list = financials.get("income_statements") or []
    if not inc_list:
        return [header, ["N/A"] * len(header)], "No income statements returned."

    rows: List[Dict[str, Any]] = []
    for item in inc_list:
        if item.get("period") != "annual":
            continue

        rp = item.get("report_period") or item.get("fiscal_period")
        if not rp:
            continue
        year = str(rp)[:4]

        sales = item.get("revenue")
        gp = item.get("gross_profit")
        op = item.get("operating_income") or item.get("ebit")
        ni = item.get("net_income")
        eps = item.get("earnings_per_share")

        rows.append({"year": year, "sales": sales, "gp": gp, "op": op, "ni": ni, "eps": eps})

    if not rows:
        return [header, ["N/A"] * len(header)], "No annual income statements found."

    rows.sort(key=lambda r: r["year"])

    def pct(num, den):
        try:
            if num is None or den in (None, 0):
                return None
            return (float(num) / float(den)) * 100.0
        except Exception:
            return None

    prev_sales = None
    prev_eps = None
    for r in rows:
        s = r["sales"]
        r["gp_margin"] = pct(r["gp"], s)
        r["op_margin"] = pct(r["op"], s)
        r["ni_margin"] = pct(r["ni"], s)

        if prev_sales not in (None, 0) and s not in (None, 0):
            r["sales_yoy"] = pct(float(s) - float(prev_sales), prev_sales)
        else:
            r["sales_yoy"] = None

        if prev_eps not in (None, 0) and r["eps"] not in (None, 0):
            r["eps_yoy"] = pct(float(r["eps"]) - float(prev_eps), prev_eps)
        else:
            r["eps_yoy"] = None

        prev_sales = s
        prev_eps = r["eps"]

    # last 5
    if len(rows) > 5:
        rows = rows[-5:]

    table: List[List[Any]] = [header]
    for r in rows:
        table.append([
            r["year"],
            fmt_int(r["sales"]),
            fmt_int(r["gp"]),
            fmt_int(r["op"]),
            fmt_int(r["ni"]),
            fmt_number(r["eps"], 2),
            (fmt_pct(r["gp_margin"], 1) if r["gp_margin"] is not None else "N/A"),
            (fmt_pct(r["op_margin"], 1) if r["op_margin"] is not None else "N/A"),
            (fmt_pct(r["ni_margin"], 1) if r["ni_margin"] is not None else "N/A"),
            (fmt_pct(r["sales_yoy"], 1) if r["sales_yoy"] is not None else "N/A"),
            (fmt_pct(r["eps_yoy"], 1) if r["eps_yoy"] is not None else "N/A"),
        ])

    return table, None

def build_single_report_data(symbol: str) -> Dict[str, Any]:
    snapshot, snap_meta = build_stock_snapshot(symbol)

    fm_snapshot, _ = fetch_financial_metrics_snapshot(symbol)
    fm_history, _ = fetch_financial_metrics_history(symbol, "annual", 10)
    analyst, _ = fetch_analyst_estimates(symbol)
    facts, _ = fetch_company_facts(symbol)
    financials, _ = fetch_financials(symbol, "annual", 10)
    news, _ = fetch_news(symbol, 5)
    insider, _ = fetch_insider_transactions(symbol, 20)
    inst, _ = fetch_institutional_ownership(symbol, 200)

    fundamentals_table, _ = build_multi_year_fundamentals_rows(financials)
    metrics_table = build_financial_metrics_rows(fm_snapshot)
    analyst_table = build_analyst_estimates_rows(analyst)
    insider_table = build_insider_rows(insider, 10)
    inst_table = build_institutional_rows(inst, 10)

    return {
        "symbol": symbol,
        "snapshot": snapshot,
        "snapshot_meta": snap_meta,
        "fundamentals": fundamentals_table,
        "metrics": metrics_table,
        "analyst_estimates": analyst_table,
        "insiders": insider_table,
        "institutional": inst_table,
        "news": news,
        "charts": {
            "price_source": snap_meta.get("price_source")
        }
    }

def build_financial_metrics_rows(fm: Optional[Dict[str, Any]]) -> List[List[Any]]:
    """
    Grid-style key/value table. Returns rows with [Metric, Value]
    """
    rows = [["Metric", "Value"]]
    if not fm:
        rows.append(["Financial Metrics Snapshot", "N/A"])
        return rows

    # Ordered list of (label, key, formatter)
    items = [
        ("Market Cap", "market_cap", lambda v: fmt_int(v)),
        ("Enterprise Value", "enterprise_value", lambda v: fmt_int(v)),
        ("P/E Ratio", "price_to_earnings_ratio", lambda v: fmt_number(v, 4)),
        ("P/B Ratio", "price_to_book_ratio", lambda v: fmt_number(v, 4)),
        ("P/S Ratio", "price_to_sales_ratio", lambda v: fmt_number(v, 4)),
        ("EV/EBITDA", "enterprise_value_to_ebitda_ratio", lambda v: fmt_number(v, 4)),
        ("EV/Sales", "enterprise_value_to_revenue_ratio", lambda v: fmt_number(v, 4)),
        ("Free Cash Flow Yield", "free_cash_flow_yield", lambda v: fmt_pct(v, 2)),
        ("PEG Ratio", "peg_ratio", lambda v: fmt_number(v, 4)),
        ("Gross Margin", "gross_margin", lambda v: fmt_pct(v, 2)),
        ("Operating Margin", "operating_margin", lambda v: fmt_pct(v, 2)),
        ("Net Margin", "net_margin", lambda v: fmt_pct(v, 2)),
        ("Return on Equity", "return_on_equity", lambda v: fmt_pct(v, 2)),
        ("Return on Assets", "return_on_assets", lambda v: fmt_pct(v, 2)),
        ("Return on Invested Capital", "return_on_invested_capital", lambda v: fmt_pct(v, 2)),
        ("Current Ratio", "current_ratio", lambda v: fmt_number(v, 4)),
        ("Quick Ratio", "quick_ratio", lambda v: fmt_number(v, 4)),
        ("Cash Ratio", "cash_ratio", lambda v: fmt_number(v, 4)),
        ("Debt to Equity", "debt_to_equity", lambda v: fmt_number(v, 4)),
        ("Debt to Assets", "debt_to_assets", lambda v: fmt_number(v, 4)),
        ("Interest Coverage", "interest_coverage", lambda v: fmt_number(v, 4)),
        ("Revenue Growth", "revenue_growth", lambda v: fmt_pct(v, 2)),
        ("Earnings Growth", "earnings_growth", lambda v: fmt_pct(v, 2)),
        ("Book Value Growth", "book_value_growth", lambda v: fmt_pct(v, 2)),
        ("EPS Growth", "earnings_per_share_growth", lambda v: fmt_pct(v, 2)),
        ("Free Cash Flow Growth", "free_cash_flow_growth", lambda v: fmt_pct(v, 2)),
        ("Operating Income Growth", "operating_income_growth", lambda v: fmt_pct(v, 2)),
        ("EBITDA Growth", "ebitda_growth", lambda v: fmt_pct(v, 2)),
        ("Payout Ratio", "payout_ratio", lambda v: fmt_pct(v, 2)),
        ("Earnings Per Share", "earnings_per_share", lambda v: fmt_number(v, 4)),
        ("Book Value Per Share", "book_value_per_share", lambda v: fmt_number(v, 4)),
        ("Free Cash Flow Per Share", "free_cash_flow_per_share", lambda v: fmt_number(v, 4)),
    ]

    for label, key, f in items:
        rows.append([label, f(fm.get(key))])

    return rows


def build_analyst_estimates_rows(est: Optional[List[Dict[str, Any]]]) -> List[List[Any]]:
    rows = [["Fiscal Period", "Period", "EPS Estimate"]]
    if not est:
        rows.append(["N/A", "N/A", "N/A"])
        return rows

    for e in est:
        rows.append([
            safe_str(e.get("fiscal_period")),
            safe_str(e.get("period")),
            fmt_number(e.get("earnings_per_share"), 4),
        ])
    return rows


def build_insider_rows(insiders: List[Dict[str, Any]], max_rows: int = 10) -> List[List[Any]]:
    rows = [["Date", "Name", "Title", "Shares", "Value"]]
    if not insiders:
        rows.append(["N/A", "N/A", "N/A", "N/A", "N/A"])
        return rows

    for t in insiders[:max_rows]:
        rows.append([
            safe_str(t.get("transaction_date") or t.get("date")),
            safe_str(t.get("name")),
            safe_str(t.get("title")),
            fmt_int(t.get("transaction_shares") or t.get("shares")),
            fmt_int(t.get("transaction_value") or t.get("value")),
        ])
    return rows


def build_institutional_rows(inst: List[Dict[str, Any]], max_rows: int = 10) -> List[List[Any]]:
    rows = [["Investor", "Shares"]]
    if not inst:
        rows.append(["N/A", "N/A"])
        return rows

    # sort by shares desc
    sorted_inst = sorted(inst, key=lambda x: (x.get("shares") or 0), reverse=True)[:max_rows]
    for h in sorted_inst:
        rows.append([safe_str(h.get("investor")), fmt_int(h.get("shares"))])
    return rows


# ==========================================================
# CHARTS (matplotlib)
# ==========================================================

def build_single_charts(symbol: str, fm_history: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    df, err, src = fetch_price_history_hybrid(symbol, days=365)
    if df is None or df.empty:
        dbg(f"{symbol}: charts skipped (no price data). err={err} src={src}")
        return None

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    hist = pd.DataFrame({"Close": close, "Volume": volume})
    hist["MA20"] = hist["Close"].rolling(20).mean()
    hist["MA50"] = hist["Close"].rolling(50).mean()
    hist["MA200"] = hist["Close"].rolling(200).mean()

    # RSI
    delta = hist["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # MACD
    ema12 = hist["Close"].ewm(span=12, adjust=False).mean()
    ema26 = hist["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    fig, axs = plt.subplots(5, 1, figsize=(8.5, 11), sharex=False)
    fig.subplots_adjust(hspace=0.45)
    dates = hist.index

    # Price + MAs
    ax = axs[0]
    ax.plot(dates, hist["Close"], label="Close")
    ax.plot(dates, hist["MA20"], label="MA20", linewidth=0.8)
    ax.plot(dates, hist["MA50"], label="MA50", linewidth=0.8)
    ax.plot(dates, hist["MA200"], label="MA200", linewidth=0.8)
    ax.set_title(f"{symbol} Price + Moving Averages")
    ax.legend(loc="upper left", fontsize=7)

    # Volume
    ax = axs[1]
    ax.bar(dates, hist["Volume"] / 1_000_000.0, width=1.0)
    ax.set_title("Daily Volume (Millions)")
    ax.set_ylabel("Shares (M)")

    # RSI
    ax = axs[2]
    ax.plot(dates, rsi)
    ax.axhline(70, linestyle="--", linewidth=0.8)
    ax.axhline(30, linestyle="--", linewidth=0.8)
    ax.set_title("RSI (14)")
    ax.set_ylim(0, 100)

    # MACD
    ax = axs[3]
    ax.plot(dates, macd, label="MACD")
    ax.plot(dates, signal, label="Signal", linestyle="--")
    ax.axhline(0, linewidth=0.8)
    ax.legend(fontsize=7)
    ax.set_title("MACD (12/26/9)")

    # Valuation history (if available)
    ax = axs[4]
    if fm_history:
        records = []
        for item in fm_history:
            lbl = item.get("report_period") or item.get("fiscal_period") or item.get("date")
            pe = item.get("price_to_earnings_ratio")
            ev_ebitda = item.get("enterprise_value_to_ebitda_ratio")
            ev_sales = item.get("enterprise_value_to_revenue_ratio")
            if lbl and (pe is not None or ev_ebitda is not None or ev_sales is not None):
                records.append((str(lbl), pe, ev_ebitda, ev_sales))

        if records:
            records.sort(key=lambda x: x[0])
            labels = [r[0] for r in records]
            x = list(range(len(labels)))
            pe_vals = [r[1] for r in records]
            ev_e_vals = [r[2] for r in records]
            ev_s_vals = [r[3] for r in records]

            if any(v is not None for v in pe_vals):
                ax.plot(x, pe_vals, marker="o", label="P/E")
            if any(v is not None for v in ev_e_vals):
                ax.plot(x, ev_e_vals, marker="o", label="EV/EBITDA")
            if any(v is not None for v in ev_s_vals):
                ax.plot(x, ev_s_vals, marker="o", label="EV/Sales")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, fontsize=7)
            ax.set_title("Valuation Over Time (Annual)")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "No valuation history available.", ha="center", va="center")
            ax.axis("off")
    else:
        ax.text(0.5, 0.5, "No valuation history available.", ha="center", va="center")
        ax.axis("off")

    out_dir = os.path.join(BASE_DIR, "generated_reports")
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, f"single_{symbol}_charts.png")
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return img_path


def build_compare_charts(s1: str, s2: str) -> Optional[str]:
    df1, err1, src1 = fetch_price_history_hybrid(s1, days=365)
    df2, err2, src2 = fetch_price_history_hybrid(s2, days=365)
    if df1 is None or df2 is None or df1.empty or df2.empty:
        dbg(f"compare charts skipped. {s1}({src1}) err={err1} | {s2}({src2}) err={err2}")
        return None

    c1 = df1["close"].astype(float)
    c2 = df2["close"].astype(float)

    fig, axs = plt.subplots(3, 1, figsize=(8.5, 11), sharex=False)
    fig.subplots_adjust(hspace=0.4)

    ax = axs[0]
    ax.plot(c1.index, c1, label=f"{s1} Close")
    ax.plot(c2.index, c2, label=f"{s2} Close")
    ax.set_title("Price History (1Y)")
    ax.legend(fontsize=8)

    ax = axs[1]
    base1 = float(c1.iloc[0])
    base2 = float(c2.iloc[0])
    ax.plot(c1.index, (c1 / base1) * 100.0, label=f"{s1} Indexed")
    ax.plot(c2.index, (c2 / base2) * 100.0, label=f"{s2} Indexed")
    ax.set_title("Indexed Performance (Start=100)")
    ax.legend(fontsize=8)

    fm1, _ = fetch_financial_metrics_history(s1, "annual", 10)
    fm2, _ = fetch_financial_metrics_history(s2, "annual", 10)

    ax = axs[2]
    recs1: List[Tuple[str, float]] = []
    recs2: List[Tuple[str, float]] = []

    if fm1:
        for item in fm1:
            lbl = item.get("report_period") or item.get("fiscal_period") or item.get("date")
            pe = item.get("price_to_earnings_ratio")
            if lbl and pe is not None:
                recs1.append((str(lbl), float(pe)))
    if fm2:
        for item in fm2:
            lbl = item.get("report_period") or item.get("fiscal_period") or item.get("date")
            pe = item.get("price_to_earnings_ratio")
            if lbl and pe is not None:
                recs2.append((str(lbl), float(pe)))

    if recs1 or recs2:
        if recs1:
            recs1.sort(key=lambda x: x[0])
            ax.plot(list(range(len(recs1))), [x[1] for x in recs1], marker="o", label=f"{s1} P/E")
        if recs2:
            recs2.sort(key=lambda x: x[0])
            ax.plot(list(range(len(recs2))), [x[1] for x in recs2], marker="o", label=f"{s2} P/E")
        ax.set_title("P/E Over Time (Annual)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No P/E history available.", ha="center", va="center")
        ax.axis("off")

    out_dir = os.path.join(BASE_DIR, "generated_reports")
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, f"compare_{s1}_{s2}_charts.png")
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return img_path


# ==========================================================
# PDF RENDERING (modern grids)
# ==========================================================

def _table_style() -> TableStyle:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),   # header background
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ALIGN", (0, 0), (-1, 0), "LEFT"),

        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#111827")),

        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])


def _make_table(data: List[List[Any]], col_widths: Optional[List[float]] = None) -> Table:
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(_table_style())
    return t


def export_pdf(
    *,
    title_line: str,
    sections: List[Any],
    chart_path: Optional[str],
    output_path: str,
) -> None:
    PAGE_WIDTH, PAGE_HEIGHT = letter

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name="Title",
        fontName="Helvetica-Bold",
        fontSize=22,
        alignment=TA_CENTER,
        spaceAfter=22,
        textColor=colors.HexColor("#111827")
    )

    subtitle_style = ParagraphStyle(
        name="Subtitle",
        fontName="Helvetica",
        fontSize=11,
        alignment=TA_CENTER,
        spaceAfter=26,
        textColor=colors.HexColor("#374151")
    )

    h_style = ParagraphStyle(
        name="Header",
        fontName="Helvetica-Bold",
        fontSize=12,
        spaceBefore=14,
        spaceAfter=8,
        textColor=colors.HexColor("#111827")
    )

    body_style = ParagraphStyle(
        name="Body",
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#111827")
    )

    small_style = ParagraphStyle(
        name="Small",
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#374151")
    )

    def header(canvas, doc):
        if canvas.getPageNumber() == 1:
            return
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(colors.HexColor("#111827"))
        canvas.drawString(50, PAGE_HEIGHT - 36, title_line)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.drawRightString(PAGE_WIDTH - 50, 32, f"Page {canvas.getPageNumber()}")

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

    story: List[Any] = []

    # Title page
    story.append(Spacer(1, 2.0 * inch))
    story.append(Paragraph("Stock Analyzer", title_style))
    story.append(Paragraph(title_line, subtitle_style))
    story.append(Paragraph("Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), subtitle_style))
    story.append(PageBreak())

    # Render sections
    for sec in sections:
        kind = sec.get("kind")

        if kind == "header":
            story.append(Paragraph(sec["text"], h_style))

        elif kind == "paragraph":
            text = sec.get("text", "")
            # escape minimal HTML
            safe = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            # Keep long AI blocks readable
            for line in safe.split("\n"):
                line = line.strip()
                if not line:
                    story.append(Spacer(1, 6))
                else:
                    story.append(Paragraph(line, body_style))

        elif kind == "table":
            title = sec.get("title")
            if title:
                story.append(Paragraph(title, h_style))
            story.append(KeepTogether([
                _make_table(sec["data"], sec.get("col_widths")),
                Spacer(1, 10)
            ]))

        elif kind == "spacer":
            story.append(Spacer(1, sec.get("h", 10)))

        elif kind == "pagebreak":
            story.append(PageBreak())

    if chart_path and os.path.exists(chart_path):
        story.append(PageBreak())
        story.append(Paragraph("Charts & Visuals", h_style))
        story.append(Spacer(1, 10))
        img = RLImage(chart_path)
        img._restrictSize(6.7 * inch, 9.0 * inch)
        story.append(img)

    doc.build(story)


# ==========================================================
# PUBLIC ENTRY POINTS
# ==========================================================

def run_single_to_pdf(symbol: str, out_dir: str) -> str:
    symbol = symbol.upper()
    os.makedirs(out_dir, exist_ok=True)

    # ---------- BUILD STRUCTURED REPORT ----------
    report = build_single_report_data(symbol)

    snapshot = report["snapshot"]
    metrics = report["metrics"]
    fundamentals = report["fundamentals"]
    analyst = report["analyst_estimates"]
    insiders = report["insiders"]
    institutional = report["institutional"]
    news = report["news"]

    # ---------- BUILD TEXT FOR PDF ----------
    lines: List[str] = []

    lines.append("=" * 72)
    lines.append(f"STOCK SNAPSHOT: {symbol}")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Name     : {snapshot.get('long_name')}")
    lines.append(f"Sector   : {snapshot.get('sector')}")
    lines.append(f"Industry : {snapshot.get('industry')}")
    lines.append(f"Website  : {snapshot.get('website')}")
    lines.append("")

    lines.append("PRICE")
    lines.append("-" * 72)
    lines.append(f"Current Price : {fmt_number(snapshot.get('current_price'))}")
    lines.append(f"Day Change %  : {fmt_number(snapshot.get('day_change_pct'))}%")
    lines.append(f"Day Change $  : {fmt_number(snapshot.get('day_change_dollar'))}")
    lines.append(f"52W Low       : {fmt_number(snapshot.get('year_low'))}")
    lines.append(f"52W High      : {fmt_number(snapshot.get('year_high'))}")
    lines.append(f"1Y Change %   : {fmt_number(snapshot.get('change_1y_pct'))}%")
    lines.append("")

    # ---------- FUNDAMENTALS ----------
    lines.append("MULTI-YEAR FUNDAMENTALS (Annual)")
    lines.append("-" * 72)
    for row in fundamentals:
        lines.append(" | ".join(row))
    lines.append("")

    # ---------- FINANCIAL METRICS ----------
    lines.append("FINANCIAL METRICS SNAPSHOT")
    lines.append("-" * 72)
    for label, value in metrics:
        lines.append(f"{label:<30}: {value}")
    lines.append("")

    # ---------- ANALYST ESTIMATES ----------
    lines.append("ANALYST ESTIMATES")
    lines.append("-" * 72)
    for row in analyst:
        lines.append(" | ".join(row))
    lines.append("")

    # ---------- INSIDERS ----------
    lines.append("INSIDER TRANSACTIONS")
    lines.append("-" * 72)
    for row in insiders:
        lines.append(" | ".join(row))
    lines.append("")

    # ---------- INSTITUTIONAL ----------
    lines.append("INSTITUTIONAL OWNERSHIP")
    lines.append("-" * 72)
    for row in institutional:
        lines.append(" | ".join(row))
    lines.append("")

    # ---------- NEWS ----------
    lines.append("LATEST NEWS")
    lines.append("-" * 72)
    for n in news:
        lines.append(f"{n.get('date')} - {n.get('title')}")
        lines.append(n.get("url", ""))
        lines.append("")

    # ---------- CHART ----------
    chart_path = build_single_charts(symbol, None)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(out_dir, f"{symbol}_{ts}.pdf")

    export_pdf(
        "\n".join(lines),
        f"{symbol} – {snapshot.get('long_name')}",
        chart_path,
        out_file
    )

    return out_file

def run_compare_to_pdf(s1: str, s2: str, out_dir: str) -> str:
    """
    Generates comparison PDF report. Returns file path.
    """
    s1 = s1.upper().strip()
    s2 = s2.upper().strip()
    os.makedirs(out_dir, exist_ok=True)

    dbg(f"run_compare_to_pdf s1={s1} s2={s2}")

    snap1, meta1 = build_stock_snapshot(s1)
    snap2, meta2 = build_stock_snapshot(s2)

    fm1, _ = fetch_financial_metrics_snapshot(s1)
    fm2, _ = fetch_financial_metrics_snapshot(s2)

    analyst1, _ = fetch_analyst_estimates(s1)
    analyst2, _ = fetch_analyst_estimates(s2)

    facts1, _ = fetch_company_facts(s1)
    facts2, _ = fetch_company_facts(s2)

    inst1, _ = fetch_institutional_ownership(s1, 200)
    inst2, _ = fetch_institutional_ownership(s2, 200)

    insider1, _ = fetch_insider_transactions(s1, 20)
    insider2, _ = fetch_insider_transactions(s2, 20)

    news1, _ = fetch_news(s1, 5)
    news2, _ = fetch_news(s2, 5)

    # AI comparison
    ai_cmp = generate_ai_combined_pair(s1, snap1, fm1, analyst1, facts1, s2, snap2, fm2, analyst2, facts2)

    sections: List[Dict[str, Any]] = []
    sections.append({"kind": "header", "text": f"COMPARISON: {s1} vs {s2}"})

    # Basic info comparison table
    comp_info = [
        ["Field", s1, s2],
        ["Name", safe_str(snap1.get("long_name")), safe_str(snap2.get("long_name"))],
        ["Sector", safe_str(snap1.get("sector")), safe_str(snap2.get("sector"))],
        ["Industry", safe_str(snap1.get("industry")), safe_str(snap2.get("industry"))],
        ["Website", safe_str(snap1.get("website")), safe_str(snap2.get("website"))],
        ["Price Source", safe_str(meta1.get("price_source"), "N/A"), safe_str(meta2.get("price_source"), "N/A")],
    ]
    sections.append({"kind": "table", "title": "Basic Info", "data": comp_info, "col_widths": [140, 170, 170]})

    # Price/performance comparison table
    comp_price = [
        ["Metric", s1, s2],
        ["Current Price", fmt_number(snap1.get("current_price"), 2), fmt_number(snap2.get("current_price"), 2)],
        ["Day Change (%)", fmt_pct(snap1.get("day_change_pct"), 2), fmt_pct(snap2.get("day_change_pct"), 2)],
        ["Day Change ($)", fmt_number(snap1.get("day_change_dollar"), 2), fmt_number(snap2.get("day_change_dollar"), 2)],
        ["52W Low", fmt_number(snap1.get("year_low"), 2), fmt_number(snap2.get("year_low"), 2)],
        ["52W High", fmt_number(snap1.get("year_high"), 2), fmt_number(snap2.get("year_high"), 2)],
        ["1Y Change (%)", fmt_pct(snap1.get("change_1y_pct"), 2), fmt_pct(snap2.get("change_1y_pct"), 2)],
    ]
    sections.append({"kind": "table", "title": "Price & Performance", "data": comp_price, "col_widths": [140, 170, 170]})

    # AI
    sections.append({"kind": "header", "text": "AI COMPARISON SUMMARY"})
    sections.append({"kind": "paragraph", "text": ai_cmp})

    # Metrics snapshot side-by-side (compact table)
    def _metrics_subset(fm: Optional[Dict[str, Any]]) -> Dict[str, str]:
        if not fm:
            return {}
        keys = [
            ("Market Cap", "market_cap", lambda v: fmt_int(v)),
            ("Enterprise Value", "enterprise_value", lambda v: fmt_int(v)),
            ("P/E", "price_to_earnings_ratio", lambda v: fmt_number(v, 3)),
            ("P/B", "price_to_book_ratio", lambda v: fmt_number(v, 3)),
            ("P/S", "price_to_sales_ratio", lambda v: fmt_number(v, 3)),
            ("EV/EBITDA", "enterprise_value_to_ebitda_ratio", lambda v: fmt_number(v, 3)),
            ("EV/Sales", "enterprise_value_to_revenue_ratio", lambda v: fmt_number(v, 3)),
            ("Gross Margin", "gross_margin", lambda v: fmt_pct(v, 2)),
            ("Op Margin", "operating_margin", lambda v: fmt_pct(v, 2)),
            ("Net Margin", "net_margin", lambda v: fmt_pct(v, 2)),
        ]
        out = {}
        for label, key, f in keys:
            out[label] = f(fm.get(key))
        return out

    m1 = _metrics_subset(fm1)
    m2 = _metrics_subset(fm2)
    labels = list(dict.fromkeys(list(m1.keys()) + list(m2.keys())))

    metrics_comp = [["Metric", s1, s2]]
    for lab in labels:
        metrics_comp.append([lab, m1.get(lab, "N/A"), m2.get(lab, "N/A")])

    sections.append({"kind": "table", "title": "Key Financial Metrics (Snapshot)", "data": metrics_comp, "col_widths": [180, 150, 150]})

    # Insider/Institutional (top 5 each as tables)
    sections.append({"kind": "table", "title": f"Insider Transactions (Top 5) — {s1}", "data": build_insider_rows(insider1, 5)})
    sections.append({"kind": "table", "title": f"Insider Transactions (Top 5) — {s2}", "data": build_insider_rows(insider2, 5)})

    sections.append({"kind": "table", "title": f"Institutional Ownership (Top 10) — {s1}", "data": build_institutional_rows(inst1, 10), "col_widths": [330, 150]})
    sections.append({"kind": "table", "title": f"Institutional Ownership (Top 10) — {s2}", "data": build_institutional_rows(inst2, 10), "col_widths": [330, 150]})

    # News
    sections.append({"kind": "header", "text": "Latest News"})
    def add_news_block(sym: str, items: List[Dict[str, Any]]):
        sections.append({"kind": "header", "text": sym})
        if not items:
            sections.append({"kind": "paragraph", "text": "No news available."})
            return
        for n in items:
            sections.append({"kind": "paragraph", "text": f"{safe_str(n.get('date'))} — {safe_str(n.get('title'))}\n{safe_str(n.get('url'))}"})

    add_news_block(s1, news1)
    add_news_block(s2, news2)

    # Charts
    chart_path = build_compare_charts(s1, s2)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(out_dir, f"{s1}_{s2}_{ts}.pdf")
    title_line = f"{s1} vs {s2}"

    export_pdf(
        title_line=title_line,
        sections=sections,
        chart_path=chart_path,
        output_path=out_file,
    )

    return out_file



