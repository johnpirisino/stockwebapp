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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Frame, PageTemplate, Image
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch

# =========================================================
# ENV / CONFIG
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FD_API_KEY = os.getenv("FINANCIAL_DATASETS_API_KEY")
FD_BASE_URL = "https://api.financialdatasets.ai"

DEBUG = os.getenv("DEBUG_ENGINE", "N").upper() == "Y"

def dbg(msg: str):
    if DEBUG:
        print(msg, flush=True)

# =========================================================
# HELPERS
# =========================================================

def fmt_number(v, d=2):
    try:
        return "N/A" if v is None else f"{float(v):,.{d}f}"
    except Exception:
        return "N/A"

def fmt_int(v):
    try:
        return "N/A" if v is None else f"{int(v):,}"
    except Exception:
        return "N/A"

def fd_headers():
    return {"X-API-Key": FD_API_KEY} if FD_API_KEY else {}

def fd_get_json(path, params=None):
    if not FD_API_KEY:
        return None, "Missing FINANCIAL_DATASETS_API_KEY"
    try:
        r = requests.get(
            f"{FD_BASE_URL}{path}",
            headers=fd_headers(),
            params=params,
            timeout=60,
            allow_redirects=True
        )
        dbg(f"FD {path} -> {r.status_code}")
        if r.status_code != 200:
            return None, r.text[:200]
        return r.json(), None
    except Exception as e:
        return None, str(e)

# =========================================================
# AI PROMPTS (PATCHED)
# =========================================================

def build_ai_single_prompt(symbol, snapshot, fm_snapshot, analyst, facts):
    company_name = snapshot.get("long_name", symbol)
    cik = (facts or {}).get("cik", "")

    return f"""
You are a senior equity analyst.

STRICT RULES:
- Return analysis only
- No greetings
- No disclaimers
- No conversational filler
- No markdown decorations

Analyze:
Ticker: {symbol}
Company: {company_name}

Provide the following sections ONLY:

1. Company Overview
2. Stock Performance
3. Valuation
4. Growth & Profitability
5. Analyst Expectations
6. Key Risks
7. Final Assessment (NOT investment advice)

DATA:
Snapshot:
{json.dumps(snapshot, indent=2)}

Financial Metrics:
{json.dumps(fm_snapshot, indent=2)}

Analyst Estimates:
{json.dumps(analyst, indent=2)}

Company Facts:
{json.dumps(facts, indent=2)}

Reference Links:
SEC: https://www.sec.gov/edgar/browse/?CIK={cik}
Yahoo Finance: https://finance.yahoo.com/quote/{symbol}
"""

def generate_ai_fundamental_single(symbol, snapshot, fm_snapshot, analyst, facts):
    if not OPENAI_API_KEY:
        return "OpenAI key missing."
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        res = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "Return analysis only. No greetings."},
                {"role": "user", "content": build_ai_single_prompt(symbol, snapshot, fm_snapshot, analyst, facts)}
            ],
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"AI ERROR: {e}"

def generate_ai_freelancing_single(symbol, snapshot):
    if not OPENAI_API_KEY:
        return "OpenAI key missing."
    company_name = snapshot.get("long_name", symbol)
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""
You are a senior equity analyst.

STRICT RULES:
- Return analysis only
- No greetings
- No filler
- No markdown

Explain everything an informed investor should know about:

Ticker: {symbol}
Company: {company_name}

Cover:
Business model, competitive position, strategy, risks, catalysts, long-term outlook.
"""

    try:
        res = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.35,
            messages=[
                {"role": "system", "content": "Return analysis only."},
                {"role": "user", "content": prompt}
            ],
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"AI ERROR: {e}"

def generate_ai_combined_pair(
    s1, d1, fm1, analyst1, facts1,
    s2, d2, fm2, analyst2, facts2
):
    if not OPENAI_API_KEY:
        return "OpenAI key missing."

    name1 = d1.get("long_name", s1)
    name2 = d2.get("long_name", s2)

    payload = {
        "stock_1": {"ticker": s1, "company": name1, "snapshot": d1, "metrics": fm1},
        "stock_2": {"ticker": s2, "company": name2, "snapshot": d2, "metrics": fm2},
    }

    prompt = f"""
You are a senior equity analyst.

STRICT RULES:
- Return analysis only
- No greetings
- No markdown
- No filler

Compare:

{ s1 } ({ name1 }) vs { s2 } ({ name2 })

Required sections:
1. Business Comparison
2. Valuation Comparison
3. Growth & Profitability
4. Risk Profile
5. Catalysts
6. Overall Assessment (NOT investment advice)

DATA:
{json.dumps(payload, indent=2)}
"""

    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        res = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "Return analysis only."},
                {"role": "user", "content": prompt}
            ],
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"AI ERROR: {e}"


def build_stock_snapshot(symbol: str) -> Dict[str, Any]:
    """
    Builds a reliable stock snapshot using FinancialDatasets.ai ONLY.
    No Yahoo Finance usage.
    """

    symbol = symbol.upper()

    snapshot = {
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

    # -------------------------------------------------
    # Company Facts (name, sector, industry, website)
    # -------------------------------------------------
    facts, err = fetch_company_facts(symbol)
    if facts:
        snapshot["long_name"] = facts.get("name") or facts.get("company_name") or symbol
        snapshot["sector"] = facts.get("sector", "N/A")
        snapshot["industry"] = facts.get("industry", "N/A")
        snapshot["website"] = (
            facts.get("website")
            or facts.get("website_url")
            or facts.get("homepage")
            or "N/A"
        )

    # -------------------------------------------------
    # Price history (FD prices endpoint)
    # -------------------------------------------------
    df, price_err = fetch_price_history_df(symbol, days=400)
    if df is None or df.empty or "close" not in df.columns:
        dbg(f"{symbol}: No price data available ({price_err})")
        return snapshot

    close = df["close"].astype(float)

    snapshot["current_price"] = float(close.iloc[-1])

    # Day change
    if len(close) >= 2:
        prev = float(close.iloc[-2])
        if prev != 0:
            delta = snapshot["current_price"] - prev
            snapshot["day_change_dollar"] = delta
            snapshot["day_change_pct"] = (delta / prev) * 100.0

    # 52-week high / low
    snapshot["year_low"] = float(close.min())
    snapshot["year_high"] = float(close.max())

    # 1-year change
    first = float(close.iloc[0])
    if first != 0:
        snapshot["change_1y_pct"] = (
            (snapshot["current_price"] - first) / first
        ) * 100.0

    return snapshot


# =========================================================
# PUBLIC ENTRY POINTS (REQUIRED BY APP)
# =========================================================

def run_single_to_pdf(symbol: str, out_dir: str) -> str:
    """
    REQUIRED ENTRY POINT
    Called by Flask / FastAPI / Railway app
    """
    from datetime import datetime
    import os

    symbol = symbol.upper()
    os.makedirs(out_dir, exist_ok=True)

    snapshot = build_stock_snapshot(symbol)
    fm_snapshot, _ = fetch_financial_metrics_snapshot(symbol)
    fm_history, _ = fetch_financial_metrics_history(symbol)
    analyst, _ = fetch_fd_analyst_estimates(symbol)
    facts, _ = fetch_company_facts(symbol)
    financials, _ = fetch_financials(symbol)

    lines = []

    lines.append(f"STOCK SNAPSHOT: {symbol}")
    lines.append("=" * 72)
    lines.append(f"Name     : {snapshot.get('long_name','N/A')}")
    lines.append(f"Sector   : {snapshot.get('sector','N/A')}")
    lines.append(f"Industry : {snapshot.get('industry','N/A')}")
    lines.append("")

    lines.append(generate_ai_fundamental_single(
        symbol, snapshot, fm_snapshot, analyst, facts
    ))

    chart_path = build_single_charts(symbol, fm_history)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(out_dir, f"{symbol}_{ts}.pdf")

    export_pdf(
        "\n".join(lines),
        f"{symbol} – {snapshot.get('long_name','')}",
        chart_path,
        out_file
    )

    return out_file


def run_compare_to_pdf(symbol1: str, symbol2: str, out_dir: str) -> str:
    """
    REQUIRED ENTRY POINT
    """
    from datetime import datetime
    import os

    s1 = symbol1.upper()
    s2 = symbol2.upper()
    os.makedirs(out_dir, exist_ok=True)

    d1 = build_stock_snapshot(s1)
    d2 = build_stock_snapshot(s2)

    fm1, _ = fetch_financial_metrics_snapshot(s1)
    fm2, _ = fetch_financial_metrics_snapshot(s2)

    analyst1, _ = fetch_fd_analyst_estimates(s1)
    analyst2, _ = fetch_fd_analyst_estimates(s2)

    facts1, _ = fetch_company_facts(s1)
    facts2, _ = fetch_company_facts(s2)

    lines = []
    lines.append(f"COMPARISON: {s1} vs {s2}")
    lines.append("=" * 72)

    lines.append(generate_ai_combined_pair(
        s1, d1, fm1, analyst1, facts1,
        s2, d2, fm2, analyst2, facts2
    ))

    chart_path = build_compare_charts(s1, s2)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(out_dir, f"{s1}_{s2}_{ts}.pdf")

    export_pdf(
        "\n".join(lines),
        f"{s1} vs {s2}",
        chart_path,
        out_file
    )

    return out_file


# =========================================================
# 🔒 EVERYTHING ELSE UNCHANGED
# =========================================================
# All FD calls, pricing logic, snapshot logic,
# PDF generation, charts, and public entry points
# remain exactly as in your last working version.


