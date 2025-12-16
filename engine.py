# engine.py
# ============================================================
# Clean, stable engine for Stock Analyzer
# ============================================================

import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd
from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch

# ============================================================
# Environment
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

FD_API_KEY = os.getenv("FINANCIAL_DATASETS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEBUG = os.getenv("DEBUG_ENGINE", "N").upper() == "Y"

FD_BASE = "https://api.financialdatasets.ai"

# ============================================================
# Debug helper
# ============================================================

def dbg(msg: str):
    if DEBUG:
        print(msg, flush=True)

# ============================================================
# Formatting helpers
# ============================================================

def safe_div(num, den):
    try:
        if num is None or den in (None, 0):
            return None
        return float(num) / float(den)
    except Exception:
        return None

def r4(x):
    return round(x, 4) if x is not None else None

def fmt(x, d=2):
    try:
        return f"{float(x):,.{d}f}"
    except Exception:
        return "N/A"

def fmt_int(x):
    try:
        return f"{int(x):,}"
    except Exception:
        return "N/A"

# ============================================================
# HTTP helpers
# ============================================================

def fd_headers():
    return {"X-API-KEY": FD_API_KEY} if FD_API_KEY else {}

def fd_get(path: str, params: dict = None):
    if not FD_API_KEY:
        return None, "FD_API_KEY missing"

    url = f"{FD_BASE}{path}"
    try:
        r = requests.get(
            url,
            headers=fd_headers(),
            params=params,
            timeout=30,
            allow_redirects=True
        )
        dbg(f"FD {path} {r.status_code}")
        if r.status_code != 200:
            return None, r.text[:200]
        return r.json(), None
    except Exception as e:
        return None, str(e)

# ============================================================
# Data fetchers (FinancialDatasets.ai)
# ============================================================

def fetch_company_facts(ticker: str):
    return fd_get("/company/facts", {"ticker": ticker})

def fetch_price_history(ticker: str, days=400):
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    return fd_get(
        "/prices",
        {
            "ticker": ticker,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "interval": "day",
            "interval_multiplier": 1,
        },
    )

def fetch_metrics_snapshot(ticker: str):
    return fd_get("/financial-metrics/snapshot/", {"ticker": ticker})

def fetch_metrics_history(ticker: str):
    return fd_get("/financial-metrics", {"ticker": ticker, "period": "annual", "limit": 10})

def fetch_financials(ticker: str):
    return fd_get("/financials", {"ticker": ticker, "period": "annual", "limit": 10})

def fetch_analyst_estimates(ticker: str):
    return fd_get("/analyst-estimates", {"ticker": ticker, "period": "annual"})

# ============================================================
# Snapshot builder (NO Yahoo)
# ============================================================

def build_stock_snapshot(ticker: str) -> Dict[str, Any]:
    snapshot = {
        "ticker": ticker,
        "company_name": ticker,
        "sector": "N/A",
        "industry": "N/A",
        "website": "N/A",
        "current_price": None,
        "day_change_pct": None,
        "day_change": None,
        "low_52w": None,
        "high_52w": None,
        "change_1y": None,
    }

    facts, _ = fetch_company_facts(ticker)
    if facts and isinstance(facts, dict):
        cf = facts.get("company_facts", {})
        snapshot["company_name"] = cf.get("name", ticker)
        snapshot["sector"] = cf.get("sector", "N/A")
        snapshot["industry"] = cf.get("industry", "N/A")
        snapshot["website"] = cf.get("website", "N/A")

    prices, err = fetch_price_history(ticker)
    if err or not prices or "prices" not in prices:
        dbg(f"{ticker} price error: {err}")
        return snapshot

    df = pd.DataFrame(prices["prices"])
    if df.empty or "close" not in df.columns:
        return snapshot

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])

    if df.empty:
        return snapshot

    snapshot["current_price"] = float(df["close"].iloc[-1])
    snapshot["low_52w"] = float(df["close"].min())
    snapshot["high_52w"] = float(df["close"].max())

    if len(df) > 1:
        prev = df["close"].iloc[-2]
        snapshot["day_change"] = snapshot["current_price"] - prev
        snapshot["day_change_pct"] = (snapshot["day_change"] / prev) * 100

    first = df["close"].iloc[0]
    snapshot["change_1y"] = ((snapshot["current_price"] - first) / first) * 100

    return snapshot

# ============================================================
# Multi-year fundamentals table
# ============================================================

def build_fundamentals(financials: dict) -> List[Dict[str, Any]]:
    rows = []
    stmts = financials.get("income_statements", []) if financials else []

    prev_rev = prev_eps = None

    for s in stmts:
        if s.get("period") != "annual":
            continue

        rev = s.get("revenue")
        gp = s.get("gross_profit")
        op = s.get("operating_income")
        ni = s.get("net_income")
        eps = s.get("earnings_per_share")

        row = {
            "year": (s.get("report_period") or "")[:4],
            "sales": rev,
            "gp": gp,
            "op": op,
            "ni": ni,
            "eps": eps,
            "gp_margin": r4(safe_div(gp, rev)),
            "op_margin": r4(safe_div(op, rev)),
            "ni_margin": r4(safe_div(ni, rev)),
            "sales_yoy": r4(safe_div(rev - prev_rev, prev_rev)) if prev_rev else None,
            "eps_yoy": r4(safe_div(eps - prev_eps, prev_eps)) if prev_eps else None,
        }

        rows.append(row)
        prev_rev = rev
        prev_eps = eps

    return rows[-5:]

# ============================================================
# PDF Export
# ============================================================

def export_pdf(title: str, blocks: List[str], chart: Optional[str], out_path: str):
    styles = getSampleStyleSheet()
    h = ParagraphStyle("h", fontSize=16, spaceAfter=10)
    b = ParagraphStyle("b", fontSize=10, leading=14)

    doc = SimpleDocTemplate(out_path, pagesize=letter)
    story = []

    story.append(Paragraph(title, h))
    story.append(Spacer(1, 12))

    for block in blocks:
        for line in block.split("\n"):
            story.append(Paragraph(line, b))
        story.append(Spacer(1, 10))

    if chart and os.path.exists(chart):
        story.append(PageBreak())
        story.append(Image(chart, 6.5 * inch, 4.5 * inch))

    doc.build(story)

# ============================================================
# PUBLIC ENTRY POINTS (USED BY app.py)
# ============================================================

def run_single_to_pdf(ticker: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)

    snapshot = build_stock_snapshot(ticker)

    metrics, _ = fetch_metrics_snapshot(ticker)
    financials, _ = fetch_financials(ticker)

    fundamentals = build_fundamentals(financials or {})

    blocks = []

    blocks.append(
        f"""STOCK SNAPSHOT
Name: {snapshot['company_name']}
Sector: {snapshot['sector']}
Industry: {snapshot['industry']}
Website: {snapshot['website']}

Price: {fmt(snapshot['current_price'])}
Day Change: {fmt(snapshot['day_change'])} ({fmt(snapshot['day_change_pct'])}%)
52W Low / High: {fmt(snapshot['low_52w'])} / {fmt(snapshot['high_52w'])}
1Y Change: {fmt(snapshot['change_1y'])}%
"""
    )

    if fundamentals:
        lines = ["MULTI-YEAR FUNDAMENTALS"]
        for r in fundamentals:
            lines.append(
                f"{r['year']} | Sales {fmt_int(r['sales'])} | EPS {fmt(r['eps'])} | "
                f"GP% {fmt(r['gp_margin'],4)} | Op% {fmt(r['op_margin'],4)}"
            )
        blocks.append("\n".join(lines))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(out_dir, f"{ticker}_{ts}.pdf")
    export_pdf(f"{ticker} Report", blocks, None, out)
    pdf_path = out_dir
    return pdf_path, report


def run_compare_to_pdf(s1: str, s2: str, out_dir: str):
    s1 = s1.upper()
    s2 = s2.upper()
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------
    # DATA COLLECTION
    # -------------------------------
    snap1 = build_stock_snapshot(s1)
    snap2 = build_stock_snapshot(s2)

    fm1, _ = fetch_financial_metrics_snapshot(s1)
    fm2, _ = fetch_financial_metrics_snapshot(s2)

    analyst1, _ = fetch_fd_analyst_estimates(s1)
    analyst2, _ = fetch_fd_analyst_estimates(s2)

    facts1, _ = fetch_company_facts(s1)
    facts2, _ = fetch_company_facts(s2)

    news1, _ = fetch_news(s1)
    news2, _ = fetch_news(s2)

    inst1, _ = fetch_institutional(s1)
    inst2, _ = fetch_institutional(s2)

    insider1, _ = fetch_insider_trades(s1)
    insider2, _ = fetch_insider_trades(s2)

    # -------------------------------
    # STRUCTURED REPORT (GUI + PDF)
    # -------------------------------
    report = {
        "mode": "compare",
        "tickers": [s1, s2],
        "left": {
            "snapshot": snap1,
            "financial_metrics": fm1,
            "analyst_estimates": analyst1,
            "company_facts": facts1,
            "news": news1,
            "institutional": inst1,
            "insider": insider1,
        },
        "right": {
            "snapshot": snap2,
            "financial_metrics": fm2,
            "analyst_estimates": analyst2,
            "company_facts": facts2,
            "news": news2,
            "institutional": inst2,
            "insider": insider2,
        },
    }

    # -------------------------------
    # TEXT BODY FOR PDF
    # -------------------------------
    lines = []
    lines.append(f"STOCK COMPARISON: {s1} vs {s2}")
    lines.append("=" * 72)

    lines.append(
        generate_ai_combined_pair(
            s1, snap1, fm1, analyst1, facts1,
            s2, snap2, fm2, analyst2, facts2
        )
    )

    # -------------------------------
    # CHARTS
    # -------------------------------
    chart_path = build_compare_charts(s1, s2)

    # -------------------------------
    # PDF EXPORT
    # -------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(out_dir, f"{s1}_{s2}_{ts}.pdf")

    export_pdf(
        text="\n".join(lines),
        title_line=f"{s1} vs {s2}",
        chart_path=chart_path,
        output_path=pdf_path
    )

    # ✅ RETURN BOTH VALUES
    return pdf_path, report


# ============================================================
# Ticker lookup (safe stub for UI autocomplete)
# ============================================================

def lookup_tickers(query: str):
    """
    Lightweight ticker lookup placeholder.
    Keeps app.py imports stable.
    Replace later with FinancialDatasets / Polygon / Nasdaq API.
    """
    if not query:
        return []

    query = query.upper().strip()

    # Minimal fallback examples
    COMMON = {
        "AAPL": "Apple Inc",
        "MSFT": "Microsoft Corp",
        "GOOGL": "Alphabet Inc",
        "AMZN": "Amazon.com Inc",
        "META": "Meta Platforms Inc",
        "TSLA": "Tesla Inc",
        "NVDA": "NVIDIA Corp",
        "FI": "Fiserv Inc",
    }

    results = []
    for ticker, name in COMMON.items():
        if query in ticker or query in name.upper():
            results.append({
                "ticker": ticker,
                "name": name
            })

    return results




