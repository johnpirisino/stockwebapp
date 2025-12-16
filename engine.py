# engine.py
import os
import json
import traceback
from datetime import datetime
from typing import Dict, Any, List, Tuple

import requests
from dotenv import load_dotenv

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ======================================================
# Environment
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

FD_API_KEY = os.getenv("FINANCIAL_DATASETS_API_KEY")
FD_BASE = "https://api.financialdatasets.ai"

DEBUG = os.getenv("DEBUG_ENGINE", "N").upper() == "Y"


def dbg(*args):
    if DEBUG:
        print("[ENGINE]", *args, flush=True)


# ======================================================
# Utilities
# ======================================================

def safe_get(url: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    r = requests.get(
        url,
        params=params,
        headers={"X-API-KEY": FD_API_KEY},
        timeout=30,
        allow_redirects=True,
    )
    dbg("GET", r.url, r.status_code)
    r.raise_for_status()
    return r.json()


def r4(val):
    try:
        return round(float(val), 4)
    except Exception:
        return None


# ======================================================
# Lookup
# ======================================================

def lookup_tickers(query: str) -> List[Dict[str, Any]]:
    """
    Simple ticker lookup using FinancialDatasets search
    """
    data = safe_get(
        f"{FD_BASE}/search",
        {"query": query, "limit": 10},
    )

    results = []
    for r in data.get("results", []):
        results.append({
            "ticker": r.get("ticker"),
            "name": r.get("name"),
            "exchange": r.get("exchange"),
        })
    return results


# ======================================================
# Data Fetchers
# ======================================================

def fetch_company_facts(ticker: str) -> Dict[str, Any]:
    data = safe_get(f"{FD_BASE}/company/facts", {"ticker": ticker})
    return data.get("company_facts", {})


def fetch_snapshot_metrics(ticker: str) -> Dict[str, Any]:
    data = safe_get(
        f"{FD_BASE}/financial-metrics/snapshot",
        {"ticker": ticker},
    )
    return data.get("snapshot", {})


def fetch_financials(ticker: str) -> List[Dict[str, Any]]:
    data = safe_get(
        f"{FD_BASE}/financials",
        {"ticker": ticker, "period": "annual", "limit": 5},
    )
    return data.get("income_statements", [])


def fetch_news(ticker: str) -> List[Dict[str, Any]]:
    data = safe_get(f"{FD_BASE}/news", {"ticker": ticker, "limit": 5})
    return data.get("news", [])


# ======================================================
# PDF Builder
# ======================================================

def build_pdf(title: str, blocks: List[Any], output_path: str):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 12))

    for block in blocks:
        if isinstance(block, str):
            story.append(Paragraph(block.replace("\n", "<br/>"), styles["BodyText"]))
            story.append(Spacer(1, 8))
        elif isinstance(block, Table):
            story.append(block)
            story.append(Spacer(1, 12))

    doc.build(story)


# ======================================================
# Core Builders
# ======================================================

def build_fundamentals_table(rows: List[Dict[str, Any]]) -> Table:
    header = [
        "Year", "Sales", "Gross Profit", "Op Income", "Net Income",
        "EPS", "GP%", "OP%", "NI%"
    ]

    table_data = [header]

    for r in rows:
        rev = r.get("revenue")
        gp = r.get("gross_profit")
        op = r.get("operating_income")
        ni = r.get("net_income")
        eps = r.get("earnings_per_share")

        table_data.append([
            r.get("report_period", "")[:4],
            r4(rev),
            r4(gp),
            r4(op),
            r4(ni),
            r4(eps),
            r4(gp / rev if rev else None),
            r4(op / rev if rev else None),
            r4(ni / rev if rev else None),
        ])

    t = Table(table_data, repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))
    return t


# ======================================================
# PUBLIC API (required by app.py)
# ======================================================

def run_single_to_pdf(ticker: str, out_dir: str) -> Tuple[str, Dict[str, Any]]:
    ticker = ticker.upper()
    os.makedirs(out_dir, exist_ok=True)

    facts = fetch_company_facts(ticker)
    snapshot = fetch_snapshot_metrics(ticker)
    financials = fetch_financials(ticker)
    news = fetch_news(ticker)

    fundamentals_table = build_fundamentals_table(financials)

    # ---- Report dict (GUI uses this) ----
    report = {
        "mode": "single",
        "ticker": ticker,
        "company_name": facts.get("name"),
        "facts": facts,
        "snapshot": snapshot,
        "financials": financials,
        "news": news,
    }

    # ---- PDF ----
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(out_dir, f"{ticker}_{ts}.pdf")

    blocks = [
        f"<b>Company:</b> {facts.get('name')} ({ticker})",
        f"<b>Sector:</b> {facts.get('sector')}<br/>"
        f"<b>Industry:</b> {facts.get('industry')}",
        "<b>Financial Metrics Snapshot</b><br/>" +
        json.dumps(snapshot, indent=2),
        "<b>Multi-Year Fundamentals</b>",
        fundamentals_table,
    ]

    build_pdf(f"{ticker} Stock Report", blocks, pdf_path)

    return pdf_path, report


def run_compare_to_pdf(
    ticker1: str,
    ticker2: str,
    out_dir: str
) -> Tuple[str, Dict[str, Any]]:
    t1 = ticker1.upper()
    t2 = ticker2.upper()
    os.makedirs(out_dir, exist_ok=True)

    f1 = fetch_company_facts(t1)
    f2 = fetch_company_facts(t2)

    s1 = fetch_snapshot_metrics(t1)
    s2 = fetch_snapshot_metrics(t2)

    report = {
        "mode": "compare",
        "ticker1": t1,
        "ticker2": t2,
        "company1": f1,
        "company2": f2,
        "snapshot1": s1,
        "snapshot2": s2,
    }

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(out_dir, f"{t1}_{t2}_{ts}.pdf")

    blocks = [
        f"<b>{f1.get('name')} ({t1})</b><br/>{json.dumps(s1, indent=2)}",
        f"<b>{f2.get('name')} ({t2})</b><br/>{json.dumps(s2, indent=2)}",
    ]

    build_pdf(f"{t1} vs {t2} Comparison", blocks, pdf_path)

    return pdf_path, report
