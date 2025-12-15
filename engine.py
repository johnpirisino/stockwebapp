import os
import json
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd
from dotenv import load_dotenv

# Optional AI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Optional yfinance fallback (can be rate-limited)
USE_YFINANCE_FALLBACK = os.getenv("USE_YFINANCE_FALLBACK", "N").upper() == "Y"
if USE_YFINANCE_FALLBACK:
    try:
        import yfinance as yf
    except Exception:
        yf = None
else:
    yf = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image,
    Table, TableStyle
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.lib import colors


# ==========================================================
# ENV
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FD_API_KEY = os.getenv("FINANCIAL_DATASETS_API_KEY")
FD_BASE_URL = "https://api.financialdatasets.ai"

DEBUG = os.getenv("DEBUG_ENGINE", "N").upper() == "Y"


def dbg(*args):
    if DEBUG:
        print("[ENGINE]", *args, flush=True)


# ==========================================================
# FORMAT HELPERS
# ==========================================================
def fmt_number(v, decimals=2) -> str:
    try:
        if v is None:
            return "N/A"
        return f"{float(v):,.{decimals}f}"
    except Exception:
        return "N/A"


def fmt_int(v) -> str:
    try:
        if v is None:
            return "N/A"
        return f"{int(float(v)):,}"
    except Exception:
        return "N/A"


def fmt_pct(v, decimals=1) -> str:
    try:
        if v is None:
            return "N/A"
        return f"{float(v) * 100:.{decimals}f}%"
    except Exception:
        return "N/A"


# ==========================================================
# FINANCIALDATASETS CLIENT
# ==========================================================
def fd_headers() -> Dict[str, str]:
    return {"X-API-KEY": FD_API_KEY} if FD_API_KEY else {}


def fd_get_json(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Tuple[Optional[Any], Optional[str]]:
    if not FD_API_KEY:
        return None, "FINANCIAL_DATASETS_API_KEY missing."

    url = f"{FD_BASE_URL}{path}"

    try:
        r = requests.get(
            url,
            headers=fd_headers(),
            params=params,
            timeout=timeout,
            allow_redirects=True,  # IMPORTANT (FD uses 301s without trailing slash)
        )
        dbg("FD GET", r.status_code, r.url)

        if r.status_code != 200:
            return None, f"{r.status_code}: {(r.text or '')[:200]}"

        return r.json(), None
    except Exception as e:
        return None, str(e)


# ==========================================================
# ENDPOINT WRAPPERS
# ==========================================================
def fetch_company_facts(symbol: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    data, err = fd_get_json("/company/facts", {"ticker": symbol.upper()}, timeout=30)
    if err or not data:
        return None, err or "No company facts."

    if isinstance(data, dict) and isinstance(data.get("company_facts"), dict):
        return data["company_facts"], None

    return None, "Unexpected company facts format."


def fetch_financial_metrics_snapshot(symbol: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    # FD redirects /financial-metrics/snapshot -> /financial-metrics/snapshot/
    data, err = fd_get_json("/financial-metrics/snapshot", {"ticker": symbol.upper()}, timeout=30)
    if err or not data:
        return None, err or "No snapshot."

    if isinstance(data, dict) and isinstance(data.get("snapshot"), dict):
        return data["snapshot"], None

    # Some accounts might return raw snapshot
    if isinstance(data, dict) and "ticker" in data:
        return data, None

    return None, "Unexpected metrics snapshot format."


def fetch_financial_metrics_history(symbol: str, period: str = "annual", limit: int = 10) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    data, err = fd_get_json("/financial-metrics", {"ticker": symbol.upper(), "period": period, "limit": limit}, timeout=30)
    if err or not data:
        return None, err or "No metrics history."

    if isinstance(data, list):
        return data, None
    if isinstance(data, dict):
        for key in ("metrics", "financial_metrics", "results"):
            if isinstance(data.get(key), list):
                return data[key], None
        return [data], None

    return None, "Unexpected metrics history format."


def fetch_financials(symbol: str, limit: int = 10) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    data, err = fd_get_json("/financials", {"ticker": symbol.upper(), "period": "annual", "limit": limit}, timeout=30)
    if err or not data:
        return None, err or "No financials."

    if isinstance(data, dict) and isinstance(data.get("income_statements"), list):
        return data, None

    if isinstance(data, dict) and isinstance(data.get("financials"), dict):
        f = data["financials"]
        if isinstance(f.get("income_statements"), list):
            return f, None

    return None, "Unexpected financials format."


def fetch_analyst_estimates(symbol: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    data, err = fd_get_json("/analyst-estimates", {"ticker": symbol.upper(), "period": "annual"}, timeout=30)
    if err or not data:
        return None, err or "No analyst estimates."

    if isinstance(data, dict) and isinstance(data.get("analyst_estimates"), list):
        return data["analyst_estimates"], None

    return None, "Unexpected analyst estimates format."


def fetch_news(symbol: str, limit: int = 6) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    data, err = fd_get_json("/news", {"ticker": symbol.upper(), "limit": limit}, timeout=30)
    if err or not data:
        return [], err or "No news."

    if isinstance(data, dict) and isinstance(data.get("news"), list):
        return data["news"], None
    return [], "Unexpected news format."


def fetch_insider_transactions(symbol: str, limit: int = 10) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    data, err = fd_get_json("/insider-transactions", {"ticker": symbol.upper(), "limit": limit}, timeout=30)
    if err or not data:
        return [], err or "No insider transactions."

    if isinstance(data, dict):
        lst = data.get("insider_transactions") or data.get("insider_trades") or []
        if isinstance(lst, list):
            return lst, None

    return [], "Unexpected insider format."


def fetch_institutional_ownership(symbol: str, limit: int = 10) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    data, err = fd_get_json("/institutional-ownership", {"ticker": symbol.upper(), "limit": limit}, timeout=45)
    if err or not data:
        return [], err or "No institutional ownership."

    if isinstance(data, dict) and isinstance(data.get("institutional_ownership"), list):
        return data["institutional_ownership"], None

    return [], "Unexpected institutional format."


def fetch_prices_fd(symbol: str, days: int = 400) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)

    params = {
        "ticker": symbol.upper(),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "interval": "day",
        "interval_multiplier": 1,
    }

    data, err = fd_get_json("/prices", params, timeout=45)
    if err or not data:
        return None, err or "No prices response."

    if not isinstance(data, dict) or not isinstance(data.get("prices"), list):
        return None, "Unexpected prices format."

    df = pd.DataFrame(data["prices"])
    if df.empty:
        return None, "Empty prices list."

    # Normalize time
    if "time" in df.columns:
        df["time"] = df["time"].astype(str).str.replace(" EDT", "", regex=False).str.replace(" EST", "", regex=False)
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df = df.dropna(subset=["time"]).set_index("time")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df = df.dropna(subset=["date"]).set_index("date")
    else:
        return None, "Prices missing time/date."

    # Column normalization
    # FD commonly uses: open/high/low/close/volume
    if "close" not in df.columns:
        return None, "Prices missing close."

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df.sort_index()
    return df, None


def fetch_prices_yf(symbol: str, days: int = 400) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if not yf:
        return None, "yfinance not enabled."

    try:
        t = yf.Ticker(symbol)
        hist = t.history(period=f"{days}d", interval="1d")
        if hist is None or hist.empty:
            return None, "No yfinance history."
        df = pd.DataFrame({
            "close": hist["Close"].astype(float),
            "volume": hist.get("Volume", 0).astype(float),
        }, index=pd.to_datetime(hist.index, utc=True))
        return df.sort_index(), None
    except Exception as e:
        return None, str(e)


# ==========================================================
# SNAPSHOT BUILDER (HYBRID PRICING)
# ==========================================================
def build_stock_snapshot(symbol: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      snapshot dict (for report)
      debug dict (errors, sources)
    """
    symbol = symbol.upper()
    debug: Dict[str, Any] = {"symbol": symbol, "sources": {}}

    snapshot: Dict[str, Any] = {
        "symbol": symbol,
        "company_name": symbol,
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

    facts, facts_err = fetch_company_facts(symbol)
    if facts:
        snapshot["company_name"] = facts.get("name") or facts.get("company_name") or symbol
        snapshot["sector"] = facts.get("sector") or "N/A"
        snapshot["industry"] = facts.get("industry") or "N/A"
        snapshot["website"] = facts.get("website_url") or facts.get("website") or "N/A"
        debug["sources"]["company_facts"] = "financialdatasets.ai"
    else:
        debug["sources"]["company_facts"] = "missing"
        debug["company_facts_error"] = facts_err

    # Pricing: FD first
    df, err = fetch_prices_fd(symbol, days=400)
    debug["sources"]["prices"] = "financialdatasets.ai"
    if df is None or df.empty:
        debug["prices_error_fd"] = err

        # Optional yfinance fallback
        if USE_YFINANCE_FALLBACK:
            df, err2 = fetch_prices_yf(symbol, days=400)
            debug["sources"]["prices"] = "yfinance"
            if df is None or df.empty:
                debug["prices_error_yf"] = err2
                return snapshot, debug
        else:
            return snapshot, debug

    close = df["close"].astype(float)

    snapshot["current_price"] = float(close.iloc[-1])

    if len(close) >= 2:
        prev = float(close.iloc[-2])
        if prev != 0:
            delta = snapshot["current_price"] - prev
            snapshot["day_change_dollar"] = delta
            snapshot["day_change_pct"] = (delta / prev) * 100.0

    # Use last ~1y from series
    snapshot["year_low"] = float(close.min())
    snapshot["year_high"] = float(close.max())

    first = float(close.iloc[0])
    if first != 0:
        snapshot["change_1y_pct"] = (snapshot["current_price"] - first) / first * 100.0

    return snapshot, debug


# ==========================================================
# AI (NO GREETINGS / NO SIDE CONVO)
# ==========================================================
def ai_enabled() -> bool:
    return bool(OPENAI_API_KEY) and (OpenAI is not None)


def ai_company_label(snapshot: Dict[str, Any]) -> str:
    sym = snapshot.get("symbol") or ""
    name = snapshot.get("company_name") or sym
    return f"{sym} ({name})"


def generate_ai_single(snapshot: Dict[str, Any], fm_snapshot: Optional[Dict[str, Any]], analyst: Optional[List[Dict[str, Any]]]) -> str:
    if not ai_enabled():
        return "OpenAI not configured (missing key or openai package)."

    label = ai_company_label(snapshot)

    prompt = f"""
Return analysis only. No greetings. No disclaimers beyond “not investment advice”.
No ### or ** formatting.

Analyze: {label}

Sections:
1) Company overview
2) Performance snapshot (1Y %, day change)
3) Valuation (use provided ratios)
4) Profitability & quality (margins, ROE/ROA/ROIC, leverage)
5) Analyst expectations
6) Key risks
7) Verdict label (Buy/Hold/Avoid) — NOT investment advice

Data:
SNAPSHOT:
{json.dumps(snapshot, indent=2)}

FINANCIAL METRICS SNAPSHOT:
{json.dumps(fm_snapshot, indent=2)}

ANALYST ESTIMATES:
{json.dumps(analyst, indent=2)}
""".strip()

    client = OpenAI(api_key=OPENAI_API_KEY)
    res = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=0.3,
        messages=[
            {"role": "system", "content": "Return analysis only. No greetings. No fluff."},
            {"role": "user", "content": prompt},
        ],
    )
    return res.choices[0].message.content


def generate_ai_compare(s1: Dict[str, Any], s2: Dict[str, Any], fm1: Optional[Dict[str, Any]], fm2: Optional[Dict[str, Any]]) -> str:
    if not ai_enabled():
        return "OpenAI not configured (missing key or openai package)."

    l1 = ai_company_label(s1)
    l2 = ai_company_label(s2)

    prompt = f"""
Return analysis only. No greetings. No side conversation. No ### or ** formatting.

Compare: {l1} vs {l2}

Required sections:
1) Business & moat
2) Momentum & performance
3) Valuation comparison
4) Profitability & balance sheet
5) Risks
6) Catalysts
7) Which looks more attractive now and why (NOT investment advice)

DATA 1:
{json.dumps({"snapshot": s1, "metrics": fm1}, indent=2)}

DATA 2:
{json.dumps({"snapshot": s2, "metrics": fm2}, indent=2)}
""".strip()

    client = OpenAI(api_key=OPENAI_API_KEY)
    res = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=0.3,
        messages=[
            {"role": "system", "content": "Return analysis only. No greetings. No fluff."},
            {"role": "user", "content": prompt},
        ],
    )
    return res.choices[0].message.content


# ==========================================================
# TABLE BUILDERS (STRUCTURED)
# ==========================================================
def build_fundamentals_rows(financials: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    inc = (financials or {}).get("income_statements") or []
    rows: List[Dict[str, Any]] = []

    for item in inc:
        if item.get("period") != "annual":
            continue
        rp = item.get("report_period") or item.get("fiscal_period")
        if not rp:
            continue
        year = rp[:4]

        revenue = item.get("revenue")
        gp = item.get("gross_profit")
        op = item.get("operating_income") or item.get("ebit")
        ni = item.get("net_income")
        eps = item.get("earnings_per_share")

        rows.append({
            "year": year,
            "revenue": revenue,
            "gross_profit": gp,
            "operating_income": op,
            "net_income": ni,
            "eps": eps,
        })

    rows.sort(key=lambda x: x["year"])
    if len(rows) > 5:
        rows = rows[-5:]

    # Add margins + YoY
    prev_rev = None
    prev_eps = None

    for r in rows:
        rev = r.get("revenue")
        gp = r.get("gross_profit")
        op = r.get("operating_income")
        ni = r.get("net_income")
        eps = r.get("eps")

        def safe_div(a, b):
            try:
                if a is None or b in (None, 0):
                    return None
                return float(a) / float(b)
            except Exception:
                return None
def r4(x):
    return round(x, 4) if x is not None else None

r["gp_margin"] = r4(safe_div(gp, rev))
r["op_margin"] = r4(safe_div(op, rev))
r["ni_margin"] = r4(safe_div(ni, rev))

if prev_rev not in (None, 0) and rev not in (None, 0):
    r["rev_yoy"] = r4((float(rev) - float(prev_rev)) / float(prev_rev))
else:
    r["rev_yoy"] = None

if prev_eps not in (None, 0) and eps not in (None, 0):
    r["eps_yoy"] = r4((float(eps) - float(prev_eps)) / float(prev_eps))
else:
    r["eps_yoy"] = None


        prev_rev = rev
        prev_eps = eps

    return rows


def metrics_grid_pairs(fm: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    if not fm:
        return []

    # label, formatted value
    pairs: List[Tuple[str, str]] = [
        ("Market Cap", fmt_int(fm.get("market_cap"))),
        ("Enterprise Value", fmt_int(fm.get("enterprise_value"))),
        ("P/E", fmt_number(fm.get("price_to_earnings_ratio"), 4)),
        ("P/B", fmt_number(fm.get("price_to_book_ratio"), 4)),
        ("P/S", fmt_number(fm.get("price_to_sales_ratio"), 4)),
        ("EV/EBITDA", fmt_number(fm.get("enterprise_value_to_ebitda_ratio"), 4)),
        ("EV/Sales", fmt_number(fm.get("enterprise_value_to_revenue_ratio"), 4)),
        ("FCF Yield", fmt_number(fm.get("free_cash_flow_yield"), 4)),
        ("PEG", fmt_number(fm.get("peg_ratio"), 4)),
        ("Gross Margin", fmt_pct(fm.get("gross_margin"), 1)),
        ("Operating Margin", fmt_pct(fm.get("operating_margin"), 1)),
        ("Net Margin", fmt_pct(fm.get("net_margin"), 1)),
        ("ROE", fmt_pct(fm.get("return_on_equity"), 1)),
        ("ROA", fmt_pct(fm.get("return_on_assets"), 1)),
        ("ROIC", fmt_pct(fm.get("return_on_invested_capital"), 1)),
        ("Current Ratio", fmt_number(fm.get("current_ratio"), 4)),
        ("Quick Ratio", fmt_number(fm.get("quick_ratio"), 4)),
        ("Cash Ratio", fmt_number(fm.get("cash_ratio"), 4)),
        ("Debt/Equity", fmt_number(fm.get("debt_to_equity"), 4)),
        ("Debt/Assets", fmt_number(fm.get("debt_to_assets"), 4)),
        ("Interest Coverage", fmt_number(fm.get("interest_coverage"), 4)),
        ("Revenue Growth", fmt_pct(fm.get("revenue_growth"), 2)),
        ("Earnings Growth", fmt_pct(fm.get("earnings_growth"), 2)),
        ("EPS Growth", fmt_pct(fm.get("earnings_per_share_growth"), 2)),
        ("FCF Growth", fmt_pct(fm.get("free_cash_flow_growth"), 2)),
    ]
    return pairs


# ==========================================================
# CHARTS
# ==========================================================
def build_single_charts(symbol: str) -> Optional[str]:
    df, err = fetch_prices_fd(symbol, days=365)
    if (df is None or df.empty) and USE_YFINANCE_FALLBACK:
        df, err = fetch_prices_yf(symbol, days=365)

    if df is None or df.empty:
        dbg("charts: no price data", symbol, err)
        return None

    close = df["close"].astype(float)
    vol = df["volume"].astype(float)

    hist = pd.DataFrame({"Close": close, "Volume": vol})
    hist["MA20"] = hist["Close"].rolling(20).mean()
    hist["MA50"] = hist["Close"].rolling(50).mean()
    hist["MA200"] = hist["Close"].rolling(200).mean()

    delta = hist["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    ema12 = hist["Close"].ewm(span=12, adjust=False).mean()
    ema26 = hist["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    fig, axs = plt.subplots(4, 1, figsize=(8.5, 11))
    fig.subplots_adjust(hspace=0.4)
    dates = hist.index

    axs[0].plot(dates, hist["Close"], label="Close")
    axs[0].plot(dates, hist["MA20"], label="MA20", linewidth=0.8)
    axs[0].plot(dates, hist["MA50"], label="MA50", linewidth=0.8)
    axs[0].plot(dates, hist["MA200"], label="MA200", linewidth=0.8)
    axs[0].set_title(f"{symbol} Price + Moving Averages")
    axs[0].legend(fontsize=8)

    axs[1].bar(dates, hist["Volume"] / 1_000_000.0, width=1.0)
    axs[1].set_title("Daily Volume (M)")

    axs[2].plot(dates, rsi)
    axs[2].axhline(70, linestyle="--", linewidth=0.8)
    axs[2].axhline(30, linestyle="--", linewidth=0.8)
    axs[2].set_title("RSI (14)")
    axs[2].set_ylim(0, 100)

    axs[3].plot(dates, macd, label="MACD")
    axs[3].plot(dates, signal, linestyle="--", label="Signal")
    axs[3].axhline(0, linewidth=0.8)
    axs[3].set_title("MACD (12/26/9)")
    axs[3].legend(fontsize=8)

    out_dir = os.path.join(BASE_DIR, "generated_reports")
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, f"charts_{symbol}.png")
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return img_path


# ==========================================================
# PDF RENDER (MODERN GRIDS)
# ==========================================================
def make_table(data: List[List[Any]], col_widths: Optional[List[int]] = None) -> Table:
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),

        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#0b1220")),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.white),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),

        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#25324a")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
    ]))
    return t


def export_pdf(report: Dict[str, Any], output_path: str) -> None:
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name="Title",
        fontName="Helvetica-Bold",
        fontSize=22,
        alignment=TA_CENTER,
        textColor=colors.white,
        spaceAfter=18,
    )
    sub_style = ParagraphStyle(
        name="Sub",
        fontName="Helvetica",
        fontSize=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#9ca3af"),
        spaceAfter=18,
    )
    h_style = ParagraphStyle(
        name="Header",
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=colors.white,
        spaceBefore=14,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        name="Body",
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.white,
        leading=12,
    )

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    story = []

    # Title
    story.append(Paragraph("John Pirisino's Stock Analyzer", title_style))
    story.append(Paragraph(report.get("title_line", ""), sub_style))

    # Snapshot Grid
    snap = report.get("snapshot") or {}
    story.append(Paragraph("STOCK SNAPSHOT", h_style))

    snap_grid = [
        ["Field", "Value"],
        ["Ticker", snap.get("symbol", "N/A")],
        ["Company", snap.get("company_name", "N/A")],
        ["Sector", snap.get("sector", "N/A")],
        ["Industry", snap.get("industry", "N/A")],
        ["Website", snap.get("website", "N/A")],
        ["Current Price", fmt_number(snap.get("current_price"), 2)],
        ["Day Change", f"{fmt_number(snap.get('day_change_dollar'), 2)}  ({fmt_number(snap.get('day_change_pct'), 2)}%)"],
        ["52W Range", f"{fmt_number(snap.get('year_low'), 2)}  –  {fmt_number(snap.get('year_high'), 2)}"],
        ["1Y Change", f"{fmt_number(snap.get('change_1y_pct'), 2)}%"],
    ]
    story.append(make_table(snap_grid, col_widths=[160, 340]))

    # Fundamentals Table
    story.append(Paragraph("MULTI-YEAR FUNDAMENTALS (Annual)", h_style))
    fundamentals = report.get("fundamentals_rows") or []
    if fundamentals:
        f_table = [["Year", "Sales", "GP", "OpInc", "NetInc", "EPS", "GP%", "OP%", "NI%", "Sls YoY", "EPS YoY"]]
        for r in fundamentals:
            f_table.append([
                r.get("year"),
                fmt_int(r.get("revenue")),
                fmt_int(r.get("gross_profit")),
                fmt_int(r.get("operating_income")),
                fmt_int(r.get("net_income")),
                fmt_number(r.get("eps"), 2),
                fmt_pct(r.get("gp_margin"), 1),
                fmt_pct(r.get("op_margin"), 1),
                fmt_pct(r.get("ni_margin"), 1),
                fmt_pct(r.get("rev_yoy"), 1),
                fmt_pct(r.get("eps_yoy"), 1),
            ])
        story.append(make_table(f_table))
    else:
        story.append(Paragraph("No fundamentals available.", body_style))

    # Metrics grid (2-column “modern” table)
    story.append(Paragraph("FINANCIAL METRICS SNAPSHOT (FinancialDatasets.ai)", h_style))
    metric_pairs = report.get("metrics_pairs") or []
    if metric_pairs:
        # Build as 2-up grid: label/value | label/value
        rows = [["Metric", "Value", "Metric", "Value"]]
        pairs = metric_pairs[:]
        # pad to even
        if len(pairs) % 2 == 1:
            pairs.append(("", ""))
        for i in range(0, len(pairs), 2):
            a = pairs[i]
            b = pairs[i + 1]
            rows.append([a[0], a[1], b[0], b[1]])
        story.append(make_table(rows, col_widths=[140, 110, 140, 110]))
    else:
        story.append(Paragraph("No metrics snapshot available.", body_style))

    # Analyst estimates
    story.append(Paragraph("ANALYST ESTIMATES (Annual)", h_style))
    analyst = report.get("analyst") or []
    if analyst:
        a_table = [["Fiscal Period", "EPS Estimate"]]
        for e in analyst:
            a_table.append([e.get("fiscal_period", "N/A"), fmt_number(e.get("earnings_per_share"), 4)])
        story.append(make_table(a_table, col_widths=[220, 280]))
    else:
        story.append(Paragraph("No analyst estimates available.", body_style))

    # News
    story.append(Paragraph("LATEST NEWS", h_style))
    news = report.get("news") or []
    if news:
        for n in news[:6]:
            story.append(Paragraph(f"{n.get('date','N/A')} – {n.get('title','N/A')}", body_style))
            story.append(Paragraph(str(n.get("url", "N/A")), body_style))
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("No news available.", body_style))

    # AI
    if report.get("ai_text"):
        story.append(PageBreak())
        story.append(Paragraph("AI ANALYSIS", h_style))
        for line in str(report["ai_text"]).split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 6))
            else:
                safe = (
                    line.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                )
                story.append(Paragraph(safe, body_style))

    # Charts
    chart = report.get("chart_path")
    if chart and os.path.exists(chart):
        story.append(PageBreak())
        story.append(Paragraph("CHARTS & VISUALS", h_style))
        story.append(Spacer(1, 10))
        img = Image(chart)
        img._restrictSize(6.7 * inch, 9.2 * inch)
        story.append(img)

    doc.build(story)


# ==========================================================
# LOOKUP (SEPARATE PAGE)
# ==========================================================
def lookup_tickers(q: str) -> List[Dict[str, Any]]:
    """
    Best-effort lookup. If FD adds a formal endpoint later, swap it here.
    For now: tries a Yahoo search only if yfinance enabled; otherwise returns empty.
    """
    q = (q or "").strip()
    if not q:
        return []

    # If yfinance is not enabled, we return empty list to avoid fragile dependencies.
    if not yf:
        return []

    try:
        # yfinance has an internal search, but it's not stable across versions.
        # We'll do a minimal safe attempt:
        import requests as _rq
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        r = _rq.get(url, params={"q": q, "quotesCount": 10, "newsCount": 0}, timeout=10)
        r.raise_for_status()
        data = r.json()
        out = []
        for it in (data.get("quotes") or []):
            out.append({
                "symbol": it.get("symbol"),
                "name": it.get("shortname") or it.get("longname") or "",
                "exchange": it.get("exchange") or it.get("exchDisp") or "",
                "type": it.get("quoteType") or "",
            })
        return [x for x in out if x.get("symbol")]
    except Exception as e:
        dbg("lookup error", e)
        return []


# ==========================================================
# PUBLIC ENTRY POINTS
# ==========================================================
def run_single_to_pdf(symbol: str, out_dir: str) -> Tuple[str, Dict[str, Any]]:
    symbol = symbol.upper()
    os.makedirs(out_dir, exist_ok=True)

    snapshot, snap_debug = build_stock_snapshot(symbol)
    fm_snapshot, fm_err = fetch_financial_metrics_snapshot(symbol)
    financials, fin_err = fetch_financials(symbol, limit=10)
    analyst, analyst_err = fetch_analyst_estimates(symbol)
    news, news_err = fetch_news(symbol, limit=6)

    fundamentals_rows = build_fundamentals_rows(financials)
    metrics_pairs = metrics_grid_pairs(fm_snapshot)

    # AI
    ai_text = None
    try:
        if ai_enabled():
            ai_text = generate_ai_single(snapshot, fm_snapshot, analyst)
    except Exception as e:
        ai_text = f"AI Error: {e}"

    chart_path = None
    try:
        chart_path = build_single_charts(symbol)
    except Exception as e:
        dbg("chart error", e)

    title_line = f"{snapshot.get('symbol')} – {snapshot.get('company_name')}"
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(out_dir, f"{symbol}_{ts}.pdf")

    report: Dict[str, Any] = {
        "mode": "single",
        "title_line": title_line,
        "snapshot": snapshot,
        "snapshot_debug": snap_debug,
        "fm_snapshot": fm_snapshot,
        "fm_error": fm_err,
        "fundamentals_rows": fundamentals_rows,
        "financials_error": fin_err,
        "metrics_pairs": metrics_pairs,
        "analyst": analyst or [],
        "analyst_error": analyst_err,
        "news": news or [],
        "news_error": news_err,
        "ai_text": ai_text,
        "chart_path": chart_path,
    }

    export_pdf(report, pdf_path)
    return pdf_path, report


def run_compare_to_pdf(s1: str, s2: str, out_dir: str) -> Tuple[str, Dict[str, Any]]:
    s1 = s1.upper()
    s2 = s2.upper()
    os.makedirs(out_dir, exist_ok=True)

    snap1, dbg1 = build_stock_snapshot(s1)
    snap2, dbg2 = build_stock_snapshot(s2)

    fm1, fm1_err = fetch_financial_metrics_snapshot(s1)
    fm2, fm2_err = fetch_financial_metrics_snapshot(s2)

    ai_text = None
    try:
        if ai_enabled():
            ai_text = generate_ai_compare(snap1, snap2, fm1, fm2)
    except Exception as e:
        ai_text = f"AI Error: {e}"

    title_line = f"{ai_company_label(snap1)} vs {ai_company_label(snap2)}"
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(out_dir, f"{s1}_{s2}_{ts}.pdf")

    report = {
        "mode": "compare",
        "title_line": title_line,
        "left": {"snapshot": snap1, "snapshot_debug": dbg1, "fm": fm1, "fm_error": fm1_err},
        "right": {"snapshot": snap2, "snapshot_debug": dbg2, "fm": fm2, "fm_error": fm2_err},
        "ai_text": ai_text,
    }

    # For compare PDF, keep it simple: snapshot + metrics + AI
    export_pdf_compare(report, pdf_path)
    return pdf_path, report


def export_pdf_compare(report: Dict[str, Any], output_path: str) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_path, pagesize=letter, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)

    title_style = ParagraphStyle("T", fontName="Helvetica-Bold", fontSize=18, alignment=TA_CENTER, textColor=colors.white, spaceAfter=14)
    h_style = ParagraphStyle("H", fontName="Helvetica-Bold", fontSize=12, textColor=colors.white, spaceBefore=12, spaceAfter=8)
    body_style = ParagraphStyle("B", fontName="Helvetica", fontSize=9, textColor=colors.white, leading=12)

    story = []
    story.append(Paragraph("John Pirisino's Stock Analyzer", title_style))
    story.append(Paragraph(report.get("title_line", ""), ParagraphStyle("S", fontName="Helvetica", fontSize=10, alignment=TA_CENTER, textColor=colors.HexColor("#9ca3af"))))

    left = report["left"]["snapshot"]
    right = report["right"]["snapshot"]

    story.append(Paragraph("SNAPSHOT (Side-by-Side)", h_style))
    grid = [
        ["Field", left.get("symbol", ""), right.get("symbol", "")],
        ["Company", left.get("company_name", ""), right.get("company_name", "")],
        ["Sector", left.get("sector", ""), right.get("sector", "")],
        ["Industry", left.get("industry", ""), right.get("industry", "")],
        ["Price", fmt_number(left.get("current_price"), 2), fmt_number(right.get("current_price"), 2)],
        ["1Y %", f"{fmt_number(left.get('change_1y_pct'),2)}%", f"{fmt_number(right.get('change_1y_pct'),2)}%"],
    ]
    story.append(make_table(grid, col_widths=[140, 180, 180]))

    story.append(Paragraph("FINANCIAL METRICS SNAPSHOT (Key)", h_style))
    fm1 = report["left"]["fm"]
    fm2 = report["right"]["fm"]
    rows = [["Metric", left.get("symbol",""), right.get("symbol","")]]

    def add_row(label, k, fmt="num4"):
        v1 = fm1.get(k) if fm1 else None
        v2 = fm2.get(k) if fm2 else None
        if fmt == "int":
            rows.append([label, fmt_int(v1), fmt_int(v2)])
        elif fmt == "pct":
            rows.append([label, fmt_pct(v1, 1), fmt_pct(v2, 1)])
        else:
            rows.append([label, fmt_number(v1, 4), fmt_number(v2, 4)])

    add_row("Market Cap", "market_cap", "int")
    add_row("Enterprise Value", "enterprise_value", "int")
    add_row("P/E", "price_to_earnings_ratio")
    add_row("EV/EBITDA", "enterprise_value_to_ebitda_ratio")
    add_row("Gross Margin", "gross_margin", "pct")
    add_row("Operating Margin", "operating_margin", "pct")
    add_row("ROE", "return_on_equity", "pct")
    add_row("Debt/Equity", "debt_to_equity")

    story.append(make_table(rows, col_widths=[180, 160, 160]))

    if report.get("ai_text"):
        story.append(PageBreak())
        story.append(Paragraph("AI ANALYSIS", h_style))
        for line in str(report["ai_text"]).split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 6))
            else:
                safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(safe, body_style))

    doc.build(story)

