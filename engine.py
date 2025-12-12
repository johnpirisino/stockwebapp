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

# =========================================================
# 🔒 EVERYTHING ELSE UNCHANGED
# =========================================================
# All FD calls, pricing logic, snapshot logic,
# PDF generation, charts, and public entry points
# remain exactly as in your last working version.
