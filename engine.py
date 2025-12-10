# ============================================
# engine.py — DEBUG DATA COLLECTION VERSION
# ============================================
import os
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import requests
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

FD_API_KEY = os.getenv("FINANCIAL_DATASETS_API_KEY")
FD_BASE_URL = "https://api.financialdatasets.ai"


# =================================================
# DEBUG LOGGER
# =================================================
def debug_log(name, value):
    """Print readable debug section to Railway logs."""
    print("\n========== DEBUG:", name, "==========")
    try:
        print(json.dumps(value, indent=2))
    except Exception:
        print(value)
    print("=====================================\n")


def fd_headers():
    if not FD_API_KEY:
        return {}
    return {"X-API-KEY": FD_API_KEY}


# =================================================
# FINANCIAL DATASETS CALLS (RAW)
# =================================================

def fd_call(name: str, endpoint: str, params=None):
    """Helper wrapper that logs everything returned from FD API."""
    if params is None:
        params = {}

    url = f"{FD_BASE_URL}/{endpoint}"

    try:
        r = requests.get(url, headers=fd_headers(), params=params, timeout=20)
        data = {}

        try:
            data = r.json()
        except Exception:
            data = {"error": "Failed to parse JSON", "raw_text": r.text}

        debug_log(name + f" (status={r.status_code})", data)
        return data, None

    except Exception as e:
        debug_log(name + " EXCEPTION", str(e))
        return None, str(e)


# =================================================
# PUBLIC ENTRYPOINTS (TEMPORARY, DEBUG-ONLY)
# =================================================

def run_single_to_pdf(symbol: str, out_dir: str) -> str:
    """
    *** TEMP DEBUG VERSION ***
    Does not generate PDF.
    Only calls the APIs and prints their exact payloads.
    """
    symbol = symbol.upper()

    # --- Snapshot
    fm_snapshot, _ = fd_call("FM SNAPSHOT", "financial-metrics/snapshot", {"ticker": symbol})

    # --- History
    fm_history, _ = fd_call("FM HISTORY", "financial-metrics", {"ticker": symbol, "period": "annual", "limit": 10})

    # --- Facts
    company_facts, _ = fd_call("COMPANY FACTS", "company/facts", {"ticker": symbol})

    # --- Financials
    financials, _ = fd_call("FINANCIALS", "financials", {"ticker": symbol, "period": "annual", "limit": 10})

    # --- News
    news, _ = fd_call("NEWS", "news", {"ticker": symbol, "limit": 5})

    # --- Insider Trades
    insider, _ = fd_call("INSIDER", "insider-trades", {"ticker": symbol, "limit": 20})

    # --- Institutional
    inst, _ = fd_call("INSTITUTIONAL", "institutional-ownership", {"ticker": symbol, "limit": 100})

    # ============================================================
    # RETURN A FAKE FILE PATH — Flask expects *some* return value
    # ============================================================
    fake_path = os.path.join(out_dir, f"{symbol}_DEBUG_ONLY.txt")
    os.makedirs(out_dir, exist_ok=True)
    with open(fake_path, "w") as f:
        f.write("DEBUG ENGINE — NO PDF GENERATED\n")

    return fake_path


def run_compare_to_pdf(s1: str, s2: str, out_dir: str) -> str:
    """
    TEMP DEBUG VERSION – Always return a dummy file.
    """
    fake_path = os.path.join(out_dir, f"{s1}_{s2}_DEBUG_ONLY.txt")
    os.makedirs(out_dir, exist_ok=True)
    with open(fake_path, "w") as f:
        f.write("DEBUG ENGINE — NO COMPARE PDF\n")

    return fake_path
