# app.py
import os
import uuid
import threading
import requests

from flask import (
    Flask, render_template, request, redirect,
    url_for, jsonify, send_file
)

from engine import run_single_to_pdf, run_compare_to_pdf, build_single_report_data

app = Flask(__name__)

# ---------------------------------------
# ENVIRONMENT
# ---------------------------------------
SQUARE_ACCESS_TOKEN = os.getenv("SQUARE_ACCESS_TOKEN")
SQUARE_LOCATION_ID  = os.getenv("SQUARE_LOCATION_ID")
SQUARE_BASE_URL     = os.getenv("SQUARE_BASE_URL", "https://connect.squareup.com")
USE_SQUARE          = os.getenv("SQUARE", "N").upper() == "Y"   # <— controls payment

# ---------------------------------------
# JOB STORE
# ---------------------------------------
jobs = {}
# job_id -> {
#   "status": pending|running|done|error,
#   "file_path": str|None,
#   "error": str|None,
#   "mode": "single"|"compare",
#   "ticker1": str,
#   "ticker2": str
# }

# ---------------------------------------
# OPTIONAL: CREATE PAYMENT LINK (Square)
# ---------------------------------------
def create_payment_link(amount_cents: int, description: str, redirect_url: str) -> str:
    """
    Uses Square HTTP API directly (no SDK).
    Only called when USE_SQUARE is True.
    """
    if not (SQUARE_ACCESS_TOKEN and SQUARE_LOCATION_ID):
        raise RuntimeError("Square is enabled but ACCESS_TOKEN or LOCATION_ID is missing.")

    url = f"{SQUARE_BASE_URL}/v2/online-checkout/payment-links"

    headers = {
        "Square-Version": "2024-01-18",
        "Authorization": f"Bearer {SQUARE_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    body = {
        "idempotency_key": str(uuid.uuid4()),
        "order": {
            "location_id": SQUARE_LOCATION_ID,
            "line_items": [
                {
                    "name": description,
                    "quantity": "1",
                    "base_price_money": {
                        "amount": amount_cents,
                        "currency": "USD",
                    },
                }
            ],
        },
        "checkout_options": {
            "redirect_url": redirect_url
        },
    }

    resp = requests.post(url, json=body, headers=headers, timeout=20)
    data = resp.json()

    if resp.status_code == 200:
        return data["payment_link"]["url"]

    raise RuntimeError(f"Square error {resp.status_code}: {data}")


# ---------------------------------------
# BACKGROUND JOB RUNNER
# ---------------------------------------
def run_job(job_id: str):
    job = jobs[job_id]
    job["status"] = "running"

    try:
        out_dir = os.path.join(os.getcwd(), "generated_reports")
        os.makedirs(out_dir, exist_ok=True)

        if job["mode"] == "single":
            fp = run_single_to_pdf(job["ticker1"], out_dir)
        else:
            fp = run_compare_to_pdf(job["ticker1"], job["ticker2"], out_dir)

        job["file_path"] = fp
        job["status"] = "done"

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


# ---------------------------------------
# ROUTES
# ---------------------------------------

@app.route("/", methods=["GET"])
def index():
    # Do NOT pre-populate last-used tickers
    return render_template("index.html", error=None)

    report = build_single_report_data(symbol)
    return render_template("report.html", report=report)

@app.route("/view-report")
def view_report():
    symbol = request.args.get("symbol", "").upper()
    if not symbol:
        return "Missing symbol", 400

@app.route("/create_checkout", methods=["POST"])
def create_checkout():
    ticker1 = request.form.get("ticker1", "").upper().strip()
    ticker2 = request.form.get("ticker2", "").upper().strip()

    if not ticker1:
        return render_template("index.html", error="Ticker 1 is required.")

    # Decide mode & price
    if ticker2:
        mode = "compare"
        amount = 1500   # $15.00
        desc = f"Two-ticker analysis: {ticker1} vs {ticker2}"
    else:
        mode = "single"
        amount = 1000   # $10.00
        desc = f"Single-ticker analysis: {ticker1}"

    # Create job
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "file_path": None,
        "error": None,
        "mode": mode,
        "ticker1": ticker1,
        "ticker2": ticker2,
    }

    # If Square is disabled, go straight to processing
    if not USE_SQUARE:
        return redirect(url_for("processing", job_id=job_id))

    # If Square enabled, redirect user to payment page
    redirect_url = url_for("processing", job_id=job_id, _external=True)
    try:
        pay_url = create_payment_link(amount_cents=amount, description=desc, redirect_url=redirect_url)
    except Exception as e:
        # On error, show message and allow retry
        return render_template("index.html", error=f"Payment error: {e}")

    return redirect(pay_url)


@app.route("/processing/<job_id>")
def processing(job_id):
    job = jobs.get(job_id)
    if not job:
        return "Invalid job id.", 404

    if job["status"] == "pending":
        # Kick off background thread
        threading.Thread(target=run_job, args=(job_id,), daemon=True).start()

    return render_template("processing.html", job_id=job_id)


@app.route("/job_status/<job_id>")
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"status": "unknown"}), 404
    return jsonify({"status": job["status"], "error": job["error"]})


@app.route("/ready/<job_id>")
def ready(job_id):
    job = jobs.get(job_id)
    if not job:
        return "Invalid job id.", 404
    if job["status"] != "done":
        return redirect(url_for("processing", job_id=job_id))
    filename = os.path.basename(job["file_path"])
    return render_template("ready.html", job_id=job_id, filename=filename)


@app.route("/analyze", methods=["POST"])
def analyze():
    symbol = request.form["symbol"].upper()

    # Build report ONCE
    report = build_single_report_data(symbol)

    # Store in session for PDF download
    session["last_report"] = report

    return render_template(
        "report.html",
        report=report
    )


# ---------------------------------------
# TICKER LOOKUP PAGE
# ---------------------------------------

YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"

@app.route("/lookup", methods=["GET"])
def lookup():
    query = request.args.get("q", "").strip()
    results = []
    error = None

    if query:
        try:
            resp = requests.get(
                YAHOO_SEARCH_URL,
                params={"q": query, "quotesCount": 10, "newsCount": 0},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                for q in data.get("quotes", []):
                    results.append({
                        "symbol": q.get("symbol"),
                        "shortname": q.get("shortname") or q.get("longname") or "",
                        "exch": q.get("exchDisp") or "",
                        "type": q.get("quoteType") or "",
                    })
            else:
                error = f"Yahoo search error {resp.status_code}"
        except Exception as e:
            error = f"Lookup error: {e}"

    return render_template("lookup.html", query=query, results=results, error=error)


if __name__ == "__main__":
    # Railway will use gunicorn via Procfile, but this lets you test locally.
    app.run(debug=True)


