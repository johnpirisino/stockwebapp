import os
import uuid
import traceback
from datetime import datetime
from typing import Dict, Any

from flask import Flask, render_template, request, redirect, url_for, send_file, abort, flash

from engine import (
    run_single_to_pdf,
    run_compare_to_pdf,
    lookup_tickers,
)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-me")

# In-memory job store (OK for 1 Railway container)
# If you scale to multiple replicas, move this to Redis or a DB.
REPORT_CACHE: Dict[str, Dict[str, Any]] = {}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SQUARE = os.getenv("SQUARE", "N").upper() == "Y"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/lookup", methods=["GET", "POST"])
def lookup():
    results = []
    q = ""
    err = None

    if request.method == "POST":
        q = (request.form.get("q") or "").strip()
        if not q:
            err = "Enter a company name or ticker to search."
        else:
            try:
                results = lookup_tickers(q)
            except Exception as e:
                err = f"Lookup failed: {e}"

    return render_template("lookup.html", results=results, q=q, err=err)


@app.route("/generate", methods=["POST"])
def generate():
    """
    Generates PDF + stores structured report data, then redirects to /ready/<job_id>
    """
    mode = (request.form.get("mode") or "single").lower()
    ticker1 = (request.form.get("ticker1") or "").strip().upper()
    ticker2 = (request.form.get("ticker2") or "").strip().upper()

    if mode not in ("single", "compare"):
        mode = "single"

    if mode == "single" and not ticker1:
        flash("Please enter a ticker.", "error")
        return redirect(url_for("index"))

    if mode == "compare" and (not ticker1 or not ticker2):
        flash("Please enter both tickers for a comparison.", "error")
        return redirect(url_for("index"))

    # Optional: payment gate (your toggle request)
    if SQUARE:
        try:
            from square_payment import ensure_payment_ok  # you can implement this
            ok, msg = ensure_payment_ok(request)
            if not ok:
                flash(msg or "Payment required.", "error")
                return redirect(url_for("index"))
        except Exception:
            # Don't crash if square_payment isn't ready yet
            flash("Payment module error (SQUARE=Y). Check square_payment.py.", "error")
            return redirect(url_for("index"))

    job_id = str(uuid.uuid4())

    try:
        if mode == "single":
            pdf_path, report = run_single_to_pdf(ticker1, OUTPUT_DIR)
        else:
            pdf_path, report = run_compare_to_pdf(ticker1, ticker2, OUTPUT_DIR)

        report["job_id"] = job_id
        report["pdf_path"] = pdf_path
        report["generated_at"] = datetime.utcnow().isoformat() + "Z"
        REPORT_CACHE[job_id] = report

        return redirect(url_for("ready", job_id=job_id))

    except Exception:
        err = traceback.format_exc()
        print("❌ Generate error:\n", err, flush=True)
        flash("Internal error generating report. Check Railway logs.", "error")
        return redirect(url_for("index"))


@app.route("/ready/<job_id>", methods=["GET"])
def ready(job_id: str):
    report = REPORT_CACHE.get(job_id)
    if not report:
        abort(404)
    return render_template("report.html", report=report)


@app.route("/pdf/<job_id>", methods=["GET"])
def pdf(job_id: str):
    report = REPORT_CACHE.get(job_id)
    if not report:
        abort(404)

    pdf_path = report.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        abort(404)

    return send_file(pdf_path, as_attachment=True)


@app.route("/health", methods=["GET"])
def health():
    return {"ok": True}


if __name__ == "__main__":
    # Railway sets PORT
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
