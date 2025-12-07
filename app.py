# app.py
import os
import uuid
import threading
import requests
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect,
    url_for, jsonify, send_file
)

from engine import run_single_to_pdf, run_compare_to_pdf

app = Flask(__name__)

# ---------------------------------------
# LOAD ENVIRONMENT VALUES
# ---------------------------------------
SQUARE_ACCESS_TOKEN = os.getenv("SQUARE_ACCESS_TOKEN")
SQUARE_LOCATION_ID  = os.getenv("SQUARE_LOCATION_ID")
SQUARE_BASE_URL     = os.getenv("SQUARE_BASE_URL")  
SQUARE_ENV          = "production"

print("Using Square environment:", SQUARE_ENV)
print("Square Base URL:", SQUARE_BASE_URL)

# ---------------------------------------
# CREATE PAYMENT LINK
# ---------------------------------------
def create_payment_link(amount_cents: int, description: str, redirect_url: str):
    url = f"{SQUARE_BASE_URL}/v2/online-checkout/payment-links"

    headers = {
        "Square-Version": "2024-01-18",
        "Authorization": f"Bearer {SQUARE_ACCESS_TOKEN}",
        "Content-Type": "application/json"
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
                        "currency": "USD"
                    }
                }
            ]
        },
        "checkout_options": {
            "redirect_url": redirect_url
        }
    }

    response = requests.post(url, json=body, headers=headers)
    data = response.json()

    if response.status_code == 200:
        return data["payment_link"]["url"]
    else:
        raise RuntimeError(f"Square checkout error {response.status_code}: {data}")

# ---------------------------------------
# JOB STORE
# ---------------------------------------
jobs = {}

def run_job(job_id: str):
    job = jobs[job_id]
    job["status"] = "running"

    try:
        out_dir = os.path.join(os.getcwd(), "generated_reports")
        os.makedirs(out_dir, exist_ok=True)

        if job["mode"] == "single":
            file_path = run_single_to_pdf(job["ticker1"], out_dir)
        else:
            file_path = run_compare_to_pdf(job["ticker1"], job["ticker2"], out_dir)

        job["file_path"] = file_path
        job["status"] = "done"

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)

# ---------------------------------------
# ROUTES
# ---------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/create_checkout", methods=["POST"])
def create_checkout_route():
    ticker1 = request.form.get("ticker1", "").upper().strip()
    ticker2 = request.form.get("ticker2", "").upper().strip()

    if not ticker1:
        return render_template("index.html", error="Ticker 1 is required.")

    if ticker2:
        mode = "compare"
        amount = 1500
        desc = f"Two-ticker analysis: {ticker1} vs {ticker2}"
    else:
        mode = "single"
        amount = 1000
        desc = f"Single-ticker analysis: {ticker1}"

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "file_path": None,
        "error": None,
        "mode": mode,
        "ticker1": ticker1,
        "ticker2": ticker2
    }

    success_redirect = url_for("processing", job_id=job_id, _external=True)

    try:
        pay_url = create_payment_link(amount, desc, success_redirect)
    except Exception as e:
        return render_template("index.html", error=str(e))

    return redirect(pay_url)

@app.route("/processing/<job_id>")
def processing(job_id):
    if job_id not in jobs:
        return "Invalid job ID", 404

    if jobs[job_id]["status"] == "pending":
        threading.Thread(target=run_job, args=(job_id,), daemon=True).start()

    return render_template("processing.html", job_id=job_id)

@app.route("/job_status/<job_id>")
def job_status(job_id):
    if job_id not in jobs:
        return jsonify({"status": "unknown"}), 404

    return jsonify({
        "status": jobs[job_id]["status"],
        "error": jobs[job_id]["error"]
    })

# ---------------------------------------------------
# NEW — READY PAGE (this prevents auto-download)
# ---------------------------------------------------
@app.route("/ready/<job_id>")
def ready(job_id):
    job = jobs.get(job_id)
    if not job:
        return "Invalid job ID", 404
    if job["status"] != "done":
        return "Report not finished yet", 400

    return render_template("ready.html", job_id=job_id)

@app.route("/download/<job_id>")
def download(job_id):
    if job_id not in jobs:
        return "Invalid job ID", 404

    job = jobs[job_id]
    if job["status"] != "done":
        return "File not ready", 400

    return send_file(
        job["file_path"],
        as_attachment=True,
        download_name=os.path.basename(job["file_path"])
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

