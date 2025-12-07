# square_payment.py
import os
import uuid
import requests

SQUARE_ACCESS_TOKEN = os.getenv("SQUARE_ACCESS_TOKEN")
SQUARE_LOCATION_ID = os.getenv("SQUARE_LOCATION_ID")
SQUARE_BASE_URL = os.getenv("SQUARE_BASE_URL", "https://connect.squareupsandbox.com")


def create_payment_link(amount_cents: int, description: str, redirect_url: str) -> str:
    """
    Create a Square Online Checkout payment link (sandbox).
    Uses /v2/online-checkout/payment-links.
    """
    if not SQUARE_ACCESS_TOKEN or not SQUARE_LOCATION_ID:
        raise RuntimeError("Missing SQUARE_ACCESS_TOKEN or SQUARE_LOCATION_ID in environment.")

    url = f"{SQUARE_BASE_URL}/v2/online-checkout/payment-links"

    headers = {
        "Authorization": f"Bearer {SQUARE_ACCESS_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        # Square version – you can adjust if needed
        "Square-Version": "2024-01-18",
    }

    payload = {
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
            "redirect_url": redirect_url,
        },
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    if resp.status_code >= 400:
        raise RuntimeError(f"Square error {resp.status_code}: {resp.text}")

    data = resp.json()
    link = data.get("payment_link", {}).get("url")
    if not link:
        raise RuntimeError(f"Payment link URL missing in Square response: {data}")

    return link

