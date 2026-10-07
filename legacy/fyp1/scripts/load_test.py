"""
load_test.py — simple throughput/latency benchmark for /add_transaction

Usage:
    1. Start the app normally in one terminal:
         python app.py
    2. In a second terminal, run this script:
         python load_test.py

This does NOT use Locust or any external load-testing framework — it's a plain
Python script using `requests` + `time.time()`, intended to produce one honest,
reproducible throughput number for your progress report instead of the
proposal's unverified "10,000 TPS" claim.

Your app has CSRFProtect enabled globally, so this script fetches a fresh
CSRF token from the relevant GET page before every POST (login, and each
add_transaction submission), exactly like a real browser would.

Adjust NUM_REQUESTS, BASE_URL, and the login credentials below to match your
local setup before running.
"""

import re
import time
import statistics
from datetime import datetime, timedelta
import requests

# ---- Configuration -----------------------------------------------------
BASE_URL = "http://127.0.0.1:5000"
LOGIN_EMAIL = "loadtest@example.com"
LOGIN_PASSWORD = "LoadTest123!"
NUM_REQUESTS = 100
# --------------------------------------------------------------------------

CSRF_RE = re.compile(r'name="csrf_token"\s+value="([^"]+)"')


def get_csrf_token(session, page_url):
    """Fetches a page and extracts the csrf_token hidden-input value from it."""
    resp = session.get(page_url)
    match = CSRF_RE.search(resp.text)
    return match.group(1) if match else None


def login(session):
    """Logs in and returns True if the session now holds a valid cookie."""
    token = get_csrf_token(session, f"{BASE_URL}/login")
    if not token:
        print("Could not find csrf_token on /login page — check that your "
              "login.html template includes {{ form.csrf_token }} or a hidden "
              "input named csrf_token.")
        return False

    resp = session.post(
        f"{BASE_URL}/login",
        data={"email": LOGIN_EMAIL, "password": LOGIN_PASSWORD, "csrf_token": token},
        allow_redirects=True,
    )
    return resp.status_code == 200 and "/login" not in resp.url


def run_load_test():
    session = requests.Session()

    print(f"Logging in as {LOGIN_EMAIL} ...")
    if not login(session):
        print("Login failed — double-check LOGIN_EMAIL / LOGIN_PASSWORD, and "
              "make sure that user exists in your real database (not the test DB).")
        return

    print(f"Logged in. Firing {NUM_REQUESTS} requests at /add_transaction ...\n")

    latencies = []
    failures = 0

    overall_start = time.time()

    for i in range(NUM_REQUESTS):
        # Fetch a fresh CSRF token for each submission, same as a real browser
        # loading the add_transaction form before submitting it.
        token = get_csrf_token(session, f"{BASE_URL}/add_transaction")
        if not token:
            failures += 1
            continue

        timestamp = (datetime.now() - timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M")

        payload = {
            "amount": 100 + (i % 50),
            "type": "deposit" if i % 2 == 0 else "withdraw",
            "device_id": f"device_{i % 10}",
            "timestamp": timestamp,
            "csrf_token": token,
        }

        req_start = time.time()
        try:
            resp = session.post(f"{BASE_URL}/add_transaction", data=payload)
            req_end = time.time()

            if resp.status_code >= 400:
                failures += 1
            else:
                latencies.append(req_end - req_start)

        except requests.exceptions.RequestException as e:
            failures += 1
            print(f"Request {i} failed: {e}")

    overall_end = time.time()
    total_time = overall_end - overall_start

    # ---- Report -----------------------------------------------------
    print("\n--- Load Test Results ---")
    print(f"Total requests attempted:  {NUM_REQUESTS}")
    print(f"Successful requests:       {len(latencies)}")
    print(f"Failed requests:           {failures}")
    print(f"Total wall-clock time:     {total_time:.3f} s")

    if latencies:
        print(f"Average latency:           {statistics.mean(latencies) * 1000:.2f} ms")
        print(f"Median latency:             {statistics.median(latencies) * 1000:.2f} ms")
        print(f"Min latency:                {min(latencies) * 1000:.2f} ms")
        print(f"Max latency:                {max(latencies) * 1000:.2f} ms")
        print(f"Throughput:                 {len(latencies) / total_time:.2f} requests/sec")
    else:
        print("No successful requests — check that the app is running and that "
              "LOGIN_EMAIL / LOGIN_PASSWORD are correct.")


if __name__ == "__main__":
    run_load_test()
