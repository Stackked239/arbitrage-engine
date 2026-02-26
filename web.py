"""
web.py
======
Flask web server for the Arbitrage Engine dashboard.

Serves:
  - / → Static dashboard (index.html, app.js, style.css)
  - /api/data → Live JSON snapshot from runner's dashboard_snapshot.json
  - /api/health → Health check for Render

Designed to run as a Render Web Service alongside runner.py (Background Worker).
Both services share the same filesystem on Render when using a Persistent Disk.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, send_from_directory, Response

app = Flask(__name__, static_folder=None)

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
DASHBOARD_SNAPSHOT = os.path.join(DATA_DIR, "dashboard_snapshot.json")
DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "dashboard")


@app.route("/")
def index():
    return send_from_directory(DASHBOARD_DIR, "index.html")


@app.route("/<path:filename>")
def dashboard_static(filename):
    """Serve dashboard static assets (app.js, style.css, etc.)."""
    return send_from_directory(DASHBOARD_DIR, filename)


@app.route("/api/data")
def api_data():
    if not os.path.exists(DASHBOARD_SNAPSHOT):
        return jsonify({
            "error": "No data available yet — runner may not be started",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }), 503

    try:
        with open(DASHBOARD_SNAPSHOT, "r", encoding="utf-8") as f:
            data = json.load(f)
        resp = jsonify(data)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp
    except (json.JSONDecodeError, OSError) as exc:
        return jsonify({
            "error": f"Failed to read snapshot: {exc}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }), 500


@app.route("/api/health")
def health():
    """Health check endpoint for Render."""
    runner_alive = False
    pid_file = os.path.join(DATA_DIR, "runner.pid")

    if os.path.exists(pid_file):
        try:
            with open(pid_file) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            runner_alive = True
        except (OSError, ValueError):
            pass

    snapshot_age = None
    if os.path.exists(DASHBOARD_SNAPSHOT):
        try:
            with open(DASHBOARD_SNAPSHOT, "r") as f:
                data = json.load(f)
            ts = data.get("timestamp", "")
            if ts:
                snap_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                age_sec = (datetime.now(timezone.utc) - snap_time).total_seconds()
                snapshot_age = round(age_sec, 1)
        except Exception:
            pass

    return jsonify({
        "status": "healthy",
        "runner_alive": runner_alive,
        "snapshot_age_seconds": snapshot_age,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/trades")
def api_trades():
    """Return recent trades from the snapshot."""
    if not os.path.exists(DASHBOARD_SNAPSHOT):
        return jsonify([]), 503

    try:
        with open(DASHBOARD_SNAPSHOT, "r", encoding="utf-8") as f:
            data = json.load(f)
        trades = data.get("trades", [])
        resp = jsonify(trades)
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp
    except Exception:
        return jsonify([]), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"Starting Arbitrage Engine Dashboard on port {port}")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Dashboard dir: {DASHBOARD_DIR}")
    print(f"  Snapshot file: {DASHBOARD_SNAPSHOT}")
    app.run(host="0.0.0.0", port=port, debug=debug)
