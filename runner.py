"""
runner.py
=========
Continuous arbitrage scanner — runs engines back-to-back as fast as
API rate limits allow.  No fixed schedule; each cycle starts the moment
the previous one finishes (plus a short cooldown to stay under rate caps).

Designed to be launched once and left running:
    nohup python3 runner.py &

Writes scan results to data/dashboard_snapshot.json and
data/paper_trades.json after every cycle.  Also appends a one-line
summary to data/cycle_log.csv for historical tracking.

Ctrl+C to stop gracefully.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Set

# ── Project imports ──────────────────────────────────────────────────────────
from config import CAPITAL, ENGINE_NAMES, PAPER_TRADING, TOTAL_CAPITAL, DATA_DIR, get_logger
from paper_trader import PaperTrader

logger = get_logger("runner")

os.makedirs(DATA_DIR, exist_ok=True)
DASHBOARD_FILE = os.path.join(DATA_DIR, "dashboard_snapshot.json")
CYCLE_LOG = os.path.join(DATA_DIR, "cycle_log.csv")
PID_FILE = os.path.join(DATA_DIR, "runner.pid")

# Minimum seconds between cycle starts
MIN_CYCLE_GAP = 180  # 3 minutes
DASHBOARD_DATA = os.path.join(os.path.dirname(__file__), "dashboard", "data.json")


# ── Lazy engine imports ──────────────────────────────────────────────────────

def _import_engine1():
    from engine1_funding_rate import FundingRateScanner
    return FundingRateScanner

def _import_engine2():
    from engine2_polymarket import PolymarketFarmer
    return PolymarketFarmer

def _import_engine3():
    from engine3_flash_loan import FlashLoanScanner
    return FlashLoanScanner

def _import_engine4():
    from engine4_triangular import TriangularArbitrage
    return TriangularArbitrage

def _import_engine5():
    from engine5_cross_exchange import CrossExchangeScanner
    return CrossExchangeScanner


_ENGINE_IMPORTERS = {1: _import_engine1, 2: _import_engine2, 3: _import_engine3,
                     4: _import_engine4, 5: _import_engine5}
_ENGINE_IDS = {1: "engine1_funding_rate", 2: "engine2_polymarket",
               3: "engine3_flash_loan", 4: "engine4_triangular",
               5: "engine5_cross_exchange"}


# ── Snapshot save ────────────────────────────────────────────────────────────

def save_snapshot(pt: PaperTrader, engine_results: Dict[str, Any]) -> None:
    portfolio = pt.get_portfolio_summary()
    trades = pt.get_trade_history(limit=50)
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paper_trading": PAPER_TRADING,
        "portfolio": portfolio,
        "engine_cycle_results": engine_results,
        "trades": trades,
    }
    tmp = DASHBOARD_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)
    os.replace(tmp, DASHBOARD_FILE)

    try:
        dash_tmp = DASHBOARD_DATA + ".tmp"
        with open(dash_tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, default=str)
        os.replace(dash_tmp, DASHBOARD_DATA)
    except Exception as exc:
        logger.debug("Dashboard data sync failed: %s", exc)


def append_cycle_log(cycle_num: int, elapsed_s: float, total_pnl: float,
                     return_pct: float, trades: int, engine_results: Dict[str, Any]) -> None:
    """Append one row to the CSV cycle log for historical tracking."""
    file_exists = os.path.exists(CYCLE_LOG)
    with open(CYCLE_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "cycle", "elapsed_s", "total_pnl", "return_pct",
                "total_trades", "e1_opps", "e2_markets", "e3_opps",
                "e4_opps", "e5_spreads", "e3_exec", "e5_exec",
            ])
        writer.writerow([
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            cycle_num,
            f"{elapsed_s:.1f}",
            f"{total_pnl:.2f}",
            f"{return_pct:.4f}",
            trades,
            engine_results.get("engine1_funding_rate", {}).get("opportunities_found", 0),
            engine_results.get("engine2_polymarket", {}).get("markets_scanned", 0),
            engine_results.get("engine3_flash_loan", {}).get("opportunities_found", 0),
            engine_results.get("engine4_triangular", {}).get("opportunities_found", 0),
            engine_results.get("engine5_cross_exchange", {}).get("spreads_found", 0),
            engine_results.get("engine3_flash_loan", {}).get("executions_this_cycle", 0),
            engine_results.get("engine5_cross_exchange", {}).get("executions_this_cycle", 0),
        ])


# ── Single cycle ─────────────────────────────────────────────────────────────

def run_cycle(pt: PaperTrader, engines: Dict[int, Any]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    run_order = sorted(engines.keys())

    for num in run_order:
        engine_id = _ENGINE_IDS[num]
        label = ENGINE_NAMES[engine_id]
        try:
            t0 = time.perf_counter()
            if num == 1:
                r = engines[num].run_cycle(auto_enter=True, max_new_positions=3)
            elif num == 2:
                r = engines[num].run_cycle(auto_deploy=True, max_new_deployments=3)
            elif num in (3, 4, 5):
                r = engines[num].run_cycle(auto_execute=True, max_executions=3)
            else:
                r = {}
            r["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            results[engine_id] = r
        except Exception as exc:
            logger.error("Engine %d (%s) failed: %s", num, label, exc, exc_info=True)
            results[engine_id] = {"error": str(exc)}

    return results


# ── Main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE) as f:
                old_pid = int(f.read().strip())
            os.kill(old_pid, 0)
            print(f"[ERROR] Runner already active (PID {old_pid}). Exiting.")
            sys.exit(1)
        except (OSError, ValueError):
            pass

    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    pt = PaperTrader()

    engines: Dict[int, Any] = {}
    for num in range(1, 6):
        engine_id = _ENGINE_IDS[num]
        try:
            cls = _ENGINE_IMPORTERS[num]()
            engines[num] = cls(paper_trader=pt)
            logger.info("Engine %d (%s) loaded.", num, engine_id)
        except Exception as exc:
            logger.error("Failed to load engine %d: %s", num, exc)
            print(f"  [WARNING] Engine {num} failed to load: {exc}")

    if not engines:
        print("[ERROR] No engines loaded. Exiting.")
        sys.exit(1)

    labels = ", ".join(f"E{n}" for n in sorted(engines.keys()))
    print(f"\n{'\u2550' * 60}")
    print(f"  ARBITRAGE ENGINE \u2014 CONTINUOUS RUNNER")
    print(f"  Mode      : {'PAPER TRADING' if PAPER_TRADING else 'LIVE'}")
    print(f"  Capital   : ${TOTAL_CAPITAL:,.2f}")
    print(f"  Engines   : {labels}")
    print(f"  Min gap   : {MIN_CYCLE_GAP}s between cycles")
    print(f"  PID       : {os.getpid()}")
    print(f"{'\u2550' * 60}\n")

    cycle_num = 0

    try:
        while True:
            cycle_num += 1
            cycle_start = time.perf_counter()
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"[Cycle {cycle_num}] {now_str}")

            try:
                engine_results = run_cycle(pt, engines)
            except Exception as exc:
                logger.error("Cycle %d crashed: %s", cycle_num, exc, exc_info=True)
                print(f"  [CYCLE ERROR] {exc}")
                time.sleep(30)
                continue

            try:
                save_snapshot(pt, engine_results)
            except Exception as exc:
                logger.error("Snapshot save failed: %s", exc)

            portfolio = pt.get_portfolio_summary()
            total_pnl = portfolio["total_net_pnl"]
            return_pct = portfolio["total_return_pct"]
            total_trades = portfolio["total_trades"]
            elapsed = time.perf_counter() - cycle_start

            try:
                append_cycle_log(cycle_num, elapsed, total_pnl, return_pct,
                                 total_trades, engine_results)
            except Exception:
                pass

            e3_exec = engine_results.get("engine3_flash_loan", {}).get("executions_this_cycle", 0)
            e5_exec = engine_results.get("engine5_cross_exchange", {}).get("executions_this_cycle", 0)
            print(
                f"  Done in {elapsed:.0f}s | P&L: ${total_pnl:+.2f} ({return_pct:+.2f}%) "
                f"| Trades: {total_trades} | Flash:{e3_exec} Cross:{e5_exec}"
            )

            remaining = MIN_CYCLE_GAP - elapsed
            if remaining > 0:
                print(f"  Cooling down {remaining:.0f}s...\n")
                time.sleep(remaining)
            else:
                print()

    except KeyboardInterrupt:
        print("\n[Interrupted] Saving final state...")
        save_snapshot(pt, {})
        print("[OK] Stopped. State saved.")
    finally:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)


if __name__ == "__main__":
    main()
