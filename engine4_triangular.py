"""
engine4_triangular.py
======================
Triangular Arbitrage Bot Engine.

Strategy:
  - Fetch all USD trading pairs from Kraken public API (works globally, no auth)
  - Fetch bid/ask prices in batch via Kraken Ticker endpoint
  - Build a directed graph of currency pairs with bid/ask prices
  - Find cycles (triangles) where executing all legs yields > starting capital
  - Account for taker fees on every leg
  - Simulate execution with realistic slippage

Algorithm for each triangle USD → BTC → ETH → USD:
  1. Start with X USD
  2. Buy BTC at ask price (taker fee applied)
  3. Buy ETH with BTC at ask price (taker fee applied)
  4. Sell ETH for USD at bid price (taker fee applied)
  5. If final USD > initial USD → profitable

Kraken naming conventions:
  - XBT = BTC (internal name for Bitcoin)
  - XXBT, XETH, XXRP etc. = legacy Kraken format with X prefix
  - ZUSD = USD in Kraken's quote notation
  - Pairs like XXBTZUSD, XETHZUSD use legacy format
  - Newer pairs like SOLUSD, ADAUSD use direct naming

All trades are PAPER (simulated). Live order book from Kraken public API.

Can be run standalone:
    python engine4_triangular.py
"""

from __future__ import annotations

import itertools
import random
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from tabulate import tabulate

from config import (
    API,
    CAPITAL,
    FEES,
    HTTP,
    MIN_PROFIT,
    RISK,
    SLIPPAGE,
    get_logger,
)
from config import TRIANGULAR as TRI_CFG  # engine4-specific config block
from paper_trader import PaperTrader

logger = get_logger(__name__)

ENGINE_ID = "engine4_triangular"
ENGINE_CAPITAL = CAPITAL[ENGINE_ID]


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BookTicker:
    """Best bid/ask for a single Kraken spot pair."""

    symbol: str            # Kraken pair name (e.g. "XXBTZUSD")
    norm_base: str         # normalised base currency (e.g. "BTC")
    norm_quote: str        # normalised quote currency (e.g. "USD")
    bid_price: float
    ask_price: float
    bid_qty: float
    ask_qty: float


@dataclass
class TrianglePath:
    """One evaluated triangular arbitrage path."""

    base: str              # starting currency (e.g. "USD")
    leg1: str              # first pair  (e.g. "XXBTZUSD")
    leg2: str              # second pair (e.g. "XETHXXBT")
    leg3: str              # third pair  (e.g. "XETHZUSD")
    mid1: str              # intermediate currency after leg1 (e.g. "BTC")
    mid2: str              # intermediate currency after leg2 (e.g. "ETH")
    # Direction: "buy" = we're buying base of the pair, "sell" = selling
    leg1_dir: str
    leg2_dir: str
    leg3_dir: str
    start_amount: float    # USD notional
    end_amount: float      # USD notional after full triangle
    gross_profit_usd: float
    net_profit_usd: float
    net_profit_pct: float
    fees_paid_usd: float
    timestamp: str


# ─────────────────────────────────────────────────────────────────────────────
# KRAKEN NAMING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Kraken internal base names → normalised symbol
_KRAKEN_BASE_NORM: Dict[str, str] = {
    "XXBT":  "BTC",
    "XETH":  "ETH",
    "XXRP":  "XRP",
    "XLTC":  "LTC",
    "XXLM":  "XLM",
    "XXMR":  "XMR",
    "XZEC":  "ZEC",
    "XETC":  "ETC",
    "XXDG":  "DOGE",
    "XMLN":  "MLN",
    "XDAO":  "DAO",
    "XREP":  "REP",
    "XICN":  "ICN",
    "XNMC":  "NMC",
    "XYBTC": "YBTC",
}

# Kraken internal quote names → normalised
_KRAKEN_QUOTE_NORM: Dict[str, str] = {
    "ZUSD": "USD",
    "ZEUR": "EUR",
    "ZGBP": "GBP",
    "ZCAD": "CAD",
    "ZJPY": "JPY",
    "ZAUD": "AUD",
    "XXBT": "BTC",
    "XETH": "ETH",
}


def _norm(name: str) -> str:
    """Normalise a Kraken base/quote name to a clean ticker."""
    return _KRAKEN_BASE_NORM.get(name, _KRAKEN_QUOTE_NORM.get(name, name))


# ─────────────────────────────────────────────────────────────────────────────
# HTTP HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _get(
    url: str,
    params: Optional[Dict] = None,
    retries: int = HTTP["max_retries"],
) -> Tuple[Any, float]:
    """GET with retry + latency. Returns (data, latency_ms)."""
    headers = {"User-Agent": HTTP["user_agent"]}
    for attempt in range(1, retries + 1):
        try:
            t0 = time.perf_counter()
            resp = requests.get(
                url, params=params, headers=headers, timeout=HTTP["timeout"]
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            return resp.json(), latency_ms
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status == 429:
                logger.warning("Rate limited on %s — sleeping 15s", url)
                time.sleep(15)
            else:
                logger.warning("[%d/%d] HTTP %d on %s", attempt, retries, status, url)
                if attempt == retries:
                    raise
                time.sleep(HTTP["retry_delay"] * attempt)
        except requests.exceptions.RequestException as exc:
            logger.warning("[%d/%d] Request error: %s", attempt, retries, exc)
            if attempt == retries:
                raise
            time.sleep(HTTP["retry_delay"] * attempt)
    raise RuntimeError(f"All retries exhausted for {url}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENGINE CLASS
# ─────────────────────────────────────────────────────────────────────────────


class TriangularArbitrage:
    """
    Engine 4: Triangular Arbitrage Bot.

    Scans Kraken spot pairs to identify triangular opportunities across
    USD, BTC, and ETH base pairs.

    Parameters
    ----------
    paper_trader : PaperTrader
        Shared paper trading ledger.
    capital      : float
        Capital allocated to this engine.
    taker_fee    : float
        Exchange taker fee (default from Kraken config).
    """

    def __init__(
        self,
        paper_trader: Optional[PaperTrader] = None,
        capital: float = ENGINE_CAPITAL,
        taker_fee: float = FEES["kraken"]["taker"],
    ) -> None:
        self.pt = paper_trader
        self.capital = capital
        self.taker_fee = taker_fee

        # Caches
        self._book_tickers: Dict[str, BookTicker] = {}       # kraken_pair -> BookTicker
        self._valid_symbols: Set[str] = set()
        self._pair_map: Dict[Tuple[str, str], str] = {}      # (norm_base, norm_quote) -> kraken_pair
        self._volume_filter: Dict[str, float] = {}           # kraken_pair -> 24h volume USD
        self._last_opportunities: List[TrianglePath] = []
        self._scan_count = 0
        self._total_executions = 0

        # Kraken-specific: map norm pair to kraken pair name
        self._kraken_pairs_info: Dict[str, Dict] = {}        # kraken_pair -> pair info

    # ── Data Fetching ────────────────────────────────────────────────────────────────────────────

    def _fetch_exchange_info(self) -> None:
        """
        Fetch all valid Kraken spot pairs via AssetPairs endpoint.
        Builds self._kraken_pairs_info and self._valid_symbols.
        """
        try:
            data, latency = _get(API["kraken_asset_pairs"])
            logger.debug("Kraken AssetPairs fetched in %.0fms", latency)
            time.sleep(HTTP["rate_limit_delay"])

            result = data.get("result", {})
            self._kraken_pairs_info = result
            self._valid_symbols = set(result.keys())
            logger.debug("Kraken: %d valid pairs loaded", len(self._valid_symbols))

        except Exception as exc:
            logger.error("Failed to fetch Kraken AssetPairs: %s", exc)

    def fetch_order_books(self) -> Dict[str, BookTicker]:
        """
        Fetch best bid/ask for all relevant pairs via Kraken Ticker endpoint.
        Kraken accepts comma-separated pairs in a single request.

        Filters to pairs involving our base currencies (USD, BTC, ETH).

        Returns
        -------
        Dict of kraken_pair -> BookTicker (normalised names inside)
        """
        if not self._kraken_pairs_info:
            self._fetch_exchange_info()

        # Select pairs that involve USD, BTC, or ETH as quote
        wanted_quotes = {"ZUSD", "XXBT", "XETH", "USD", "XBT", "ETH"}
        target_pairs: List[str] = []
        for pname, pinfo in self._kraken_pairs_info.items():
            quote = pinfo.get("quote", "")
            norm_q = _norm(quote)
            if norm_q in ("USD", "BTC", "ETH"):
                target_pairs.append(pname)

        if not target_pairs:
            logger.error("No Kraken target pairs found")
            return {}

        # Batch into chunks of 50 (Kraken handles large batches fine)
        chunk_size = 50
        tickers: Dict[str, BookTicker] = {}

        for i in range(0, len(target_pairs), chunk_size):
            chunk = target_pairs[i: i + chunk_size]
            pairs_str = ",".join(chunk)
            try:
                data, latency = _get(API["kraken_ticker"], params={"pair": pairs_str})
                logger.debug("Kraken ticker batch %d/%d fetched in %.0fms",
                             i // chunk_size + 1, (len(target_pairs) - 1) // chunk_size + 1, latency)
                time.sleep(HTTP["rate_limit_delay"])

                result = data.get("result", {})
                for k_pair, ticker in result.items():
                    # Look up pair info (Kraken may return a canonical name)
                    pinfo = self._kraken_pairs_info.get(k_pair, {})
                    if not pinfo:
                        # Try to find by altname
                        for pn, pi in self._kraken_pairs_info.items():
                            if pi.get("altname", "") == k_pair:
                                pinfo = pi
                                k_pair = pn
                                break

                    if not pinfo:
                        continue

                    base = pinfo.get("base", "")
                    quote = pinfo.get("quote", "")
                    norm_base = _norm(base)
                    norm_quote = _norm(quote)

                    try:
                        bid = float(ticker["b"][0])
                        ask = float(ticker["a"][0])
                        # volume data: ticker["v"] = [today_vol, 24h_vol] in base units
                        # estimate USD volume = 24h_vol * last_price
                        last_price = float(ticker["c"][0]) if ticker.get("c") else ask
                        vol_base_24h = float(ticker["v"][1]) if ticker.get("v") else 0.0
                        # Estimate USD volume: for USD-quoted pairs use direct calc,
                        # for BTC/ETH-quoted pairs, approximate via last_price
                        if norm_quote == "USD":
                            vol_usd_24h = vol_base_24h * last_price
                        elif norm_quote == "BTC":
                            vol_usd_24h = vol_base_24h * last_price * 60000  # rough BTC/USD
                        elif norm_quote == "ETH":
                            vol_usd_24h = vol_base_24h * last_price * 3000   # rough ETH/USD
                        else:
                            vol_usd_24h = 0.0

                        if bid > 0 and ask > 0:
                            tickers[k_pair] = BookTicker(
                                symbol=k_pair,
                                norm_base=norm_base,
                                norm_quote=norm_quote,
                                bid_price=bid,
                                ask_price=ask,
                                bid_qty=0.0,
                                ask_qty=0.0,
                            )
                            self._volume_filter[k_pair] = vol_usd_24h
                    except (KeyError, ValueError, IndexError, TypeError):
                        pass

            except Exception as exc:
                logger.warning("Kraken ticker batch failed: %s", exc)

        self._book_tickers = tickers
        logger.debug("Kraken: %d book tickers loaded", len(tickers))
        return tickers

    # ── Graph Building ───────────────────────────────────────────────────────────────────────────

    def build_pair_graph(
        self, tickers: Optional[Dict[str, BookTicker]] = None
    ) -> Dict[Tuple[str, str], str]:
        """
        Build a mapping of (norm_base, norm_quote) -> kraken_pair_name from
        available tickers.  Includes reverse direction for bidirectional lookup.

        Returns
        -------
        Dict: {(base_ccy, quote_ccy): kraken_pair_name}
        """
        if tickers is None:
            tickers = self._book_tickers

        pair_map: Dict[Tuple[str, str], str] = {}

        for k_pair, bt in tickers.items():
            if bt.bid_price <= 0 or bt.ask_price <= 0:
                continue
            b = bt.norm_base
            q = bt.norm_quote
            if not b or not q:
                continue
            # Forward: base → quote (selling base for quote)
            pair_map[(b, q)] = k_pair
            # Reverse: quote → base (buying base with quote)
            pair_map[(q, b)] = k_pair

        self._pair_map = pair_map
        return pair_map

    def _get_rate(
        self, from_ccy: str, to_ccy: str, tickers: Dict[str, BookTicker]
    ) -> Tuple[float, str, str]:
        """
        Get the effective exchange rate from from_ccy to to_ccy.

        Returns
        -------
        (rate, kraken_pair_symbol, direction)
          rate = units of to_ccy per unit of from_ccy, before fee
          direction = "buy" (we buy to_ccy with from_ccy)
                    | "sell" (we sell from_ccy for to_ccy)
        """
        sym = self._pair_map.get((from_ccy, to_ccy))
        if sym is None:
            return 0.0, "", ""

        bt = tickers.get(sym)
        if bt is None:
            return 0.0, "", ""

        b = bt.norm_base
        q = bt.norm_quote

        if from_ccy == q and to_ccy == b:
            # Buying base (to_ccy) with quote (from_ccy): use ask, rate = 1/ask
            if bt.ask_price > 0:
                return 1.0 / bt.ask_price, sym, "buy"
        elif from_ccy == b and to_ccy == q:
            # Selling base (from_ccy) for quote (to_ccy): use bid, rate = bid
            return bt.bid_price, sym, "sell"

        return 0.0, "", ""

    # ── Opportunity Finding ──────────────────────────────────────────────────────────────────

    def find_triangular_opportunities(
        self,
        tickers: Optional[Dict[str, BookTicker]] = None,
        min_profit_pct: float = TRI_CFG["min_profit_pct"],
        capital: float = 10_000.0,
    ) -> List[TrianglePath]:
        """
        Scan all valid 3-leg triangular paths for profitability.

        Uses Kraken pairs across USD, BTC, ETH bases.
        Triangle structure: BASE → MID1 → MID2 → BASE

        Parameters
        ----------
        tickers        : Dict of BookTicker (uses cached if None)
        min_profit_pct : Minimum net profit as decimal (e.g. 0.001 = 0.1%)
        capital        : Starting amount in base currency per triangle test

        Returns
        -------
        List of profitable TrianglePath, sorted by net_profit_pct descending.
        """
        if tickers is None:
            tickers = self._book_tickers

        if not tickers:
            return []

        if not self._pair_map:
            self.build_pair_graph(tickers)

        ts = datetime.now(timezone.utc).isoformat()
        opportunities: List[TrianglePath] = []

        # Normalised base currencies to anchor triangles from
        bases = ["USD", "BTC", "ETH"]

        for base in bases:
            # Get all currencies accessible directly from this base
            connected: Set[str] = set()
            for (a, b_ccy) in self._pair_map:
                if a == base and b_ccy != base:
                    connected.add(b_ccy)

            # Test all triangles: base → mid1 → mid2 → base
            for mid1, mid2 in itertools.permutations(connected, 2):
                if mid1 == mid2 or mid1 == base or mid2 == base:
                    continue

                # Check all three legs exist
                r1, sym1, dir1 = self._get_rate(base, mid1, tickers)
                r2, sym2, dir2 = self._get_rate(mid1, mid2, tickers)
                r3, sym3, dir3 = self._get_rate(mid2, base, tickers)

                if r1 <= 0 or r2 <= 0 or r3 <= 0:
                    continue
                if not sym1 or not sym2 or not sym3:
                    continue

                # Skip triangles where all three legs are the same pair (degenerate)
                if sym1 == sym2 or sym2 == sym3 or sym1 == sym3:
                    continue

                # Volume filter: ALL triangle legs must have minimum 24h volume
                # This prevents false positives from stale/illiquid pairs (e.g. ETHW)
                min_vol = TRI_CFG["min_volume_usd"]
                leg_volumes = [
                    self._volume_filter.get(s, 0) for s in [sym1, sym2, sym3]
                ]
                if any(v < min_vol for v in leg_volumes):
                    continue

                # Execute the triangle
                fee = self.taker_fee
                amt_after_leg1 = capital * r1 * (1 - fee)
                amt_after_leg2 = amt_after_leg1 * r2 * (1 - fee)
                amt_after_leg3 = amt_after_leg2 * r3 * (1 - fee)

                # Convert back to USD
                if base == "USD":
                    end_usd = amt_after_leg3
                    start_usd = capital
                else:
                    base_usd = self._estimate_base_usd_price(base, tickers)
                    if base_usd <= 0:
                        continue
                    end_usd = amt_after_leg3 * base_usd
                    start_usd = capital * base_usd

                gross_profit = end_usd - start_usd
                fees_usd = start_usd * fee * 3  # 3 legs
                net_profit = gross_profit  # fees already deducted in leg calculations
                net_profit_pct = net_profit / start_usd if start_usd > 0 else 0.0

                # Sanity cap: reject triangles claiming > 2% net profit
                # On a single centralized exchange, real triangular arb is
                # typically 0.01-0.10%.  Anything above 2% is likely stale data.
                if net_profit_pct > 0.02:
                    continue

                if net_profit_pct >= min_profit_pct:
                    opportunities.append(TrianglePath(
                        base=base,
                        leg1=sym1,
                        leg2=sym2,
                        leg3=sym3,
                        mid1=mid1,
                        mid2=mid2,
                        leg1_dir=dir1,
                        leg2_dir=dir2,
                        leg3_dir=dir3,
                        start_amount=round(start_usd, 2),
                        end_amount=round(end_usd, 4),
                        gross_profit_usd=round(gross_profit, 4),
                        net_profit_usd=round(net_profit, 4),
                        net_profit_pct=round(net_profit_pct, 6),
                        fees_paid_usd=round(fees_usd, 4),
                        timestamp=ts,
                    ))

        opportunities.sort(key=lambda x: x.net_profit_pct, reverse=True)
        self._last_opportunities = opportunities
        return opportunities

    def _estimate_base_usd_price(
        self, base: str, tickers: Dict[str, BookTicker]
    ) -> float:
        """Approximate USD price of a base currency via its USD pair."""
        for k_pair, bt in tickers.items():
            if bt.norm_base == base and bt.norm_quote == "USD" and bt.bid_price > 0:
                return bt.bid_price
        return 1.0

    def scan_all_bases(
        self,
        bases: Optional[List[str]] = None,
        capital: float = 10_000.0,
    ) -> List[TrianglePath]:
        """
        Fetch fresh order books from Kraken and scan all base-currency triangles.

        Parameters
        ----------
        bases   : Base currencies to scan from (ignored — uses USD/BTC/ETH internally)
        capital : Simulated starting capital per triangle

        Returns
        -------
        Combined list of profitable triangles, sorted by net_profit_pct desc.
        """
        logger.info("Engine4: scanning triangular opportunities via Kraken...")

        # Fetch all pair info if not loaded
        if not self._kraken_pairs_info:
            self._fetch_exchange_info()

        # Fetch book tickers
        tickers = self.fetch_order_books()
        if not tickers:
            logger.warning("Engine4: No Kraken tickers fetched — skipping scan")
            return []

        # Build graph
        self.build_pair_graph(tickers)

        # Find all profitable triangles
        all_opportunities = self.find_triangular_opportunities(
            tickers=tickers,
            min_profit_pct=TRI_CFG["min_profit_pct"],
            capital=capital,
        )

        return all_opportunities

    # ── Execution Simulation ──────────────────────────────────────────────────────────────────

    def simulate_execution(
        self,
        path: TrianglePath,
        capital: float,
    ) -> Optional[Any]:
        """
        Paper-trade a triangular arbitrage path.

        Executes three sequential legs, applying realistic slippage to each.
        All legs are executed atomically in the paper trader as a single
        net-profit trade.

        Parameters
        ----------
        path    : TrianglePath to execute
        capital : USD notional to deploy

        Returns
        -------
        Trade record of the net result, or None on failure.
        """
        if self.pt is None:
            return None

        if len(self.pt.get_positions(ENGINE_ID)) >= RISK["max_open_positions"]:
            logger.warning("Engine4: max open positions reached")
            return None

        # Apply slippage to each leg (3 legs)
        slip = [1 + random.uniform(SLIPPAGE["min"], SLIPPAGE["max"]) for _ in range(3)]
        effective_pct = path.net_profit_pct - sum(s - 1 for s in slip) * 0.01

        net_profit_usd = capital * effective_pct

        amount_units = capital / max(path.start_amount, 0.01)
        entry_price = path.start_amount
        exit_price = entry_price * (1 + path.net_profit_pct - (sum(s - 1 for s in slip) * 0.002))

        try:
            open_trade = self.pt.execute_trade(
                engine=ENGINE_ID,
                symbol=f"{path.base}/{path.mid1}/{path.mid2}",
                side="long",
                amount=amount_units,
                price=entry_price,
                leverage=1.0,
                order_type="taker",
                metadata={
                    "strategy": "triangular_arb",
                    "path": f"{path.base}→{path.mid1}→{path.mid2}→{path.base}",
                    "leg1": path.leg1,
                    "leg2": path.leg2,
                    "leg3": path.leg3,
                    "expected_profit_pct": path.net_profit_pct,
                    "source": "kraken",
                },
            )

            close_trade = self.pt.close_position(
                engine=ENGINE_ID,
                position_id=open_trade.position_id,
                exit_price=exit_price,
                order_type="taker",
                metadata={
                    "strategy": "triangular_arb",
                    "legs_executed": 3,
                },
            )

            self._total_executions += 1
            logger.info(
                "Engine4: Triangle executed %s→%s→%s→%s | net=$%.4f (%.4f%%)",
                path.base, path.mid1, path.mid2, path.base,
                net_profit_usd, path.net_profit_pct * 100,
            )
            return close_trade

        except Exception as exc:
            logger.error("Engine4: execution failed: %s", exc)
            return None

    # ── Display ───────────────────────────────────────────────────────────────────────────

    def print_opportunity_table(
        self,
        opportunities: Optional[List[TrianglePath]] = None,
        top_n: int = 15,
    ) -> None:
        """Print a formatted table of top triangular arbitrage opportunities."""
        if opportunities is None:
            opportunities = self._last_opportunities

        rows = []
        for i, op in enumerate(opportunities[:top_n], 1):
            path_str = f"{op.base}→{op.mid1}→{op.mid2}→{op.base}"
            rows.append([
                i,
                path_str,
                op.leg1,
                op.leg2,
                op.leg3,
                f"${op.start_amount:,.0f}",
                f"${op.end_amount:,.2f}",
                f"${op.net_profit_usd:.4f}",
                f"{op.net_profit_pct * 100:.4f}%",
            ])

        headers = [
            "#", "Triangle Path", "Leg 1", "Leg 2", "Leg 3",
            "Capital", "End Value", "Net Profit", "Net %",
        ]
        print("\n" + "─" * 90)
        print("  ENGINE 4: TRIANGULAR ARBITRAGE OPPORTUNITIES (Kraken)")
        print("─" * 90)
        if rows:
            print(tabulate(rows, headers=headers, tablefmt="simple"))
        else:
            print("  No profitable triangular opportunities found above threshold.")
        print("─" * 90 + "\n")

    # ── Full Scan Cycle ────────────────────────────────────────────────────────────────

    def run_cycle(
        self,
        auto_execute: bool = True,
        max_executions: int = 3,
    ) -> Dict[str, Any]:
        """
        Execute one full scan → triangle search → (optional) simulation cycle.

        Parameters
        ----------
        auto_execute   : If True, simulate top opportunities
        max_executions : Maximum paper trades to execute this cycle

        Returns
        -------
        Cycle summary dict for dashboard display.
        """
        logger.info("Engine4: starting Kraken triangular arbitrage scan cycle...")
        t_start = time.perf_counter()

        # Fetch tickers and build graph
        try:
            opportunities = self.scan_all_bases(
                capital=TRI_CFG["simulation_capital"],
            )
        except Exception as exc:
            logger.error("Engine4: scan failed: %s", exc)
            pnl_data = self.pt.get_pnl(ENGINE_ID) if self.pt else {"realized_pnl": 0.0}
            return {
                "engine": ENGINE_ID,
                "error": str(exc),
                "session_pnl": pnl_data["realized_pnl"],
            }

        # Execute top opportunities
        executions = 0
        if auto_execute and opportunities and self.pt is not None:
            per_trade_capital = self.capital / max(RISK["max_open_positions"], 1)
            for op in opportunities[:max_executions]:
                if op.net_profit_pct >= MIN_PROFIT[ENGINE_ID]:
                    try:
                        self.simulate_execution(op, capital=per_trade_capital)
                        executions += 1
                    except Exception as exc:
                        logger.warning("Engine4: execution failed: %s", exc)

        pnl = self.pt.get_pnl(ENGINE_ID) if self.pt else {"realized_pnl": 0.0}
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)
        self._scan_count += 1

        top = opportunities[0] if opportunities else None
        top_path = f"{top.base}→{top.mid1}→{top.mid2}→{top.base}" if top else "N/A"

        return {
            "engine": ENGINE_ID,
            "pairs_in_graph": len(self._book_tickers),
            "opportunities_found": len(opportunities),
            "executions_this_cycle": executions,
            "total_executions": self._total_executions,
            "top_triangle": top_path,
            "top_profit_pct": top.net_profit_pct * 100 if top else 0.0,
            "top_profit_usd": top.net_profit_usd if top else 0.0,
            "elapsed_ms": elapsed_ms,
            "session_pnl": pnl["realized_pnl"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pt = PaperTrader()
    engine = TriangularArbitrage(paper_trader=pt)

    print("\n[Engine 4] Scanning Kraken pairs for triangular arbitrage (live data)...\n")
    cycle_result = engine.run_cycle(auto_execute=True, max_executions=3)
    engine.print_opportunity_table(top_n=10)

    print("\nCycle Summary:")
    for k, v in cycle_result.items():
        print(f"  {k}: {v}")

    print("\nPortfolio P&L:")
    pnl = pt.get_pnl(ENGINE_ID)
    for k, v in pnl.items():
        print(f"  {k}: {v}")
