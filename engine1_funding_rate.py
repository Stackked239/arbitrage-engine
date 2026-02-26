"""
engine1_funding_rate.py
========================
Funding Rate Harvester Engine.

Strategy:
  - Fetch current perpetual funding rates from CoinGecko derivatives API
    (PRIMARY: https://api.coingecko.com/api/v3/derivatives)
  - Fallback: derive implied funding from spot-perp spread via Bybit tickers
  - Identify rates above threshold (baseline, strong, exceptional tiers)
  - Simulate delta-neutral entry: long spot + short perp
  - Track simulated funding income over time
  - Exit if rate flips below threshold or position hits stop-loss

All trades are PAPER (simulated). Live prices via CoinGecko public API.

Can be run standalone:
    python engine1_funding_rate.py
"""

from __future__ import annotations

import time
import random
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import requests
from tabulate import tabulate

from config import (
    API,
    CAPITAL,
    FEES,
    FUNDING_RATE_TIERS,
    HTTP,
    RISK,
    SLIPPAGE,
    get_logger,
)
from paper_trader import PaperTrader

logger = get_logger(__name__)

ENGINE_ID = "engine1_funding_rate"
ENGINE_CAPITAL = CAPITAL[ENGINE_ID]
HOURS_PER_YEAR = 8760
FUNDING_PERIODS_PER_YEAR = HOURS_PER_YEAR / 8  # 8-hr funding cycle → 1095/yr


# ───────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ───────────────────────────────────────────────────────────────────────────────


@dataclass
class FundingRateInfo:
    """Snapshot of a single instrument's funding data."""

    symbol: str
    funding_rate: float             # current 8-hr rate (decimal)
    predicted_rate: Optional[float] # next predicted rate (if available)
    mark_price: float
    index_price: float
    spot_price: Optional[float]
    premium_pct: float              # (mark - index) / index
    annualized_rate: float          # rate * periods_per_year
    annualized_with_leverage: float # annualized_rate * leverage
    tier: str                       # "baseline" | "moderate" | "strong" | "exceptional"
    timestamp: str


@dataclass
class FundingPosition:
    """Tracks an open delta-neutral funding position."""

    position_id: str
    symbol: str
    funding_rate_at_entry: float
    spot_amount: float          # units of base asset bought (long leg)
    perp_amount: float          # units sold short on perp
    spot_entry_price: float
    perp_entry_price: float
    leverage: float
    capital_deployed: float     # total USD deployed
    opened_at: str
    accumulated_funding_usd: float = 0.0
    payments_received: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ───────────────────────────────────────────────────────────────────────────────
# HTTP HELPERS
# ───────────────────────────────────────────────────────────────────────────────


def _get(url: str, params: Optional[Dict] = None, retries: int = HTTP["max_retries"]) -> Tuple[Any, float]:
    """
    Perform a GET request with retry logic.

    Returns
    -------
    (parsed_json, latency_ms)  or raises on failure.
    """
    headers = {"User-Agent": HTTP["user_agent"]}
    for attempt in range(1, retries + 1):
        try:
            t0 = time.perf_counter()
            resp = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=HTTP["timeout"],
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            return resp.json(), latency_ms
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                logger.warning("Rate limited on %s; sleeping 10s", url)
                time.sleep(10)
            else:
                logger.warning("[attempt %d/%d] HTTP error %s on %s", attempt, retries, exc, url)
                if attempt == retries:
                    raise
                time.sleep(HTTP["retry_delay"] * attempt)
        except requests.exceptions.RequestException as exc:
            logger.warning("[attempt %d/%d] Request error %s on %s", attempt, retries, exc, url)
            if attempt == retries:
                raise
            time.sleep(HTTP["retry_delay"] * attempt)
    raise RuntimeError(f"All retries exhausted for {url}")


# ───────────────────────────────────────────────────────────────────────────────
# MAIN ENGINE CLASS
# ───────────────────────────────────────────────────────────────────────────────


class FundingRateScanner:
    """
    Engine 1: Funding Rate Harvester.

    Scans perpetual futures funding rates via CoinGecko derivatives API
    (global, no auth), identifies delta-neutral opportunities, and manages
    simulated positions.

    Data source: GET https://api.coingecko.com/api/v3/derivatives
    Fallback:    Bybit public tickers API

    Parameters
    ----------
    paper_trader : PaperTrader
        Shared paper trading ledger.
    capital      : float
        Capital allocated to this engine.
    leverage     : float
        Leverage for the short perp leg (default from config).
    """

    def __init__(
        self,
        paper_trader: Optional[PaperTrader] = None,
        capital: float = ENGINE_CAPITAL,
        leverage: float = RISK["default_leverage"],
    ) -> None:
        self.pt = paper_trader
        self.capital = capital
        self.leverage = leverage
        self._active_positions: Dict[str, FundingPosition] = {}  # symbol -> position
        self._last_scan: Optional[List[FundingRateInfo]] = None
        self._scan_count = 0

    # ── Data Fetching ─────────────────────────────────────────────────────────────────────────────

    def _fetch_coingecko_derivatives(self) -> Tuple[List[Dict], float]:
        """
        Fetch perpetual contract data from CoinGecko derivatives endpoint.

        Returns list of records with fields:
          market, symbol, index_id, price, contract_type,
          index, basis, spread, funding_rate, open_interest,
          volume_24h, last_traded_at

        CoinGecko funding_rate is already in percentage form (e.g. 0.003693
        means 0.003693%).  We normalise to decimal (divide by 100).
        """
        data, latency = _get(API["coingecko_derivatives"])
        time.sleep(HTTP["rate_limit_delay"])
        return data, latency

    def _fetch_bybit_tickers_fallback(self) -> Tuple[List[Dict], float]:
        """
        Fallback: fetch Bybit perpetual tickers (includes fundingRate field).
        Returns list of ticker dicts.
        """
        data, latency = _get(API["bybit_tickers"], params={"category": "linear"})
        time.sleep(HTTP["rate_limit_delay"])
        return data.get("result", {}).get("list", []), latency

    def _fetch_spot_prices_coingecko(
        self, symbols: List[str]
    ) -> Dict[str, float]:
        """
        Fetch spot prices from CoinGecko simple/price for a list of symbols
        mapped through the known CoinGecko ID map.
        """
        # Simple ID map for common perp base tokens (index_id from CoinGecko)
        _CG_ID: Dict[str, str] = {
            "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
            "XRP": "ripple", "DOGE": "dogecoin", "ADA": "cardano",
            "AVAX": "avalanche-2", "DOT": "polkadot", "LINK": "chainlink",
            "MATIC": "matic-network", "UNI": "uniswap", "AAVE": "aave",
            "BNB": "binancecoin", "LTC": "litecoin", "ATOM": "cosmos",
            "FIL": "filecoin", "NEAR": "near", "APT": "aptos",
            "ARB": "arbitrum", "OP": "optimism", "INJ": "injective-protocol",
            "SUI": "sui", "TIA": "celestia", "SEI": "sei-network",
        }
        ids = [_CG_ID[s] for s in symbols if s in _CG_ID]
        if not ids:
            return {}
        params = {"ids": ",".join(ids[:50]), "vs_currencies": "usd"}
        try:
            data, _ = _get(API["coingecko_simple_price"], params=params)
            id_to_sym = {v: k for k, v in _CG_ID.items() if k in symbols}
            return {
                id_to_sym[cg_id]: float(vals["usd"])
                for cg_id, vals in data.items()
                if cg_id in id_to_sym and "usd" in vals
            }
        except Exception as exc:
            logger.debug("Spot price lookup failed: %s", exc)
            return {}

    # ── Analysis ──────────────────────────────────────────────────────────────────────────────

    def _classify_tier(self, rate: float) -> str:
        """Classify a funding rate into a named tier."""
        abs_rate = abs(rate)
        if abs_rate >= FUNDING_RATE_TIERS["exceptional"]:
            return "exceptional"
        elif abs_rate >= FUNDING_RATE_TIERS["strong"]:
            return "strong"
        elif abs_rate >= FUNDING_RATE_TIERS["moderate"]:
            return "moderate"
        elif abs_rate >= FUNDING_RATE_TIERS["baseline"]:
            return "baseline"
        return "below_threshold"

    def calculate_annualized_return(
        self, rate: float, leverage: float = 1.0
    ) -> float:
        """
        Annualize an 8-hour funding rate.

        Parameters
        ----------
        rate     : 8-hour funding rate as decimal (e.g. 0.001 = 0.1%)
        leverage : Leverage multiplier applied to short perp leg

        Returns
        -------
        Annualized return as decimal (e.g. 0.10 = 10% APY)
        """
        return abs(rate) * FUNDING_PERIODS_PER_YEAR * leverage

    def scan_rates(self) -> List[FundingRateInfo]:
        """
        Fetch and parse current funding rates from CoinGecko derivatives API.

        PRIMARY source: CoinGecko /derivatives (perpetual contracts, all exchanges)
        FALLBACK: Bybit public tickers

        CoinGecko funding_rate field is a percentage as a float, e.g. 0.003693
        means 0.003693% per funding period.  We convert to decimal (divide by 100).

        Returns
        -------
        List of FundingRateInfo, ordered by absolute funding rate descending.
        """
        logger.info("Engine1: scanning funding rates via CoinGecko derivatives...")
        t_start = time.perf_counter()

        raw_data: List[Dict] = []
        lat1 = 0.0

        # PRIMARY: CoinGecko derivatives
        try:
            raw_data, lat1 = self._fetch_coingecko_derivatives()
            logger.debug("CoinGecko derivatives: %d records in %.0fms", len(raw_data), lat1)
        except Exception as exc:
            logger.warning("CoinGecko derivatives failed (%s), trying Bybit fallback", exc)

        # FALLBACK: Bybit if CoinGecko returned nothing
        if not raw_data:
            try:
                bybit_tickers, lat1 = self._fetch_bybit_tickers_fallback()
                # Normalise Bybit ticker format to CoinGecko-like dicts
                for t in bybit_tickers:
                    sym = t.get("symbol", "")
                    if not sym.endswith("USDT"):
                        continue
                    try:
                        price = float(t.get("lastPrice", 0))
                        fr = float(t.get("fundingRate", 0))
                        index_id = sym.replace("USDT", "")
                        raw_data.append({
                            "market": "Bybit (Futures)",
                            "symbol": sym,
                            "index_id": index_id,
                            "price": str(price),
                            "contract_type": "perpetual",
                            "index": price,
                            "basis": 0.0,
                            "spread": 0.0,
                            # Bybit fundingRate is already a decimal (e.g. 0.0001)
                            # Scale to CoinGecko's percentage style for uniform parsing
                            "funding_rate": fr * 100,
                            "open_interest": float(t.get("openInterest", 0)),
                            "volume_24h": float(t.get("volume24h", 0)),
                        })
                    except (ValueError, TypeError):
                        pass
                logger.debug("Bybit fallback: %d records", len(raw_data))
            except Exception as exc2:
                logger.error("Both funding rate sources failed: %s", exc2)
                return []

        ts = datetime.now(timezone.utc).isoformat()
        results: List[FundingRateInfo] = []

        # Deduplicate by (index_id, market) -- take highest absolute rate per symbol
        # Also keep only perpetual contracts
        seen_symbols: Dict[str, FundingRateInfo] = {}

        # Mainstream exchange filter — exclude exotic/illiquid venues that
        # report extreme funding rates but have virtually no liquidity.
        _BLOCKED_VENUES = {
            "grvt", "ostium", "navigator", "zeta", "drift",
            "aevo", "vertex", "rabbitx", "bluefin", "synfutures",
            "globe", "phemex", "coindcx",
        }

        for item in raw_data:
            try:
                # Only perpetual contracts
                if item.get("contract_type") != "perpetual":
                    continue

                index_id: str = str(item.get("index_id", "")).upper()
                market: str = str(item.get("market", ""))
                cg_symbol: str = str(item.get("symbol", ""))

                if not index_id:
                    continue

                # ── Venue filter: skip exotic/illiquid exchanges ───────────
                market_lower = market.lower()
                if any(blocked in market_lower for blocked in _BLOCKED_VENUES):
                    continue

                # ── Sanity cap: reject funding rates > 1% per 8hr period ─
                # These are almost always data errors or illiquid markets
                raw_fr = item.get("funding_rate")
                if raw_fr is not None and abs(float(raw_fr)) > 10.0:
                    # 10 in CoinGecko's pct format = 0.10% per 8hrs — already very high
                    # Anything beyond this is almost certainly a data error
                    continue

                # CoinGecko funding_rate is in percent (0.003693 = 0.003693%)
                # Convert to 8h decimal: divide by 100
                funding_rate_pct = item.get("funding_rate")
                if funding_rate_pct is None:
                    continue
                funding_rate = float(funding_rate_pct) / 100.0

                # Mark price = current price field
                mark_price = float(item.get("price", 0) or 0)
                # Index price from CoinGecko 'index' field
                index_price = float(item.get("index", 0) or mark_price)

                if mark_price <= 0:
                    continue

                premium_pct = (
                    (mark_price - index_price) / index_price
                    if index_price > 0 else 0.0
                )
                ann_rate = self.calculate_annualized_return(funding_rate, leverage=1.0)
                ann_leveraged = self.calculate_annualized_return(funding_rate, leverage=self.leverage)
                tier = self._classify_tier(funding_rate)

                # Build a unique display symbol: "BTC (Binance Futures)"
                display_symbol = f"{index_id} ({market})"
                # For deduplication key, use index_id only -- keep best rate per symbol
                dedup_key = index_id

                info = FundingRateInfo(
                    symbol=display_symbol,
                    funding_rate=funding_rate,
                    predicted_rate=None,
                    mark_price=mark_price,
                    index_price=index_price,
                    spot_price=None,  # filled in batch below
                    premium_pct=premium_pct,
                    annualized_rate=ann_rate,
                    annualized_with_leverage=ann_leveraged,
                    tier=tier,
                    timestamp=ts,
                )

                # Keep only highest absolute rate per underlying symbol
                existing = seen_symbols.get(dedup_key)
                if existing is None or abs(funding_rate) > abs(existing.funding_rate):
                    seen_symbols[dedup_key] = info

            except (KeyError, ValueError, TypeError) as exc:
                logger.debug("Skipping record: %s", exc)

        results = list(seen_symbols.values())

        # Batch-fetch spot prices for top symbols
        top_syms = [k for k in seen_symbols]
        spot_prices = {}
        try:
            spot_prices = self._fetch_spot_prices_coingecko(top_syms[:40])
        except Exception:
            pass

        for info in results:
            # Extract index_id from display_symbol like "BTC (Binance Futures)"
            sym_key = info.symbol.split(" (")[0]
            if sym_key in spot_prices:
                info.spot_price = spot_prices[sym_key]

        # Sort by absolute funding rate descending
        results.sort(key=lambda x: abs(x.funding_rate), reverse=True)

        elapsed = (time.perf_counter() - t_start) * 1000
        logger.info(
            "Engine1: scan complete — %d unique symbols in %.0fms (source_lat=%.0fms)",
            len(results), elapsed, lat1,
        )
        self._last_scan = results
        self._scan_count += 1
        return results

    def find_opportunities(
        self, min_tier: str = "baseline"
    ) -> List[FundingRateInfo]:
        """
        Filter scan results to actionable opportunities.

        Parameters
        ----------
        min_tier : Minimum tier to include ("baseline", "moderate", "strong", "exceptional")

        Returns
        -------
        Filtered list, sorted by annualized leveraged return descending.
        """
        tier_order = ["below_threshold", "baseline", "moderate", "strong", "exceptional"]
        min_idx = tier_order.index(min_tier)

        if self._last_scan is None:
            self.scan_rates()

        ops = [
            info for info in (self._last_scan or [])
            if tier_order.index(info.tier) >= min_idx
        ]
        ops.sort(key=lambda x: x.annualized_with_leverage, reverse=True)
        return ops

    # ── Position Management ──────────────────────────────────────────────────────────────────────

    def simulate_entry(
        self,
        symbol: str,
        rate: float,
        capital: float,
        leverage: float,
        spot_price: Optional[float] = None,
    ) -> Optional[FundingPosition]:
        """
        Paper-trade a delta-neutral position: long spot + short perp.

        Capital is split equally between the two legs (for a truly delta-neutral
        position the notionals are matched, not the cash outlay).

        Parameters
        ----------
        symbol       : e.g. "BTC (Binance Futures)"
        rate         : Current 8-hour funding rate
        capital      : USD capital to deploy
        leverage     : Leverage on perp leg
        spot_price   : Current spot price (uses mark price if None)

        Returns
        -------
        FundingPosition if entry simulated successfully, else None.
        """
        if self.pt is None:
            return None

        if symbol in self._active_positions:
            logger.debug("Already have position in %s", symbol)
            return None

        # Check open position cap
        open_pos = self.pt.get_positions(ENGINE_ID)
        if len(open_pos) >= RISK["max_open_positions"]:
            logger.warning("Max open positions reached for %s", ENGINE_ID)
            return None

        # Max position size check
        max_pos = self.capital * RISK["max_position_size_pct"]
        if capital > max_pos:
            capital = max_pos
            logger.debug("Capped position to $%.2f", capital)

        if spot_price is None or spot_price <= 0:
            logger.error("No valid spot price for %s", symbol)
            return None

        # Simulate slippage on entry
        slip_factor_spot = 1 + random.uniform(SLIPPAGE["min"], SLIPPAGE["max"])
        slip_factor_perp = 1 + random.uniform(SLIPPAGE["min"], SLIPPAGE["max"])
        adjusted_spot_price = spot_price * slip_factor_spot
        adjusted_perp_price = spot_price * slip_factor_perp

        half_capital = capital / 2.0
        spot_amount = half_capital / adjusted_spot_price

        # Long spot leg
        spot_trade = self.pt.execute_trade(
            engine=ENGINE_ID,
            symbol=symbol,
            side="long",
            amount=spot_amount,
            price=adjusted_spot_price,
            leverage=1.0,
            order_type="taker",
            latency_ms=0.0,
            metadata={"leg": "spot", "strategy": "delta_neutral_funding"},
        )

        # Short perp leg
        perp_amount = half_capital * leverage / adjusted_perp_price
        perp_trade = self.pt.execute_trade(
            engine=ENGINE_ID,
            symbol=symbol,
            side="short",
            amount=perp_amount,
            price=adjusted_perp_price,
            leverage=leverage,
            order_type="taker",
            latency_ms=0.0,
            metadata={"leg": "perp", "strategy": "delta_neutral_funding"},
        )

        fp = FundingPosition(
            position_id=f"{spot_trade.position_id}:{perp_trade.position_id}",
            symbol=symbol,
            funding_rate_at_entry=rate,
            spot_amount=spot_amount,
            perp_amount=perp_amount,
            spot_entry_price=adjusted_spot_price,
            perp_entry_price=adjusted_perp_price,
            leverage=leverage,
            capital_deployed=capital,
            opened_at=datetime.now(timezone.utc).isoformat(),
        )

        self._active_positions[symbol] = fp
        logger.info(
            "Opened delta-neutral position in %s | capital=$%.2f | rate=%.4f%% | ann=%.1f%%",
            symbol, capital, rate * 100, self.calculate_annualized_return(rate, leverage) * 100,
        )
        return fp

    def simulate_funding_payment(self, position: FundingPosition) -> float:
        """
        Calculate and log one funding payment for an open position.

        The short perp receives funding when rate > 0 (longs pay shorts).

        Parameters
        ----------
        position : Open FundingPosition

        Returns
        -------
        USD amount of funding received/paid (positive = income)
        """
        # Notional of perp position
        perp_notional = position.perp_amount * position.perp_entry_price
        payment = perp_notional * abs(position.funding_rate_at_entry)

        # If rate is positive, our short receives payment; negative = we pay
        if position.funding_rate_at_entry < 0:
            payment = -payment

        position.accumulated_funding_usd += payment
        position.payments_received += 1

        if self.pt is not None:
            # Extract the two position IDs stored as "spot_id:perp_id"
            parts = position.position_id.split(":")
            perp_position_id = parts[1] if len(parts) == 2 else parts[0]

            self.pt.log_periodic_income(
                engine=ENGINE_ID,
                position_id=perp_position_id,
                amount_usd=payment,
                income_type="funding_payment",
                metadata={
                    "symbol": position.symbol,
                    "rate": position.funding_rate_at_entry,
                    "payment_number": position.payments_received,
                },
            )

        logger.debug(
            "Funding payment #%d for %s: $%.4f (total $%.4f)",
            position.payments_received,
            position.symbol,
            payment,
            position.accumulated_funding_usd,
        )
        return payment

    def monitor_positions(
        self, current_rates: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """
        Review all open positions; exit any that have degraded.

        Exits if:
          - Funding rate has flipped sign (from positive to negative)
          - Rate dropped below baseline threshold

        Parameters
        ----------
        current_rates : {symbol: current_funding_rate}
                        If None, triggers a fresh scan.

        Returns
        -------
        List of symbols that were exited.
        """
        if current_rates is None:
            scan = self.scan_rates()
            current_rates = {info.symbol: info.funding_rate for info in scan}

        exited: List[str] = []
        for symbol, fp in list(self._active_positions.items()):
            current_rate = current_rates.get(symbol)
            if current_rate is None:
                continue

            should_exit = False
            reason = ""

            if current_rate < 0 and fp.funding_rate_at_entry > 0:
                should_exit = True
                reason = "rate flipped negative"
            elif abs(current_rate) < FUNDING_RATE_TIERS["baseline"]:
                should_exit = True
                reason = "rate below baseline threshold"

            if should_exit:
                logger.info("Exiting %s — %s (rate=%.4f%%)", symbol, reason, current_rate * 100)
                parts = fp.position_id.split(":")
                spot_pid = parts[0]
                perp_pid = parts[1] if len(parts) == 2 else parts[0]

                exit_price = fp.spot_entry_price  # use entry price as fallback

                if self.pt is not None:
                    self.pt.close_position(ENGINE_ID, spot_pid, exit_price)
                    self.pt.close_position(ENGINE_ID, perp_pid, exit_price)
                del self._active_positions[symbol]
                exited.append(symbol)

        return exited

    # ── Display ──────────────────────────────────────────────────────────────────────────────

    def print_opportunity_table(
        self, opportunities: Optional[List[FundingRateInfo]] = None, top_n: int = 20
    ) -> None:
        """Print a formatted table of top funding rate opportunities."""
        if opportunities is None:
            opportunities = self.find_opportunities(min_tier="baseline")

        rows = []
        for i, info in enumerate(opportunities[:top_n], 1):
            rows.append([
                i,
                info.symbol,
                f"{info.funding_rate * 100:+.4f}%",
                f"{info.annualized_rate * 100:.1f}%",
                f"{info.annualized_with_leverage * 100:.1f}%",
                f"${info.mark_price:,.4f}",
                f"{info.premium_pct * 100:+.3f}%",
                info.tier.upper(),
            ])

        headers = [
            "#", "Symbol", "Rate (8h)", "Ann. (1x)",
            f"Ann. ({self.leverage}x lev)", "Mark Price", "Premium", "Tier",
        ]
        print("\n" + "─" * 70)
        print("  ENGINE 1: FUNDING RATE OPPORTUNITIES")
        print("─" * 70)
        if rows:
            print(tabulate(rows, headers=headers, tablefmt="simple"))
        else:
            print("  No opportunities above baseline threshold.")
        print("─" * 70 + "\n")

    # ── Full Scan Cycle ─────────────────────────────────────────────────────────────────────────────

    def run_cycle(self, auto_enter: bool = True, max_new_positions: int = 3) -> Dict[str, Any]:
        """
        Execute one full scan → opportunity ranking → (optional) entry cycle.

        Parameters
        ----------
        auto_enter       : If True, auto-simulate top opportunities
        max_new_positions: Maximum new paper trades to open this cycle

        Returns
        -------
        Cycle summary dict suitable for dashboard display.
        """
        scan_results = self.scan_rates()
        ops = self.find_opportunities(min_tier="baseline")
        top_op = ops[0] if ops else None

        new_entries = 0
        if auto_enter and ops and self.pt is not None:
            per_position_capital = self.capital / max(RISK["max_open_positions"], 1)
            # Only enter symbols not already held
            for op in ops[:max_new_positions]:
                if op.symbol not in self._active_positions:
                    if op.funding_rate > 0:  # positive rate → longs pay shorts → harvest
                        fp = self.simulate_entry(
                            symbol=op.symbol,
                            rate=op.funding_rate,
                            capital=per_position_capital,
                            leverage=self.leverage,
                            spot_price=op.spot_price or op.mark_price,
                        )
                        if fp:
                            new_entries += 1
                            time.sleep(HTTP["rate_limit_delay"])

        # Simulate a funding payment for all open positions (demo mode)
        for fp in self._active_positions.values():
            self.simulate_funding_payment(fp)

        pnl = self.pt.get_pnl(ENGINE_ID) if self.pt else {"realized_pnl": 0.0}

        return {
            "engine": ENGINE_ID,
            "symbols_scanned": len(scan_results),
            "opportunities_found": len(ops),
            "new_positions_opened": new_entries,
            "active_positions": len(self._active_positions),
            "top_symbol": top_op.symbol if top_op else "N/A",
            "top_rate_pct": top_op.funding_rate * 100 if top_op else 0.0,
            "top_ann_pct": top_op.annualized_with_leverage * 100 if top_op else 0.0,
            "session_pnl": pnl["realized_pnl"],
        }


# ───────────────────────────────────────────────────────────────────────────────
# STANDALONE ENTRYPOINT
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    pt = PaperTrader()
    engine = FundingRateScanner(paper_trader=pt)

    print("\n[Engine 1] Scanning funding rates via CoinGecko derivatives API (live data)...\n")
    cycle_result = engine.run_cycle(auto_enter=True, max_new_positions=3)
    engine.print_opportunity_table(top_n=15)

    print("\nCycle Summary:")
    for k, v in cycle_result.items():
        print(f"  {k}: {v}")

    print("\nPortfolio P&L:")
    pnl = pt.get_pnl(ENGINE_ID)
    for k, v in pnl.items():
        print(f"  {k}: {v}")
