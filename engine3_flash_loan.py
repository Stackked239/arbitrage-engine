"""
engine3_flash_loan.py
======================
Flash Loan Arbitrage Scanner Engine.

Strategy:
  - Fetch token prices from multiple DEX/aggregator sources:
      CoinGecko (top 50 by volume), DEXScreener, DeFi Llama
  - Find tokens where the price spread between sources exceeds the
    flash loan fee + gas cost (min_spread_pct threshold)
  - Simulate flash loan execution: borrow → buy cheap → sell high → repay
  - Track: opportunities found, simulated executions, simulated P&L
  - Also scan triangular DEX paths on simulated pool data

All trades are PAPER (simulated). Live prices via public APIs.

Can be run standalone:
    python engine3_flash_loan.py
"""

from __future__ import annotations

import random
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from tabulate import tabulate

from config import (
    CAPITAL,
    FEES,
    HTTP,
    MIN_PROFIT,
    RISK,
    SLIPPAGE,
    get_logger,
)
from config import FLASH_LOAN as FL_CFG  # engine3-specific config block
from paper_trader import PaperTrader
from execution_simulator import simulate_flash_loan_execution, RealisticFill

logger = get_logger(__name__)

ENGINE_ID = "engine3_flash_loan"
ENGINE_CAPITAL = CAPITAL[ENGINE_ID]


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TokenPrice:
    """Price quote for a token from a single source."""

    token_id: str          # CoinGecko ID (e.g. "bitcoin")
    symbol: str            # ticker (e.g. "BTC")
    source: str            # e.g. "coingecko", "defillama", "dexscreener"
    price_usd: float
    volume_24h: float      # 0 if unavailable
    timestamp: str


@dataclass
class FlashOpportunity:
    """A detected cross-DEX price discrepancy worth simulating."""

    token_id: str
    symbol: str
    buy_source: str
    sell_source: str
    buy_price: float
    sell_price: float
    spread_pct: float      # (sell - buy) / buy
    borrow_amount: float   # simulated flash loan size in USD
    flash_fee_usd: float   # borrow_amount * flash_fee_rate
    gas_estimate_usd: float
    gross_profit_usd: float   # (spread_pct * borrow_amount)
    net_profit_usd: float     # gross - flash_fee - gas
    net_profit_pct: float     # net / borrow_amount
    timestamp: str


@dataclass
class TriangularPath:
    """A triangular arbitrage path across simulated DEX pools."""

    path: List[str]          # e.g. ["WETH", "USDC", "USDT", "WETH"]
    start_amount: float      # USD
    end_amount: float        # USD after completing the triangle
    profit_usd: float
    profit_pct: float
    dex: str                 # which DEX these pools belong to
    timestamp: str


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


class FlashLoanScanner:
    """
    Engine 3: Flash Loan Arbitrage Scanner.

    Scans multiple public price sources for cross-DEX price discrepancies,
    simulates flash loan executions, and paper-trades profitable opportunities.

    Parameters
    ----------
    paper_trader : PaperTrader
        Shared paper trading ledger.
    capital      : float
        Capital allocated to this engine.
    """

    def __init__(
        self,
        paper_trader: PaperTrader,
        capital: float = ENGINE_CAPITAL,
    ) -> None:
        self.pt = paper_trader
        self.capital = capital
        self._last_prices: Dict[str, Dict[str, TokenPrice]] = {}  # symbol -> source -> price
        self._last_opportunities: List[FlashOpportunity] = []
        self._scan_count = 0
        self._total_simulated_executions = 0
        self._token_list: List[Dict[str, str]] = []   # [{id, symbol}, ...]

    # ── Data Fetching ────────────────────────────────────────────────────────────────────────────

    def _fetch_top_tokens_coingecko(self, top_n: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch top tokens by market cap / volume from CoinGecko.

        Returns list of dicts with keys: id, symbol, current_price,
        total_volume, market_cap, price_change_percentage_24h.
        """
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "volume_desc",
            "per_page": top_n,
            "page": 1,
            "sparkline": "false",
        }
        try:
            data, latency = _get(url, params=params)
            logger.debug("CoinGecko top tokens fetched in %.0fms", latency)
            time.sleep(HTTP["rate_limit_delay"])
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning("CoinGecko top tokens fetch failed: %s", exc)
            return []

    def _fetch_prices_coingecko(self, coin_ids: List[str]) -> Dict[str, float]:
        """
        Fetch USD prices for a list of CoinGecko IDs.
        Returns {coin_id: price_usd}.
        """
        if not coin_ids:
            return {}
        # CoinGecko allows up to ~100 IDs per request
        chunk_size = 80
        prices: Dict[str, float] = {}
        for i in range(0, len(coin_ids), chunk_size):
            chunk = coin_ids[i : i + chunk_size]
            params = {
                "ids": ",".join(chunk),
                "vs_currencies": "usd",
                "include_24hr_vol": "true",
            }
            try:
                data, latency = _get(
                    "https://api.coingecko.com/api/v3/simple/price",
                    params=params,
                )
                for cid, vals in data.items():
                    p = vals.get("usd")
                    if p:
                        prices[cid] = float(p)
                time.sleep(HTTP["rate_limit_delay"])
            except Exception as exc:
                logger.warning("CoinGecko price fetch failed for chunk: %s", exc)
        return prices

    def _fetch_prices_defillama(self, token_ids: List[str]) -> Dict[str, float]:
        """
        Fetch prices from DeFi Llama coins API.
        token_ids should be in format: "coingecko:bitcoin", "coingecko:ethereum", etc.
        Returns {raw_id: price_usd}.
        """
        if not token_ids:
            return {}
        # DeFi Llama accepts comma-joined tokens
        joined = ",".join(token_ids[:50])  # max 50 per request
        url = f"https://coins.llama.fi/prices/current/{joined}"
        try:
            data, latency = _get(url)
            logger.debug("DeFi Llama prices fetched in %.0fms", latency)
            time.sleep(HTTP["rate_limit_delay"])
            coins = data.get("coins", {})
            result: Dict[str, float] = {}
            for key, val in coins.items():
                p = val.get("price")
                if p:
                    result[key] = float(p)
            return result
        except Exception as exc:
            logger.warning("DeFi Llama price fetch failed: %s", exc)
            return {}

    def _fetch_prices_dexscreener(self, addresses: List[str]) -> Dict[str, float]:
        """
        Fetch DEX prices from DEXScreener by token contract addresses.
        Returns {address_lower: price_usd}.
        """
        if not addresses:
            return {}
        # DEXScreener allows up to 30 addresses per request
        results: Dict[str, float] = {}
        chunk_size = 30
        for i in range(0, len(addresses), chunk_size):
            chunk = addresses[i : i + chunk_size]
            joined = ",".join(chunk)
            url = f"https://api.dexscreener.com/latest/dex/tokens/{joined}"
            try:
                data, latency = _get(url)
                logger.debug("DEXScreener prices fetched in %.0fms", latency)
                pairs = data.get("pairs") or []
                for pair in pairs:
                    # Use the base token address as key
                    token_addr = (pair.get("baseToken") or {}).get("address", "").lower()
                    price_str = pair.get("priceUsd")
                    if token_addr and price_str:
                        try:
                            p = float(price_str)
                            # Keep the first (highest liquidity) pair per token
                            if token_addr not in results:
                                results[token_addr] = p
                        except (ValueError, TypeError):
                            pass
                time.sleep(HTTP["rate_limit_delay"])
            except Exception as exc:
                logger.warning("DEXScreener fetch failed for chunk: %s", exc)
        return results

    # ── Core Scanner ───────────────────────────────────────────────────────────────────────

    def scan_dex_prices(self, token_list: Optional[List[Dict[str, str]]] = None) -> Dict[str, Dict[str, TokenPrice]]:
        """
        Fetch prices for token_list from CoinGecko and DeFi Llama.

        Parameters
        ----------
        token_list : List of {id: coingecko_id, symbol: ticker}
                     If None, fetches the top 50 tokens by volume first.

        Returns
        -------
        {symbol: {source: TokenPrice}}
        """
        if token_list is None:
            if not self._token_list:
                raw = self._fetch_top_tokens_coingecko(top_n=FL_CFG["top_n_tokens"])
                self._token_list = [
                    {"id": t["id"], "symbol": t["symbol"].upper()}
                    for t in raw
                    if t.get("id") and t.get("symbol")
                ]
            token_list = self._token_list

        if not token_list:
            logger.warning("Engine3: No tokens to scan")
            return {}

        coin_ids = [t["id"] for t in token_list]
        ts = datetime.now(timezone.utc).isoformat()

        # --- Source 1: CoinGecko simple/price ---
        cg_prices = self._fetch_prices_coingecko(coin_ids)

        # --- Source 2: DeFi Llama (coingecko: prefix) ---
        dl_ids = [f"coingecko:{cid}" for cid in coin_ids]
        dl_raw = self._fetch_prices_defillama(dl_ids)
        # Remap: "coingecko:bitcoin" -> "bitcoin"
        dl_prices: Dict[str, float] = {}
        for k, v in dl_raw.items():
            if k.startswith("coingecko:"):
                dl_prices[k[len("coingecko:"):]] = v

        # Build structured price map
        price_map: Dict[str, Dict[str, TokenPrice]] = {}
        for token in token_list:
            cid = token["id"]
            sym = token["symbol"]
            price_map[sym] = {}

            cg_p = cg_prices.get(cid)
            if cg_p:
                price_map[sym]["coingecko"] = TokenPrice(
                    token_id=cid,
                    symbol=sym,
                    source="coingecko",
                    price_usd=cg_p,
                    volume_24h=0.0,
                    timestamp=ts,
                )

            dl_p = dl_prices.get(cid)
            if dl_p:
                price_map[sym]["defillama"] = TokenPrice(
                    token_id=cid,
                    symbol=sym,
                    source="defillama",
                    price_usd=dl_p,
                    volume_24h=0.0,
                    timestamp=ts,
                )

        self._last_prices = price_map
        logger.info(
            "Engine3: prices fetched for %d tokens (cg=%d, dl=%d)",
            len(token_list), len(cg_prices), len(dl_prices),
        )
        return price_map

    def find_discrepancies(
        self,
        price_map: Optional[Dict[str, Dict[str, TokenPrice]]] = None,
        min_spread_pct: float = FL_CFG["min_spread_pct"],
    ) -> List[FlashOpportunity]:
        """
        Identify tokens where price differs between sources by >= min_spread_pct.

        Parameters
        ----------
        price_map      : Output of scan_dex_prices(); uses cached if None.
        min_spread_pct : Minimum spread as decimal (e.g. 0.005 = 0.5%)

        Returns
        -------
        List of FlashOpportunity sorted by net_profit_usd descending.
        """
        if price_map is None:
            price_map = self._last_prices

        if not price_map:
            return []

        opportunities: List[FlashOpportunity] = []
        ts = datetime.now(timezone.utc).isoformat()

        for sym, sources in price_map.items():
            source_list = [(src, tp.price_usd) for src, tp in sources.items() if tp.price_usd > 0]
            if len(source_list) < 2:
                continue  # Need at least 2 sources to compare

            # Find cheapest and most expensive source
            source_list.sort(key=lambda x: x[1])
            buy_src, buy_price = source_list[0]
            sell_src, sell_price = source_list[-1]

            spread_pct = (sell_price - buy_price) / buy_price if buy_price > 0 else 0.0
            if spread_pct < min_spread_pct:
                continue

            # Simulate flash loan amounts (range from config)
            for borrow_amt in FL_CFG["borrow_amounts"]:
                opp = self.calculate_flash_profit(
                    borrow_amount=borrow_amt,
                    buy_price=buy_price,
                    sell_price=sell_price,
                    flash_fee=FL_CFG["flash_fee_rate"],
                    gas_estimate_usd=FL_CFG["gas_estimate_eth_usd"],
                )
                if opp["net_profit_usd"] > 0:
                    opportunities.append(FlashOpportunity(
                        token_id=sym,
                        symbol=sym,
                        buy_source=buy_src,
                        sell_source=sell_src,
                        buy_price=buy_price,
                        sell_price=sell_price,
                        spread_pct=spread_pct,
                        borrow_amount=borrow_amt,
                        flash_fee_usd=opp["flash_fee_usd"],
                        gas_estimate_usd=opp["gas_estimate_usd"],
                        gross_profit_usd=opp["gross_profit_usd"],
                        net_profit_usd=opp["net_profit_usd"],
                        net_profit_pct=opp["net_profit_pct"],
                        timestamp=ts,
                    ))
                    break  # Take the first profitable borrow size

        opportunities.sort(key=lambda x: x.net_profit_usd, reverse=True)
        self._last_opportunities = opportunities
        return opportunities

    def calculate_flash_profit(
        self,
        borrow_amount: float,
        buy_price: float,
        sell_price: float,
        flash_fee: float = 0.0009,    # 0.09% Aave V3 default
        gas_estimate_usd: float = 5.0,
    ) -> Dict[str, float]:
        """
        Calculate net profit from a flash loan arbitrage trade.

        Parameters
        ----------
        borrow_amount     : USD value borrowed
        buy_price         : Price on cheap source
        sell_price        : Price on expensive source
        flash_fee         : Flash loan fee as decimal (0.0009 = 0.09%)
        gas_estimate_usd  : Estimated gas cost in USD

        Returns
        -------
        Dict with: gross_profit_usd, flash_fee_usd, gas_estimate_usd,
                   net_profit_usd, net_profit_pct
        """
        spread = (sell_price - buy_price) / buy_price if buy_price > 0 else 0.0
        gross_profit = borrow_amount * spread
        flash_fee_usd = borrow_amount * flash_fee
        # DEX swap fee on buy + sell legs (use default taker fee)
        dex_fees = borrow_amount * FEES["default"]["taker"] * 2
        net_profit = gross_profit - flash_fee_usd - gas_estimate_usd - dex_fees
        net_profit_pct = net_profit / borrow_amount if borrow_amount > 0 else 0.0

        return {
            "gross_profit_usd": round(gross_profit, 4),
            "flash_fee_usd": round(flash_fee_usd, 4),
            "dex_fees_usd": round(dex_fees, 4),
            "gas_estimate_usd": round(gas_estimate_usd, 4),
            "net_profit_usd": round(net_profit, 4),
            "net_profit_pct": round(net_profit_pct, 6),
        }

    def simulate_flash_execution(
        self,
        token: str,
        buy_dex: str,
        sell_dex: str,
        borrow_amount: float,
        buy_price: float,
        sell_price: float,
    ) -> Optional[Any]:
        """
        REALISTIC flash loan arbitrage simulation.

        Uses execution_simulator to model:
        - MEV bot competition (85% of opportunities get front-run)
        - Dynamic gas costs
        - DEX slippage (much higher than CEX)
        - AMM price impact

        Returns
        -------
        Trade record, or None if blocked by MEV / unprofitable.
        """
        if len(self.pt.get_positions(ENGINE_ID)) >= RISK["max_open_positions"]:
            logger.warning("Engine3: max open positions reached")
            return None

        spread_pct = (sell_price - buy_price) / buy_price if buy_price > 0 else 0.0

        # Use realistic execution simulator
        fill = simulate_flash_loan_execution(
            symbol=token,
            borrow_amount_usd=borrow_amount,
            spread_pct=spread_pct,
            buy_price=buy_price,
            sell_price=sell_price,
        )

        # MEV blocked — opportunity was front-run
        if fill.blocked_by_mev:
            logger.info(
                "Engine3: Flash arb BLOCKED by MEV | %s | borrow=$%.0f | spread=%.3f%%",
                token, borrow_amount, spread_pct * 100,
            )
            return None

        # Calculate realistic net profit
        effective_spread = spread_pct
        gross_profit = borrow_amount * effective_spread
        total_friction = fill.total_friction_usd
        net_profit = gross_profit - total_friction

        # If unprofitable after realistic costs, skip
        if net_profit <= 0:
            logger.info(
                "Engine3: Flash arb unprofitable after friction | %s | gross=$%.2f friction=$%.2f",
                token, gross_profit, total_friction,
            )
            return None

        import uuid
        position_id = str(uuid.uuid4())

        trade = self.pt.log_periodic_income(
            engine=ENGINE_ID,
            position_id=position_id,
            amount_usd=net_profit,
            income_type="flash_loan_arb",
            metadata={
                "symbol": f"{token}/USD",
                "strategy": "flash_loan_arb",
                "realistic_mode": True,
                "buy_dex": buy_dex,
                "sell_dex": sell_dex,
                "borrow_amount": borrow_amount,
                "spread_pct": round(spread_pct * 100, 4),
                "gross_profit_usd": round(gross_profit, 4),
                "slippage_cost": fill.slippage_cost_usd,
                "market_impact": fill.market_impact_usd,
                "gas_and_fees": fill.network_fee_usd,
                "total_friction": round(total_friction, 4),
                "net_profit_usd": round(net_profit, 4),
                "mev_blocked": False,
            },
        )

        self._total_simulated_executions += 1

        logger.info(
            "Engine3: Flash arb executed (REALISTIC) | %s | borrow=$%.0f | gross=$%.2f friction=$%.2f net=$%.4f",
            token, borrow_amount, gross_profit, total_friction, net_profit,
        )
        return trade

    def scan_triangular_dex(
        self, base_tokens: Optional[List[str]] = None
    ) -> List[TriangularPath]:
        """
        Simulate triangular arbitrage across DEX pool prices.

        Uses simulated pool pricing with small random deviations to model
        real-world DEX pool imbalances. In production this would query
        Uniswap v3 on-chain pool state.

        Parameters
        ----------
        base_tokens : Tokens to use as path anchors (default from config)

        Returns
        -------
        List of profitable TriangularPath, sorted by profit_pct descending.
        """
        if base_tokens is None:
            base_tokens = FL_CFG["triangular_base_tokens"]

        ts = datetime.now(timezone.utc).isoformat()
        paths: List[TriangularPath] = []

        # Use CoinGecko prices as base for pool simulation
        cg_prices = self._fetch_prices_coingecko(
            [t.lower() for t in FL_CFG["triangular_token_ids"]]
        )

        # Map symbol -> price using known IDs
        sym_price = {
            "WETH": cg_prices.get("ethereum", 2800.0),
            "USDC": 1.0,
            "USDT": 1.0,
            "WBTC": cg_prices.get("wrapped-bitcoin") or cg_prices.get("bitcoin", 50000.0),
            "DAI":  cg_prices.get("dai", 1.0),
            "LINK": cg_prices.get("chainlink", 15.0),
            "UNI":  cg_prices.get("uniswap", 8.0),
            "AAVE": cg_prices.get("aave", 100.0),
        }

        dex_name = "uniswap_v3_simulated"
        taker_fee = FEES["default"]["taker"]

        for base in base_tokens:
            if base not in sym_price:
                continue
            base_price = sym_price[base]
            if base_price <= 0:
                continue

            # Try all 2-hop paths: base → A → B → base
            candidates = [sym for sym in sym_price if sym != base and sym_price[sym] > 0]
            for mid1 in candidates:
                for mid2 in candidates:
                    if mid2 == mid1 or mid2 == base:
                        continue

                    start_usd = 10_000.0  # simulate $10k starting amount

                    # Each leg has a random DEX pool premium/discount
                    imbalance1 = 1 + random.uniform(-0.003, 0.003)
                    imbalance2 = 1 + random.uniform(-0.003, 0.003)
                    imbalance3 = 1 + random.uniform(-0.003, 0.003)

                    # Simulate executing the triangle
                    # Leg 1: Convert base to mid1 (units of mid1 received)
                    units_base = start_usd / sym_price[base]
                    units_mid1 = units_base * (sym_price[base] / sym_price[mid1]) * imbalance1 * (1 - taker_fee)
                    # Leg 2: Convert mid1 to mid2 (units of mid2 received)
                    units_mid2 = units_mid1 * (sym_price[mid1] / sym_price[mid2]) * imbalance2 * (1 - taker_fee)
                    # Leg 3: Convert mid2 back to base (units of base received)
                    units_base_end = units_mid2 * (sym_price[mid2] / sym_price[base]) * imbalance3 * (1 - taker_fee)
                    end_usd = units_base_end * sym_price[base]

                    profit_usd = end_usd - start_usd
                    profit_pct = profit_usd / start_usd

                    # Sanity cap: reject any path claiming > 5% profit
                    # (DEX imbalances of 0.3% cannot produce multi-thousand-% returns)
                    if profit_pct > 0.05:
                        continue

                    if profit_pct >= FL_CFG["triangular_min_profit_pct"]:
                        paths.append(TriangularPath(
                            path=[base, mid1, mid2, base],
                            start_amount=start_usd,
                            end_amount=round(end_usd, 4),
                            profit_usd=round(profit_usd, 4),
                            profit_pct=round(profit_pct, 6),
                            dex=dex_name,
                            timestamp=ts,
                        ))

        paths.sort(key=lambda x: x.profit_pct, reverse=True)
        return paths[:20]  # return top 20 paths

    # ── Display ───────────────────────────────────────────────────────────────────────────

    def print_opportunity_table(
        self,
        opportunities: Optional[List[FlashOpportunity]] = None,
        top_n: int = 15,
    ) -> None:
        """Print a formatted table of top flash loan opportunities."""
        if opportunities is None:
            opportunities = self._last_opportunities

        rows = []
        for i, op in enumerate(opportunities[:top_n], 1):
            rows.append([
                i,
                op.symbol,
                op.buy_source,
                op.sell_source,
                f"${op.buy_price:.4f}",
                f"${op.sell_price:.4f}",
                f"{op.spread_pct * 100:.3f}%",
                f"${op.borrow_amount:,.0f}",
                f"${op.net_profit_usd:.2f}",
                f"{op.net_profit_pct * 100:.3f}%",
            ])

        headers = [
            "#", "Symbol", "Buy@", "Sell@", "Buy $", "Sell $",
            "Spread", "Borrow", "Net Profit", "Net %",
        ]
        print("\n" + "─" * 90)
        print("  ENGINE 3: FLASH LOAN ARBITRAGE OPPORTUNITIES")
        print("─" * 90)
        if rows:
            print(tabulate(rows, headers=headers, tablefmt="simple"))
        else:
            print("  No profitable flash loan opportunities found.")
        print("─" * 90 + "\n")

    # ── Full Scan Cycle ────────────────────────────────────────────────────────────────

    def run_cycle(
        self,
        auto_execute: bool = True,
        max_executions: int = 3,
    ) -> Dict[str, Any]:
        """
        Execute one full scan → discrepancy → (optional) simulation cycle.

        Parameters
        ----------
        auto_execute   : If True, simulate top opportunities
        max_executions : Maximum paper trades to execute this cycle

        Returns
        -------
        Cycle summary dict for dashboard display.
        """
        logger.info("Engine3: starting flash loan scan cycle...")
        t_start = time.perf_counter()

        # Fetch prices
        try:
            price_map = self.scan_dex_prices()
        except Exception as exc:
            logger.error("Engine3: price scan failed: %s", exc)
            return {
                "engine": ENGINE_ID,
                "error": str(exc),
                "session_pnl": self.pt.get_pnl(ENGINE_ID)["realized_pnl"],
            }

        # Find opportunities
        opportunities = self.find_discrepancies(price_map)

        # Scan triangular paths (on DEX pool simulation)
        try:
            tri_paths = self.scan_triangular_dex()
        except Exception as exc:
            logger.warning("Engine3: triangular scan failed: %s", exc)
            tri_paths = []

        # Execute top opportunities
        executions = 0
        if auto_execute and opportunities:
            for op in opportunities[:max_executions]:
                if op.net_profit_usd > 0:
                    try:
                        self.simulate_flash_execution(
                            token=op.symbol,
                            buy_dex=op.buy_source,
                            sell_dex=op.sell_source,
                            borrow_amount=op.borrow_amount,
                            buy_price=op.buy_price,
                            sell_price=op.sell_price,
                        )
                        executions += 1
                    except Exception as exc:
                        logger.warning("Engine3: execution failed for %s: %s", op.symbol, exc)

        pnl = self.pt.get_pnl(ENGINE_ID)
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

        top_op = opportunities[0] if opportunities else None

        return {
            "engine": ENGINE_ID,
            "tokens_scanned": len(price_map),
            "opportunities_found": len(opportunities),
            "triangular_paths_found": len(tri_paths),
            "executions_this_cycle": executions,
            "total_executions": self._total_simulated_executions,
            "top_token": top_op.symbol if top_op else "N/A",
            "top_spread_pct": top_op.spread_pct * 100 if top_op else 0.0,
            "top_net_profit_usd": top_op.net_profit_usd if top_op else 0.0,
            "best_tri_path": " → ".join(tri_paths[0].path) if tri_paths else "N/A",
            "best_tri_profit_pct": tri_paths[0].profit_pct * 100 if tri_paths else 0.0,
            "elapsed_ms": elapsed_ms,
            "session_pnl": pnl["realized_pnl"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pt = PaperTrader()
    engine = FlashLoanScanner(paper_trader=pt)

    print("\n[Engine 3] Scanning DEX prices for flash loan opportunities (live data)...\n")
    cycle_result = engine.run_cycle(auto_execute=True, max_executions=3)
    engine.print_opportunity_table(top_n=10)

    print("\nCycle Summary:")
    for k, v in cycle_result.items():
        print(f"  {k}: {v}")

    print("\nPortfolio P&L:")
    pnl = pt.get_pnl(ENGINE_ID)
    for k, v in pnl.items():
        print(f"  {k}: {v}")
