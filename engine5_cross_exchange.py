"""
engine5_cross_exchange.py
==========================
Cross-Exchange Spread Hunter Engine.

Strategy:
  - Fetch prices for 30 tokens from Kraken, CoinGecko, KuCoin, and CoinPaprika
  - Identify tokens where the price on one exchange differs from another
    by more than the total round-trip cost (fees + slippage)
  - Rank by net profit potential
  - Simulate "pre-funded" convergence arbitrage:
      both exchanges already have funds -> no withdrawal delay, instant execution

Data sources (all global, no auth required):
  - Kraken:      GET https://api.kraken.com/0/public/Ticker?pair=...
  - CoinGecko:   GET https://api.coingecko.com/api/v3/simple/price
  - KuCoin:      GET https://api.kucoin.com/api/v1/market/allTickers
  - CoinPaprika: GET https://api.coinpaprika.com/v1/tickers (bulk endpoint)

Pair name normalisation:
  - Kraken uses XBT instead of BTC, and XDG for DOGE
  - KuCoin pairs: BTC-USDT, ETH-USDT, etc.
  - CoinGecko uses lowercase IDs
  - CoinPaprika uses hyphenated slugs (btc-bitcoin, eth-ethereum)

All trades are PAPER (simulated). Live prices from public APIs.

Can be run standalone:
    python engine5_cross_exchange.py
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
    API,
    CAPITAL,
    FEES,
    HTTP,
    MIN_PROFIT,
    RISK,
    SLIPPAGE,
    get_logger,
)
from config import CROSS_EXCHANGE as CE_CFG  # engine5-specific config block
from paper_trader import PaperTrader
from execution_simulator import simulate_cross_exchange_execution, simulate_spot_execution, RealisticFill

logger = get_logger(__name__)

ENGINE_ID = "engine5_cross_exchange"
ENGINE_CAPITAL = CAPITAL[ENGINE_ID]


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------


@dataclass
class ExchangePrice:
    """Price quote for a token from a single exchange."""

    symbol: str           # normalised ticker (e.g. "BTC")
    exchange: str         # e.g. "kraken", "kucoin", "coingecko", "coinpaprika"
    bid: float            # best bid price in USD
    ask: float            # best ask price in USD
    mid: float            # (bid + ask) / 2
    timestamp: str


@dataclass
class SpreadOpportunity:
    """A detected cross-exchange price spread."""

    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_ask: float        # we buy at ask on the cheap exchange
    sell_bid: float       # we sell at bid on the expensive exchange
    gross_spread_pct: float  # (sell_bid - buy_ask) / buy_ask
    buy_fee: float        # as decimal
    sell_fee: float       # as decimal
    withdrawal_fee_usd: float  # 0 for pre-funded (convergence arb)
    capital_usd: float    # notional
    net_profit_usd: float
    net_profit_pct: float
    strategy: str         # "convergence" (pre-funded) or "transfer" (with withdrawal)
    timestamp: str


# ---------------------------------------------------------------------------
# PAIR NAME NORMALISATION MAPS
# ---------------------------------------------------------------------------

# Kraken uses non-standard names for some assets.
# Maps normalised symbol -> Kraken pair name (XYZUSD style).
# Built from the live AssetPairs lookup; this map covers the most common assets.
KRAKEN_PAIR_MAP: Dict[str, str] = {
    "BTC":    "XXBTZUSD",
    "ETH":    "XETHZUSD",
    "XRP":    "XXRPZUSD",
    "LTC":    "XLTCZUSD",
    "DOGE":   "XDGUSD",
    "ADA":    "ADAUSD",
    "SOL":    "SOLUSD",
    "AVAX":   "AVAXUSD",
    "DOT":    "DOTUSD",
    "LINK":   "LINKUSD",
    "UNI":    "UNIUSD",
    "AAVE":   "AAVEUSD",
    "SUSHI":  "SUSHIUSD",
    "CRV":    "CRVUSD",
    "SNX":    "SNXUSD",
    "COMP":   "COMPUSD",
    "YFI":    "YFIUSD",
    "RUNE":   "RUNEUSD",
    "INJ":    "INJUSD",
    "FET":    "FETUSD",
    "RENDER": "RENDERUSD",
    "AR":     "ARUSD",
    "TIA":    "TIAUSD",
    "SEI":    "SEIUSD",
    "SUI":    "SUIUSD",
    "APT":    "APTUSD",
    "OP":     "OPUSD",
    "ARB":    "ARBUSD",
    "NEAR":   "NEARUSD",
}

# CoinGecko ID mapping: symbol -> coingecko_id
COINGECKO_ID_MAP: Dict[str, str] = {
    "BTC":    "bitcoin",
    "ETH":    "ethereum",
    "SOL":    "solana",
    "XRP":    "ripple",
    "DOGE":   "dogecoin",
    "ADA":    "cardano",
    "AVAX":   "avalanche-2",
    "DOT":    "polkadot",
    "LINK":   "chainlink",
    "MATIC":  "matic-network",
    "UNI":    "uniswap",
    "AAVE":   "aave",
    "SUSHI":  "sushi",
    "CRV":    "curve-dao-token",
    "SNX":    "synthetix-network-token",
    "COMP":   "compound-governance-token",
    "MKR":    "maker",
    "YFI":    "yearn-finance",
    "RUNE":   "thorchain",
    "INJ":    "injective-protocol",
    "FET":    "fetch-ai",
    "RENDER": "render-token",
    "AR":     "arweave",
    "TIA":    "celestia",
    "SEI":    "sei-network",
    "SUI":    "sui",
    "APT":    "aptos",
    "OP":     "optimism",
    "ARB":    "arbitrum",
    "NEAR":   "near",
}

# CoinPaprika ID mapping: symbol -> coinpaprika_id
COINPAPRIKA_ID_MAP: Dict[str, str] = {
    "BTC":    "btc-bitcoin",
    "ETH":    "eth-ethereum",
    "SOL":    "sol-solana",
    "XRP":    "xrp-xrp",
    "DOGE":   "doge-dogecoin",
    "ADA":    "ada-cardano",
    "AVAX":   "avax-avalanche",
    "DOT":    "dot-polkadot",
    "LINK":   "link-chainlink",
    "MATIC":  "matic-polygon",
    "UNI":    "uni-uniswap",
    "AAVE":   "aave-aave",
    "SUSHI":  "sushi-sushiswap",
    "CRV":    "crv-curve-dao-token",
    "SNX":    "snx-synthetix-network-token",
    "COMP":   "comp-compound",
    "MKR":    "mkr-maker",
    "YFI":    "yfi-yearnfinance",
    "RUNE":   "rune-thorchain",
    "INJ":    "inj-injective-protocol",
    "FET":    "fet-fetch-ai",
    "AR":     "ar-arweave",
    "TIA":    "tia-celestia",
    "SUI":    "sui-sui",
    "APT":    "apt-aptos",
    "OP":     "op-optimism",
    "ARB":    "arb-arbitrum",
    "NEAR":   "near-near-protocol",
}

# KuCoin uses BASE-USDT format (e.g. BTC-USDT)
# For lookups we construct this on the fly from symbol name


# ---------------------------------------------------------------------------
# HTTP HELPERS
# ---------------------------------------------------------------------------


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
                logger.warning("Rate limited on %s -- sleeping 15s", url)
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


# ---------------------------------------------------------------------------
# MAIN ENGINE CLASS
# ---------------------------------------------------------------------------


class CrossExchangeScanner:
    """
    Engine 5: Cross-Exchange Spread Hunter.

    Fetches prices for a common token list from Kraken, CoinGecko, KuCoin,
    and CoinPaprika, identifies exploitable spreads, and paper-trades the best
    convergence arbitrage opportunities.

    All data sources are public APIs that work globally (no Binance).

    Parameters
    ----------
    paper_trader : PaperTrader
        Shared paper trading ledger.
    capital      : float
        Capital allocated to this engine.
    """

    def __init__(
        self,
        paper_trader: Optional[PaperTrader] = None,
        capital: float = ENGINE_CAPITAL,
    ) -> None:
        self.pt = paper_trader
        self.capital = capital
        self._price_table: Dict[str, Dict[str, ExchangePrice]] = {}  # symbol -> exchange -> price
        self._last_spreads: List[SpreadOpportunity] = []
        self._scan_count = 0
        self._total_executions = 0
        self._token_list: List[str] = CE_CFG["token_list"]

        # KuCoin all-tickers cache (fetched once per cycle)
        self._kucoin_ticker_cache: Dict[str, Dict] = {}  # "BTC-USDT" -> ticker dict

    # -- Data Fetching --------------------------------------------------------

    def _fetch_kraken_prices(
        self, symbols: List[str]
    ) -> Dict[str, ExchangePrice]:
        """
        Fetch bid/ask for symbols from Kraken public Ticker endpoint.
        Uses a lookup table for Kraken pair names.
        Queries in batches of 20.
        Returns {symbol_upper: ExchangePrice}.
        """
        ts = datetime.now(timezone.utc).isoformat()
        result: Dict[str, ExchangePrice] = {}

        pairs_to_fetch: List[Tuple[str, str]] = []
        for sym in symbols:
            k_pair = KRAKEN_PAIR_MAP.get(sym)
            if k_pair:
                pairs_to_fetch.append((k_pair, sym))

        if not pairs_to_fetch:
            return result

        chunk_size = 20
        for i in range(0, len(pairs_to_fetch), chunk_size):
            chunk = pairs_to_fetch[i: i + chunk_size]
            pairs_str = ",".join(p for p, _ in chunk)
            chunk_map = {p: s for p, s in chunk}
            try:
                data, latency = _get(
                    API["kraken_ticker"],
                    params={"pair": pairs_str},
                )
                logger.debug("Kraken ticker chunk fetched in %.0fms", latency)
                time.sleep(HTTP["rate_limit_delay"])

                kraken_result = data.get("result", {})
                for k_pair_ret, ticker in kraken_result.items():
                    sym = chunk_map.get(k_pair_ret)
                    if sym is None:
                        for req_pair, req_sym in chunk_map.items():
                            if (k_pair_ret.upper() == req_pair.upper() or
                                    k_pair_ret.upper().replace("X", "").replace("Z", "") ==
                                    req_pair.upper().replace("X", "").replace("Z", "")):
                                sym = req_sym
                                break
                    if sym is None:
                        stripped = k_pair_ret.upper().rstrip("USD").lstrip("X").lstrip("Z")
                        for req_pair, req_sym in chunk_map.items():
                            if req_sym.upper() == stripped or (
                                req_sym == "BTC" and stripped in ("XBT", "XXBT")
                            ) or (
                                req_sym == "DOGE" and stripped in ("XDG", "XXDG")
                            ):
                                sym = req_sym
                                break

                    if sym is None:
                        continue

                    try:
                        bid = float(ticker["b"][0])
                        ask = float(ticker["a"][0])
                        if bid > 0 and ask > 0:
                            result[sym] = ExchangePrice(
                                symbol=sym,
                                exchange="kraken",
                                bid=bid,
                                ask=ask,
                                mid=(bid + ask) / 2.0,
                                timestamp=ts,
                            )
                    except (ValueError, KeyError, IndexError):
                        pass

            except Exception as exc:
                logger.warning("Kraken ticker chunk failed: %s", exc)

        return result

    def _fetch_kucoin_prices(
        self, symbols: List[str]
    ) -> Dict[str, ExchangePrice]:
        """
        Fetch bid/ask for symbols from KuCoin allTickers endpoint.
        KuCoin uses BASE-USDT format. We fetch all tickers once and cache.
        Returns {symbol_upper: ExchangePrice}.
        """
        ts = datetime.now(timezone.utc).isoformat()
        result: Dict[str, ExchangePrice] = {}

        if not self._kucoin_ticker_cache:
            try:
                data, latency = _get(API["kucoin_all_tickers"])
                logger.debug("KuCoin allTickers fetched in %.0fms", latency)
                time.sleep(HTTP["rate_limit_delay"])

                tickers_list = data.get("data", {}).get("ticker", [])
                for t in tickers_list:
                    sym_name = t.get("symbol", "")
                    self._kucoin_ticker_cache[sym_name] = t
            except Exception as exc:
                logger.warning("KuCoin allTickers fetch failed: %s", exc)
                return result

        for sym in symbols:
            kucoin_pair = f"{sym}-USDT"
            t = self._kucoin_ticker_cache.get(kucoin_pair)
            if not t:
                continue
            try:
                bid = float(t.get("buy") or 0)
                ask = float(t.get("sell") or 0)
                if bid > 0 and ask > 0:
                    result[sym] = ExchangePrice(
                        symbol=sym,
                        exchange="kucoin",
                        bid=bid,
                        ask=ask,
                        mid=(bid + ask) / 2.0,
                        timestamp=ts,
                    )
            except (ValueError, TypeError):
                pass

        return result

    def _fetch_coingecko_prices(
        self, symbols: List[str]
    ) -> Dict[str, ExchangePrice]:
        """
        Fetch mid prices for symbols from CoinGecko simple/price.
        CoinGecko doesn't provide bid/ask, so bid = ask = mid.
        Returns {symbol_upper: ExchangePrice}.
        """
        ts = datetime.now(timezone.utc).isoformat()

        cg_ids = [COINGECKO_ID_MAP[s] for s in symbols if s in COINGECKO_ID_MAP]
        if not cg_ids:
            return {}

        params = {
            "ids": ",".join(cg_ids),
            "vs_currencies": "usd",
        }
        try:
            data, latency = _get(
                API["coingecko_simple_price"],
                params=params,
            )
            logger.debug("CoinGecko prices fetched in %.0fms", latency)
            time.sleep(HTTP["rate_limit_delay"])

            id_to_sym = {v: k for k, v in COINGECKO_ID_MAP.items() if k in symbols}
            result: Dict[str, ExchangePrice] = {}
            for cg_id, vals in data.items():
                p = vals.get("usd")
                if p:
                    sym = id_to_sym.get(cg_id)
                    if sym:
                        try:
                            price = float(p)
                            result[sym] = ExchangePrice(
                                symbol=sym,
                                exchange="coingecko",
                                bid=price,
                                ask=price,
                                mid=price,
                                timestamp=ts,
                            )
                        except (ValueError, TypeError):
                            pass

            return result
        except Exception as exc:
            logger.warning("CoinGecko price fetch failed: %s", exc)
            return {}

    def _fetch_coinpaprika_prices(
        self, symbols: List[str]
    ) -> Dict[str, ExchangePrice]:
        """
        Fetch USD prices from CoinPaprika bulk tickers endpoint.
        CoinPaprika /v1/tickers returns all coins; we filter by known IDs.
        CoinPaprika doesn't provide bid/ask, so bid = ask = price.
        Returns {symbol_upper: ExchangePrice}.
        """
        ts = datetime.now(timezone.utc).isoformat()

        wanted_ids = {COINPAPRIKA_ID_MAP[s] for s in symbols if s in COINPAPRIKA_ID_MAP}
        if not wanted_ids:
            return {}

        try:
            data, latency = _get(API["coinpaprika_tickers"])
            logger.debug("CoinPaprika bulk tickers fetched in %.0fms, %d items", latency, len(data))
            time.sleep(HTTP["rate_limit_delay"])

            id_to_sym = {v: k for k, v in COINPAPRIKA_ID_MAP.items() if k in symbols}

            result: Dict[str, ExchangePrice] = {}
            for coin in data:
                coin_id = coin.get("id", "")
                if coin_id not in wanted_ids:
                    continue
                sym = id_to_sym.get(coin_id)
                if not sym:
                    continue
                try:
                    price = float(
                        coin.get("quotes", {}).get("USD", {}).get("price", 0) or 0
                    )
                    if price > 0:
                        result[sym] = ExchangePrice(
                            symbol=sym,
                            exchange="coinpaprika",
                            bid=price,
                            ask=price,
                            mid=price,
                            timestamp=ts,
                        )
                except (ValueError, TypeError):
                    pass

            return result
        except Exception as exc:
            logger.warning("CoinPaprika price fetch failed: %s", exc)
            return {}

    def fetch_all_prices(
        self,
        token_list: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, ExchangePrice]]:
        """
        Fetch prices for all tokens from all exchanges.

        Sources (all global, no auth):
          1. Kraken   -- bid/ask from Ticker endpoint
          2. KuCoin   -- bid/ask from allTickers endpoint
          3. CoinGecko -- mid price from simple/price
          4. CoinPaprika -- mid price from /v1/tickers

        Parameters
        ----------
        token_list : Symbols to fetch (default: CE_CFG token list)

        Returns
        -------
        {symbol: {exchange: ExchangePrice}}
        """
        if token_list is None:
            token_list = self._token_list

        logger.info("Engine5: fetching prices for %d tokens from Kraken/KuCoin/CoinGecko/CoinPaprika",
                    len(token_list))

        self._kucoin_ticker_cache = {}

        kraken_prices    = self._fetch_kraken_prices(token_list)
        kucoin_prices    = self._fetch_kucoin_prices(token_list)
        coingecko_prices = self._fetch_coingecko_prices(token_list)
        cpaprika_prices  = self._fetch_coinpaprika_prices(token_list)

        price_table: Dict[str, Dict[str, ExchangePrice]] = {sym: {} for sym in token_list}

        for sym in token_list:
            if sym in kraken_prices:
                price_table[sym]["kraken"] = kraken_prices[sym]
            if sym in kucoin_prices:
                price_table[sym]["kucoin"] = kucoin_prices[sym]
            if sym in coingecko_prices:
                price_table[sym]["coingecko"] = coingecko_prices[sym]
            if sym in cpaprika_prices:
                price_table[sym]["coinpaprika"] = cpaprika_prices[sym]

        price_table = {sym: ex for sym, ex in price_table.items() if ex}
        self._price_table = price_table

        logger.info(
            "Engine5: prices fetched | kraken=%d kucoin=%d coingecko=%d coinpaprika=%d tokens_with_data=%d",
            len(kraken_prices), len(kucoin_prices), len(coingecko_prices),
            len(cpaprika_prices), len(price_table),
        )
        return price_table

    # -- Opportunity Analysis -------------------------------------------------

    def find_spreads(
        self,
        price_table: Optional[Dict[str, Dict[str, ExchangePrice]]] = None,
        min_spread_pct: float = CE_CFG["min_spread_pct"],
    ) -> List[SpreadOpportunity]:
        """
        Identify tokens with exploitable price differences across exchanges.

        Parameters
        ----------
        price_table    : Output of fetch_all_prices() -- uses cached if None
        min_spread_pct : Minimum gross spread as decimal (e.g. 0.003 = 0.3%)

        Returns
        -------
        List of SpreadOpportunity sorted by net_profit_usd descending.
        """
        if price_table is None:
            price_table = self._price_table

        opportunities: List[SpreadOpportunity] = []
        ts = datetime.now(timezone.utc).isoformat()
        capital = CE_CFG["simulation_capital_per_trade"]

        for sym, exchanges in price_table.items():
            if len(exchanges) < 2:
                continue

            ex_list = list(exchanges.items())
            for i in range(len(ex_list)):
                for j in range(len(ex_list)):
                    if i == j:
                        continue
                    buy_ex_name, buy_ep = ex_list[i]
                    sell_ex_name, sell_ep = ex_list[j]

                    buy_ask = buy_ep.ask
                    sell_bid = sell_ep.bid

                    if buy_ask <= 0 or sell_bid <= 0:
                        continue

                    gross_spread = (sell_bid - buy_ask) / buy_ask
                    if gross_spread < min_spread_pct:
                        continue

                    buy_fee = FEES.get(buy_ex_name, FEES["default"])["taker"]
                    sell_fee = FEES.get(sell_ex_name, FEES["default"])["taker"]

                    profit = self.calculate_net_profit(
                        spread=gross_spread,
                        capital=capital,
                        buy_fee=buy_fee,
                        sell_fee=sell_fee,
                        withdrawal_fee=0.0,
                    )

                    if profit["net_profit_usd"] > 0:
                        opportunities.append(SpreadOpportunity(
                            symbol=sym,
                            buy_exchange=buy_ex_name,
                            sell_exchange=sell_ex_name,
                            buy_ask=buy_ask,
                            sell_bid=sell_bid,
                            gross_spread_pct=gross_spread,
                            buy_fee=buy_fee,
                            sell_fee=sell_fee,
                            withdrawal_fee_usd=0.0,
                            capital_usd=capital,
                            net_profit_usd=profit["net_profit_usd"],
                            net_profit_pct=profit["net_profit_pct"],
                            strategy="convergence",
                            timestamp=ts,
                        ))

        opportunities.sort(key=lambda x: x.net_profit_usd, reverse=True)
        self._last_spreads = opportunities
        return opportunities

    def calculate_net_profit(
        self,
        spread: float,
        capital: float,
        buy_fee: float,
        sell_fee: float,
        withdrawal_fee: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate net profit after all costs for a cross-exchange trade.

        Parameters
        ----------
        spread          : Gross spread as decimal (e.g. 0.003 = 0.3%)
        capital         : USD notional
        buy_fee         : Taker fee on buy side as decimal
        sell_fee        : Taker fee on sell side as decimal
        withdrawal_fee  : Flat USD withdrawal/transfer cost (0 for pre-funded)

        Returns
        -------
        Dict with gross_profit_usd, total_fees_usd, net_profit_usd, net_profit_pct
        """
        gross_profit = capital * spread
        buy_fee_usd = capital * buy_fee
        sell_fee_usd = capital * sell_fee
        total_fees = buy_fee_usd + sell_fee_usd + withdrawal_fee
        net_profit = gross_profit - total_fees
        net_profit_pct = net_profit / capital if capital > 0 else 0.0

        return {
            "gross_profit_usd": round(gross_profit, 4),
            "buy_fee_usd": round(buy_fee_usd, 4),
            "sell_fee_usd": round(sell_fee_usd, 4),
            "withdrawal_fee_usd": round(withdrawal_fee, 4),
            "total_fees_usd": round(total_fees, 4),
            "net_profit_usd": round(net_profit, 4),
            "net_profit_pct": round(net_profit_pct, 6),
        }

    def simulate_cross_arb(
        self,
        token: str,
        buy_exchange: str,
        sell_exchange: str,
        capital: float,
        buy_price: float,
        sell_price: float,
    ) -> Optional[Any]:
        """
        REALISTIC cross-exchange arbitrage simulation.

        Uses execution_simulator to model:
        - Slippage on both legs
        - Market impact from order book depth
        - Inter-leg execution delay and price drift
        - Rebalancing costs (amortized network fees)
        - Partial fills on thin books
        """
        if self.pt is None:
            return None

        if len(self.pt.get_positions(ENGINE_ID)) >= RISK["max_open_positions"]:
            logger.warning("Engine5: max open positions reached")
            return None

        fill = simulate_cross_exchange_execution(
            symbol=token,
            order_size_usd=capital,
            buy_price=buy_price,
            sell_price=sell_price,
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            pre_funded=True,
        )

        if fill.partial_fill:
            capital = capital * fill.fill_pct

        if fill.fill_price <= 0 or capital <= 0:
            return None

        gross_spread = (sell_price - buy_price) / buy_price if buy_price > 0 else 0.0
        gross_profit = capital * gross_spread
        total_friction = fill.total_friction_usd

        buy_fee_rate = FEES.get(buy_exchange, FEES["default"])["taker"]
        sell_fee_rate = FEES.get(sell_exchange, FEES["default"])["taker"]
        exchange_fees = capital * (buy_fee_rate + sell_fee_rate)

        net_profit = gross_profit - total_friction - exchange_fees

        if net_profit <= 0:
            logger.info(
                "Engine5: Cross-arb unprofitable after friction | %s | gross=$%.4f friction=$%.4f fees=$%.4f",
                token, gross_profit, total_friction, exchange_fees,
            )
            return None

        effective_buy = fill.fill_price
        token_amount = capital / effective_buy if effective_buy > 0 else 0
        effective_sell = effective_buy * (1 + gross_spread) * (1 - fill.friction_pct)

        try:
            open_trade = self.pt.execute_trade(
                engine=ENGINE_ID,
                symbol=f"{token}/USDT",
                side="long",
                amount=token_amount,
                price=effective_buy,
                leverage=1.0,
                order_type="taker",
                metadata={
                    "strategy": "cross_exchange_arb",
                    "realistic_mode": True,
                    "buy_exchange": buy_exchange,
                    "sell_exchange": sell_exchange,
                    "capital_usd": capital,
                    "gross_spread_pct": round(gross_spread * 100, 4),
                    "slippage_cost": fill.slippage_cost_usd,
                    "market_impact": fill.market_impact_usd,
                    "latency_drift": fill.latency_drift_usd,
                    "network_fee": fill.network_fee_usd,
                    "partial_fill": fill.partial_fill,
                },
            )

            close_trade = self.pt.close_position(
                engine=ENGINE_ID,
                position_id=open_trade.position_id,
                exit_price=effective_sell,
                order_type="taker",
                metadata={
                    "strategy": "cross_exchange_arb",
                    "realistic_mode": True,
                    "sell_exchange": sell_exchange,
                    "net_profit_usd": round(net_profit, 4),
                },
            )

            self._total_executions += 1
            logger.info(
                "Engine5: Cross-arb executed (REALISTIC) | %s | %s->%s | gross=$%.4f friction=$%.4f net=$%.4f",
                token, buy_exchange, sell_exchange, gross_profit, total_friction, net_profit,
            )
            return close_trade

        except Exception as exc:
            logger.error("Engine5: execution failed for %s: %s", token, exc)
            return None

    def rank_opportunities(
        self, spreads: Optional[List[SpreadOpportunity]] = None
    ) -> List[SpreadOpportunity]:
        """
        Sort and deduplicate opportunities by net profit potential.
        Only keeps the best opportunity per token (highest net profit).
        """
        if spreads is None:
            spreads = self._last_spreads

        best_per_token: Dict[str, SpreadOpportunity] = {}
        for op in spreads:
            if op.symbol not in best_per_token or \
               op.net_profit_usd > best_per_token[op.symbol].net_profit_usd:
                best_per_token[op.symbol] = op

        ranked = sorted(best_per_token.values(), key=lambda x: x.net_profit_usd, reverse=True)
        return ranked

    # -- Display --------------------------------------------------------------

    def print_spread_table(
        self,
        spreads: Optional[List[SpreadOpportunity]] = None,
        top_n: int = 15,
    ) -> None:
        """Print a formatted table of cross-exchange spread opportunities."""
        if spreads is None:
            spreads = self.rank_opportunities()

        rows = []
        for i, op in enumerate(spreads[:top_n], 1):
            rows.append([
                i,
                op.symbol,
                f"{op.buy_exchange[:7]}@${op.buy_ask:.4f}",
                f"{op.sell_exchange[:7]}@${op.sell_bid:.4f}",
                f"{op.gross_spread_pct * 100:.3f}%",
                f"{(op.buy_fee + op.sell_fee) * 100:.3f}%",
                f"${op.capital_usd:,.0f}",
                f"${op.net_profit_usd:.2f}",
                f"{op.net_profit_pct * 100:.3f}%",
                op.strategy,
            ])

        headers = [
            "#", "Symbol", "Buy", "Sell", "Spread",
            "Fees", "Capital", "Net Profit", "Net %", "Strategy",
        ]
        print("\n" + "-" * 100)
        print("  ENGINE 5: CROSS-EXCHANGE SPREAD OPPORTUNITIES (Kraken/KuCoin/CoinGecko/CoinPaprika)")
        print("-" * 100)
        if rows:
            print(tabulate(rows, headers=headers, tablefmt="simple"))
        else:
            print("  No profitable cross-exchange spreads found.")
        print("-" * 100 + "\n")

    # -- Full Scan Cycle ------------------------------------------------------

    def run_cycle(
        self,
        auto_execute: bool = True,
        max_executions: int = 3,
    ) -> Dict[str, Any]:
        """
        Execute one full fetch -> spread find -> (optional) simulation cycle.

        Parameters
        ----------
        auto_execute   : If True, simulate top spread opportunities
        max_executions : Maximum paper trades to execute this cycle

        Returns
        -------
        Cycle summary dict for dashboard display.
        """
        logger.info("Engine5: starting cross-exchange spread scan cycle...")
        t_start = time.perf_counter()

        try:
            price_table = self.fetch_all_prices()
        except Exception as exc:
            logger.error("Engine5: price fetch failed: %s", exc)
            pnl_data = self.pt.get_pnl(ENGINE_ID) if self.pt else {"realized_pnl": 0.0}
            return {
                "engine": ENGINE_ID,
                "error": str(exc),
                "session_pnl": pnl_data["realized_pnl"],
            }

        spreads = self.find_spreads(price_table)
        ranked = self.rank_opportunities(spreads)

        executions = 0
        if auto_execute and ranked and self.pt is not None:
            per_trade_capital = min(
                self.capital / max(RISK["max_open_positions"], 1),
                CE_CFG["simulation_capital_per_trade"],
            )
            for op in ranked[:max_executions]:
                if op.net_profit_pct >= MIN_PROFIT[ENGINE_ID]:
                    try:
                        self.simulate_cross_arb(
                            token=op.symbol,
                            buy_exchange=op.buy_exchange,
                            sell_exchange=op.sell_exchange,
                            capital=per_trade_capital,
                            buy_price=op.buy_ask,
                            sell_price=op.sell_bid,
                        )
                        executions += 1
                    except Exception as exc:
                        logger.warning("Engine5: execution failed for %s: %s", op.symbol, exc)

        pnl = self.pt.get_pnl(ENGINE_ID) if self.pt else {"realized_pnl": 0.0}
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)
        self._scan_count += 1

        top = ranked[0] if ranked else None

        return {
            "engine": ENGINE_ID,
            "tokens_scanned": len(price_table),
            "spreads_found": len(spreads),
            "unique_tokens_with_spread": len(ranked),
            "executions_this_cycle": executions,
            "total_executions": self._total_executions,
            "top_token": top.symbol if top else "N/A",
            "top_spread_pct": top.gross_spread_pct * 100 if top else 0.0,
            "top_net_profit_usd": top.net_profit_usd if top else 0.0,
            "top_buy_exchange": top.buy_exchange if top else "N/A",
            "top_sell_exchange": top.sell_exchange if top else "N/A",
            "elapsed_ms": elapsed_ms,
            "session_pnl": pnl["realized_pnl"],
        }


# ---------------------------------------------------------------------------
# STANDALONE ENTRYPOINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pt = PaperTrader()
    engine = CrossExchangeScanner(paper_trader=pt)

    print("\n[Engine 5] Scanning cross-exchange spreads (Kraken/KuCoin/CoinGecko/CoinPaprika)...\n")
    cycle_result = engine.run_cycle(auto_execute=True, max_executions=3)
    engine.print_spread_table(top_n=10)

    print("\nCycle Summary:")
    for k, v in cycle_result.items():
        print(f"  {k}: {v}")

    print("\nPortfolio P&L:")
    pnl = pt.get_pnl(ENGINE_ID)
    for k, v in pnl.items():
        print(f"  {k}: {v}")
