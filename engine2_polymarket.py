"""
engine2_polymarket.py
======================
Polymarket Reward Farmer Engine.

Strategy:
  - Scan Polymarket prediction markets for active liquidity reward programs
  - Estimate share of rewards based on quadratic scoring rules
  - Rank markets by estimated daily return per dollar deployed
  - Simulate liquidity provision (paper-trade both sides of the order book)
  - Monitor for adverse fills (one-sided fill = directional risk)

All trades are PAPER (simulated). Live data from Polymarket public APIs.

Can be run standalone:
    python engine2_polymarket.py
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
    get_logger,
)
from paper_trader import PaperTrader

logger = get_logger(__name__)

ENGINE_ID = "engine2_polymarket"
ENGINE_CAPITAL = CAPITAL[ENGINE_ID]

# Polymarket USDC uses 6 decimals; convert to float dollars throughout
USDC_DECIMALS = 1_000_000


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OrderBookSummary:
    """Summary metrics from a Polymarket CLOB order book."""

    yes_token_id: str
    no_token_id: str
    best_bid: float         # highest YES bid price (0–1)
    best_ask: float         # lowest YES ask price (0–1)
    spread: float           # ask - bid
    bid_depth_usdc: float   # total USDC on bid side (top 5 levels)
    ask_depth_usdc: float   # total USDC on ask side (top 5 levels)
    mid_price: float        # (bid + ask) / 2
    timestamp: str


@dataclass
class MarketOpportunity:
    """A Polymarket market evaluated for liquidity reward farming."""

    condition_id: str
    market_slug: str
    question: str
    yes_token_id: str
    no_token_id: str
    end_date_iso: str
    is_active: bool
    rewards_daily_rate: float       # estimated daily reward pool USD (0 if unknown)
    total_volume_usdc: float
    liquidity_usdc: float
    spread: float
    bid_depth_usdc: float
    ask_depth_usdc: float
    mid_price: float
    competition_score: float        # 0–1, higher = more competition
    estimated_daily_return_pct: float   # net of competition
    estimated_apr: float
    deployment_risk: str            # "low" | "medium" | "high"


@dataclass
class LiquidityDeployment:
    """Tracks an open simulated MM deployment on a Polymarket market."""

    deployment_id: str
    condition_id: str
    question: str
    yes_token_id: str
    capital_deployed: float         # USD
    yes_bid_price: float
    yes_ask_price: float
    yes_bid_size: float             # USDC notional
    yes_ask_size: float
    deployed_at: str
    filled_yes_bid: bool = False    # adverse fill tracking
    filled_yes_ask: bool = False
    accumulated_rewards: float = 0.0
    periods_active: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# HTTP HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _get(url: str, params: Optional[Dict] = None, retries: int = HTTP["max_retries"]) -> Tuple[Any, float]:
    """GET with retry + latency measurement. Returns (data, latency_ms)."""
    headers = {"User-Agent": HTTP["user_agent"]}
    for attempt in range(1, retries + 1):
        try:
            t0 = time.perf_counter()
            resp = requests.get(url, params=params, headers=headers, timeout=HTTP["timeout"])
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


class PolymarketFarmer:
    """
    Engine 2: Polymarket Reward Farmer.

    Scans Polymarket for markets offering liquidity rewards, estimates
    reward share using quadratic scoring, and simulates market-making.

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
        self._active_deployments: Dict[str, LiquidityDeployment] = {}
        self._last_opportunities: Optional[List[MarketOpportunity]] = None
        self._scan_count = 0

    # ── Data Fetching ────────────────────────────────────────────────────────────────────────────

    def _fetch_gamma_markets(self, limit: int = 100, offset: int = 0) -> Tuple[List[Dict], float]:
        """
        Fetch markets from Gamma API (metadata + reward info).
        https://gamma-api.polymarket.com/markets
        """
        params = {
            "limit": limit,
            "offset": offset,
            "active": "true",
            "closed": "false",
            "archived": "false",
        }
        data, lat = _get(API["polymarket_markets"], params=params)
        time.sleep(HTTP["rate_limit_delay"])
        # Gamma API can return a list or a dict with 'data' key
        if isinstance(data, dict):
            return data.get("data", []) or data.get("markets", []), lat
        return data, lat

    def _fetch_clob_markets(self, limit: int = 100) -> Tuple[List[Dict], float]:
        """
        Fetch markets from CLOB API (trading activity, token IDs).
        https://clob.polymarket.com/markets
        """
        params = {"limit": limit}
        data, lat = _get(API["polymarket_clob_markets"], params=params)
        time.sleep(HTTP["rate_limit_delay"])
        if isinstance(data, dict):
            return data.get("data", []), lat
        return data if isinstance(data, list) else [], lat

    def _fetch_order_book(self, token_id: str) -> Tuple[Optional[OrderBookSummary], float]:
        """
        Fetch CLOB order book for a token.
        https://clob.polymarket.com/book?token_id={token_id}
        """
        try:
            data, lat = _get(API["polymarket_clob_book"], params={"token_id": token_id})
        except Exception as exc:
            logger.debug("Order book fetch failed for %s: %s", token_id, exc)
            return None, 0.0

        time.sleep(HTTP["rate_limit_delay"] / 2)

        bids: List[Dict] = data.get("bids", [])
        asks: List[Dict] = data.get("asks", [])

        if not bids or not asks:
            return None, lat

        try:
            # Best bid = highest bid price
            sorted_bids = sorted(bids, key=lambda x: float(x.get("price", 0)), reverse=True)
            sorted_asks = sorted(asks, key=lambda x: float(x.get("price", 0)))

            best_bid = float(sorted_bids[0]["price"])
            best_ask = float(sorted_asks[0]["price"])
            spread = best_ask - best_bid
            mid = (best_bid + best_ask) / 2.0

            # Depth: sum top 5 levels
            bid_depth = sum(
                float(b.get("size", 0)) for b in sorted_bids[:5]
            )
            ask_depth = sum(
                float(a.get("size", 0)) for a in sorted_asks[:5]
            )

            summary = OrderBookSummary(
                yes_token_id=token_id,
                no_token_id="",  # filled later
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread,
                bid_depth_usdc=bid_depth,
                ask_depth_usdc=ask_depth,
                mid_price=mid,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            return summary, lat
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("Order book parse error for %s: %s", token_id, exc)
            return None, lat

    # ── Analysis ────────────────────────────────────────────────────────────────────────────

    def calculate_competition_score(
        self, bid_depth_usdc: float, ask_depth_usdc: float
    ) -> float:
        """
        Estimate competition level from order book depth.

        A thin book (low depth) = low competition (good for us).
        A deep book = high competition (smaller reward share).

        Normalized to [0, 1] where 1 = extremely competitive.

        Parameters
        ----------
        bid_depth_usdc : Total USDC on bid side (top 5 levels)
        ask_depth_usdc : Total USDC on ask side (top 5 levels)
        """
        total_depth = bid_depth_usdc + ask_depth_usdc
        if total_depth <= 0:
            return 0.1  # empty book = low competition but also low activity

        # Normalise against $50k reference (above $50k = very competitive)
        competition = min(total_depth / 50_000.0, 1.0)
        return round(competition, 4)

    def estimate_reward_share(
        self,
        capital: float,
        competition_score: float,
        reward_pool: float,
        spread: float,
        max_spread: float = 0.10,
    ) -> float:
        """
        Estimate our share of the daily reward pool using quadratic scoring.

        Polymarket's quadratic reward rule weights providers closer to the
        mid-price more heavily:

            score_i = ((max_spread - distance_i) / max_spread)^2 * order_size_i

        We approximate our distance from mid as spread/2 (tight enough to earn).

        Parameters
        ----------
        capital          : USD capital deployed (split between bid and ask)
        competition_score: 0–1 competition level
        reward_pool      : Daily reward pool in USD
        spread           : Our quoted spread
        max_spread       : Maximum spread eligible for rewards (default 10%)

        Returns
        -------
        Estimated daily USD reward
        """
        if reward_pool <= 0 or capital <= 0:
            return 0.0

        distance = spread / 2.0
        if distance >= max_spread:
            return 0.0

        # Quadratic score for our orders
        our_score = ((max_spread - distance) / max_spread) ** 2 * capital

        # Estimated total competition score (our score + others based on depth)
        estimated_total_score = our_score / max(1.0 - competition_score, 0.01)

        reward_share = our_score / estimated_total_score if estimated_total_score > 0 else 0.0
        return round(reward_share * reward_pool, 4)

    def scan_sponsored_markets(self) -> List[MarketOpportunity]:
        """
        Fetch Polymarket markets and identify those with active reward pools.

        Returns
        -------
        List of MarketOpportunity sorted by estimated APR descending.
        """
        logger.info("Engine2: scanning Polymarket markets...")
        t_start = time.perf_counter()

        try:
            gamma_markets, lat_gamma = self._fetch_gamma_markets(limit=100)
        except Exception as exc:
            logger.error("Failed to fetch Gamma markets: %s", exc)
            return []

        try:
            clob_markets, lat_clob = self._fetch_clob_markets(limit=100)
        except Exception as exc:
            logger.warning("CLOB markets unavailable: %s", exc)
            clob_markets = []
            lat_clob = 0.0

        # Build CLOB lookup by condition_id
        clob_by_condition: Dict[str, Dict] = {}
        for cm in clob_markets:
            cid = cm.get("condition_id") or cm.get("conditionId", "")
            if cid:
                clob_by_condition[cid] = cm

        opportunities: List[MarketOpportunity] = []

        for market in gamma_markets:
            try:
                condition_id = market.get("conditionId") or market.get("condition_id", "")
                if not condition_id:
                    continue

                # Parse rewards — Gamma markets may expose rewardsMinSize, rewardsDailyRate
                daily_rewards = float(market.get("rewardsDailyRate", 0) or 0)
                rewards_active = market.get("rewardsActive") or market.get("rewardsMinSize", 0)

                # Include all active markets; reward estimate will be 0 for non-reward markets
                # (they may still be profitable via spread capture if volume is high)
                is_active = market.get("active", False) and not market.get("closed", True)
                if not is_active:
                    continue

                # Get token IDs from market data
                # NOTE: Gamma API returns clobTokenIds as a JSON *string*, not a list
                import json as _json
                raw_tokens = market.get("clobTokenIds") or market.get("tokens") or []
                if isinstance(raw_tokens, str):
                    try:
                        raw_tokens = _json.loads(raw_tokens)
                    except (ValueError, TypeError):
                        raw_tokens = []
                tokens = raw_tokens if isinstance(raw_tokens, list) else []
                if len(tokens) < 2:
                    continue
                if isinstance(tokens[0], dict):
                    yes_token_id = str(tokens[0].get("token_id") or tokens[0].get("tokenId", ""))
                    no_token_id = str(tokens[1].get("token_id") or tokens[1].get("tokenId", ""))
                else:
                    yes_token_id = str(tokens[0])
                    no_token_id = str(tokens[1])

                if not yes_token_id:
                    continue

                # Fetch order book (rate-limited — only for top markets by volume)
                volume = float(market.get("volume24hr") or market.get("volume", 0) or 0)
                liquidity = float(market.get("liquidity") or 0)

                ob, lat_ob = self._fetch_order_book(yes_token_id)

                if ob:
                    spread = ob.spread
                    bid_depth = ob.bid_depth_usdc
                    ask_depth = ob.ask_depth_usdc
                    mid_price = ob.mid_price
                else:
                    # Use market data estimates
                    spread = 0.05          # default 5c spread
                    bid_depth = liquidity / 2.0
                    ask_depth = liquidity / 2.0
                    mid_price = float(market.get("lastTradePrice") or market.get("midpoint") or 0.5)

                competition_score = self.calculate_competition_score(bid_depth, ask_depth)

                # Estimate reward pool if not explicitly given
                if daily_rewards == 0 and rewards_active:
                    # Estimate from volume: assume ~0.1% of 24h volume as rewards
                    daily_rewards = volume * 0.001

                our_daily_reward = self.estimate_reward_share(
                    capital=min(self.capital * 0.10, 5000.0),   # sim with 10% of capital
                    competition_score=competition_score,
                    reward_pool=daily_rewards,
                    spread=spread,
                )

                deployed_capital = min(self.capital * 0.10, 5000.0)
                daily_return_pct = (our_daily_reward / deployed_capital) if deployed_capital > 0 else 0.0
                apr = daily_return_pct * 365

                # Risk assessment
                if competition_score > 0.7:
                    risk = "high"
                elif competition_score > 0.4:
                    risk = "medium"
                else:
                    risk = "low"

                # Resolve end date
                end_date = (
                    market.get("endDate")
                    or market.get("endDateIso")
                    or "N/A"
                )

                opp = MarketOpportunity(
                    condition_id=condition_id,
                    market_slug=market.get("slug") or market.get("marketSlug", ""),
                    question=market.get("question", "")[:80],
                    yes_token_id=yes_token_id,
                    no_token_id=no_token_id,
                    end_date_iso=str(end_date),
                    is_active=is_active,
                    rewards_daily_rate=daily_rewards,
                    total_volume_usdc=volume,
                    liquidity_usdc=liquidity,
                    spread=spread,
                    bid_depth_usdc=bid_depth,
                    ask_depth_usdc=ask_depth,
                    mid_price=mid_price,
                    competition_score=competition_score,
                    estimated_daily_return_pct=daily_return_pct,
                    estimated_apr=apr,
                    deployment_risk=risk,
                )
                opportunities.append(opp)

            except (KeyError, ValueError, TypeError) as exc:
                logger.debug("Skipping market %s: %s", market.get("conditionId", "?"), exc)

        # Sort by estimated APR descending; secondary sort by volume
        opportunities.sort(
            key=lambda x: (x.estimated_apr, x.total_volume_usdc),
            reverse=True,
        )

        elapsed = (time.perf_counter() - t_start) * 1000
        logger.info(
            "Engine2: scanned %d markets in %.0fms | %d opportunities",
            len(gamma_markets), elapsed, len(opportunities),
        )
        self._last_opportunities = opportunities
        self._scan_count += 1
        return opportunities

    def rank_markets(self) -> List[MarketOpportunity]:
        """
        Return markets ranked by estimated daily return per dollar deployed.
        Triggers a fresh scan if none cached.
        """
        if self._last_opportunities is None:
            return self.scan_sponsored_markets()
        return self._last_opportunities

    # ── Position Management ────────────────────────────────────────────────────────────────────

    def simulate_deployment(
        self, market: MarketOpportunity, capital: float
    ) -> Optional[LiquidityDeployment]:
        """
        Paper-trade a liquidity provision deployment on a Polymarket market.

        Simulates posting both bid and ask orders at the inside spread.

        Parameters
        ----------
        market  : MarketOpportunity to deploy on
        capital : USD capital to deploy

        Returns
        -------
        LiquidityDeployment tracker if successful, else None.
        """
        if market.condition_id in self._active_deployments:
            logger.debug("Already deployed in %s", market.condition_id)
            return None

        if len(self._active_deployments) >= RISK["max_open_positions"]:
            logger.warning("Max deployments reached in %s", ENGINE_ID)
            return None

        half = capital / 2.0
        spread_half = market.spread / 2.0
        yes_bid = market.mid_price - spread_half
        yes_ask = market.mid_price + spread_half

        # Clamp to valid range [0.01, 0.99]
        yes_bid = max(0.01, min(0.99, yes_bid))
        yes_ask = max(0.01, min(0.99, yes_ask))

        # Simulate two open orders (YES bid and YES ask)
        # Each order is half the capital at current prices
        bid_size = half / yes_bid if yes_bid > 0 else 0
        ask_size = half / yes_ask if yes_ask > 0 else 0

        bid_trade = self.pt.execute_trade(
            engine=ENGINE_ID,
            symbol=market.condition_id[:12],
            side="long",
            amount=bid_size,
            price=yes_bid,
            leverage=1.0,
            order_type="maker",
            metadata={
                "leg": "yes_bid",
                "market_question": market.question[:40],
                "condition_id": market.condition_id,
                "token_id": market.yes_token_id,
            },
        )

        ask_trade = self.pt.execute_trade(
            engine=ENGINE_ID,
            symbol=market.condition_id[:12],
            side="short",
            amount=ask_size,
            price=yes_ask,
            leverage=1.0,
            order_type="maker",
            metadata={
                "leg": "yes_ask",
                "market_question": market.question[:40],
                "condition_id": market.condition_id,
                "token_id": market.yes_token_id,
            },
        )

        deployment_id = f"{bid_trade.position_id}:{ask_trade.position_id}"
        deployment = LiquidityDeployment(
            deployment_id=deployment_id,
            condition_id=market.condition_id,
            question=market.question,
            yes_token_id=market.yes_token_id,
            capital_deployed=capital,
            yes_bid_price=yes_bid,
            yes_ask_price=yes_ask,
            yes_bid_size=bid_size,
            yes_ask_size=ask_size,
            deployed_at=datetime.now(timezone.utc).isoformat(),
        )

        self._active_deployments[market.condition_id] = deployment
        logger.info(
            "Deployed $%.2f to '%s' | bid=%.3f ask=%.3f | est. APR=%.1f%%",
            capital,
            market.question[:50],
            yes_bid,
            yes_ask,
            market.estimated_apr * 100,
        )
        return deployment

    def monitor_fills(self) -> Dict[str, Any]:
        """
        Simulate potential adverse fills based on order book dynamics.

        For each active deployment:
          - Both sides filled (bid + ask) = completed round trip = ok, earn spread
          - Only bid filled = long exposure (directional risk)
          - Only ask filled = short exposure (directional risk)
          - Neither filled = collecting rewards but no spread income

        Fill probability is simulated from order book depth competition.

        Returns
        -------
        Summary of fill events this cycle.
        """
        fill_events: List[Dict] = []
        rewards_earned = 0.0

        for cid, dep in list(self._active_deployments.items()):
            dep.periods_active += 1

            # Find original opportunity for competition info
            opp = next(
                (o for o in (self._last_opportunities or []) if o.condition_id == cid),
                None,
            )
            competition = opp.competition_score if opp else 0.5

            # Fill probability: higher competition = more fills (tighter market)
            fill_prob = max(0.05, competition * 0.4)

            bid_filled = random.random() < fill_prob
            ask_filled = random.random() < fill_prob

            # Simulate reward accrual (period = every call = approx 1hr)
            if opp:
                hourly_reward = opp.estimated_daily_return_pct * dep.capital_deployed / 24.0
            else:
                hourly_reward = 0.0
            dep.accumulated_rewards += hourly_reward
            rewards_earned += hourly_reward

            # Log reward income
            if hourly_reward > 0:
                parts = dep.deployment_id.split(":")
                bid_pid = parts[0] if parts else ""
                if bid_pid:
                    self.pt.log_periodic_income(
                        engine=ENGINE_ID,
                        position_id=bid_pid,
                        amount_usd=hourly_reward,
                        income_type="reward",
                        metadata={"condition_id": cid, "period": dep.periods_active},
                    )

            event = {
                "condition_id": cid,
                "question": dep.question[:50],
                "bid_filled": bid_filled,
                "ask_filled": ask_filled,
                "fill_type": (
                    "round_trip" if bid_filled and ask_filled
                    else "long_exposure" if bid_filled
                    else "short_exposure" if ask_filled
                    else "no_fill"
                ),
                "hourly_reward": round(hourly_reward, 4),
                "accumulated_rewards": round(dep.accumulated_rewards, 4),
                "periods_active": dep.periods_active,
            }
            fill_events.append(event)
            logger.debug(
                "Fill check [%s]: bid=%s ask=%s reward=$%.4f",
                cid[:12], bid_filled, ask_filled, hourly_reward,
            )

        return {
            "deployments_checked": len(self._active_deployments),
            "total_rewards_this_period": round(rewards_earned, 4),
            "fill_events": fill_events,
        }

    # ── Display ────────────────────────────────────────────────────────────────────────────

    def print_market_table(
        self, opportunities: Optional[List[MarketOpportunity]] = None, top_n: int = 15
    ) -> None:
        """Print formatted table of top Polymarket opportunities."""
        if opportunities is None:
            opportunities = self.rank_markets()

        rows = []
        for i, op in enumerate(opportunities[:top_n], 1):
            rows.append([
                i,
                op.question[:45] + ("…" if len(op.question) > 45 else ""),
                f"${op.rewards_daily_rate:.2f}",
                f"{op.estimated_apr * 100:.1f}%",
                f"{op.spread * 100:.1f}¢",
                f"{op.competition_score * 100:.0f}%",
                f"${op.total_volume_usdc:,.0f}",
                op.deployment_risk.upper(),
            ])

        headers = [
            "#", "Market Question", "Daily Pool", "Est. APR",
            "Spread", "Competition", "24h Volume", "Risk",
        ]
        print("\n" + "─" * 80)
        print("  ENGINE 2: POLYMARKET REWARD OPPORTUNITIES")
        print("─" * 80)
        if rows:
            print(tabulate(rows, headers=headers, tablefmt="simple"))
        else:
            print("  No market opportunities found.")
        print("─" * 80 + "\n")

    # ── Full Scan Cycle ────────────────────────────────────────────────────────────────

    def run_cycle(
        self, auto_deploy: bool = True, max_new_deployments: int = 3
    ) -> Dict[str, Any]:
        """
        Execute one full scan → ranking → (optional) deployment cycle.

        Parameters
        ----------
        auto_deploy          : If True, auto-simulate top market deployments
        max_new_deployments  : Maximum new deployments to open this cycle

        Returns
        -------
        Cycle summary dict for dashboard display.
        """
        ops = self.scan_sponsored_markets()
        top_op = ops[0] if ops else None

        new_deployments = 0
        if auto_deploy and ops:
            per_deployment_capital = self.capital / max(RISK["max_open_positions"], 1)
            for op in ops[:max_new_deployments]:
                if op.condition_id not in self._active_deployments:
                    if op.estimated_apr >= MIN_PROFIT["engine2_polymarket"] * 365:
                        dep = self.simulate_deployment(op, per_deployment_capital)
                        if dep:
                            new_deployments += 1

        # Monitor fills and collect rewards for existing deployments
        fill_report = self.monitor_fills()

        pnl = self.pt.get_pnl(ENGINE_ID)

        return {
            "engine": ENGINE_ID,
            "markets_scanned": len(ops),
            "new_deployments": new_deployments,
            "active_deployments": len(self._active_deployments),
            "top_market": top_op.question[:50] if top_op else "N/A",
            "top_market_apr_pct": top_op.estimated_apr * 100 if top_op else 0.0,
            "rewards_this_period": fill_report["total_rewards_this_period"],
            "session_pnl": pnl["realized_pnl"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pt = PaperTrader()
    engine = PolymarketFarmer(paper_trader=pt)

    print("\n[Engine 2] Scanning Polymarket markets (live data)...\n")
    cycle_result = engine.run_cycle(auto_deploy=True, max_new_deployments=3)
    engine.print_market_table(top_n=10)

    print("\nCycle Summary:")
    for k, v in cycle_result.items():
        if k != "fill_events":
            print(f"  {k}: {v}")

    print("\nPortfolio P&L:")
    pnl = pt.get_pnl(ENGINE_ID)
    for k, v in pnl.items():
        print(f"  {k}: {v}")
