"""
execution_simulator.py
=======================
Realistic execution simulation layer.

Models real-world execution friction:
1. ORDER BOOK DEPTH / LIQUIDITY
2. SLIPPAGE MODEL (non-uniform)
3. EXECUTION LATENCY
4. PARTIAL FILLS
5. FLASH LOAN REALISM (MEV competition)
6. CROSS-EXCHANGE REALISM
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any


LIQUIDITY_TIERS: Dict[str, float] = {
    "BTC":     500000,
    "ETH":     300000,
    "SOL":     100000,
    "XRP":      80000,
    "DOGE":     50000,
    "ADA":      40000,
    "AVAX":     30000,
    "DOT":      30000,
    "LINK":     40000,
    "UNI":      20000,
    "AAVE":     15000,
    "MKR":      10000,
    "default":  15000,
}

VOLATILITY_TIERS: Dict[str, float] = {
    "BTC":     0.60,
    "ETH":     0.75,
    "SOL":     1.20,
    "XRP":     1.00,
    "DOGE":    1.50,
    "ADA":     1.00,
    "AVAX":    1.10,
    "default": 1.00,
}

TRANSFER_TIMES: Dict[str, int] = {
    "BTC":     30,
    "ETH":     5,
    "SOL":     2,
    "XRP":     1,
    "default": 15,
}

NETWORK_FEES: Dict[str, float] = {
    "BTC":     15.0,
    "ETH":      8.0,
    "SOL":      0.01,
    "XRP":      0.10,
    "DOGE":     2.0,
    "default":  5.0,
}

FLASH_LOAN_MEV_RATE = 0.85
FLASH_LOAN_GAS_BASE = 15.0
FLASH_LOAN_GAS_VARIANCE = 30.0

API_LATENCY_MIN_MS = 50
API_LATENCY_MAX_MS = 500
PRICE_DRIFT_PER_SEC = 0.000054


@dataclass
class RealisticFill:
    """Result of a realistic execution simulation."""
    fill_price: float
    fill_amount: float
    fill_pct: float
    slippage_cost_usd: float
    market_impact_usd: float
    latency_drift_usd: float
    network_fee_usd: float
    estimated_liquidity: float
    execution_latency_ms: float
    partial_fill: bool
    blocked_by_mev: bool

    @property
    def total_friction_usd(self) -> float:
        return (self.slippage_cost_usd + self.market_impact_usd +
                self.latency_drift_usd + self.network_fee_usd)

    @property
    def friction_pct(self) -> float:
        notional = self.fill_price * self.fill_amount
        if notional <= 0:
            return 0.0
        return self.total_friction_usd / notional


def _get_liquidity(symbol: str, exchange: str = "default") -> float:
    base = symbol.split("/")[0].split("-")[0].upper()
    base_liquidity = LIQUIDITY_TIERS.get(base, LIQUIDITY_TIERS["default"])
    exchange_mult = {
        "kraken": 0.7,
        "kucoin": 0.5,
        "coingecko": 0.3,
        "coinpaprika": 0.3,
        "polymarket": 0.2,
        "default": 0.5,
    }
    mult = exchange_mult.get(exchange.lower(), exchange_mult["default"])
    return base_liquidity * mult


def _get_volatility(symbol: str) -> float:
    base = symbol.split("/")[0].split("-")[0].upper()
    return VOLATILITY_TIERS.get(base, VOLATILITY_TIERS["default"])


def simulate_spot_execution(
    symbol: str,
    side: str,
    order_size_usd: float,
    quoted_price: float,
    exchange: str = "default",
) -> RealisticFill:
    liquidity = _get_liquidity(symbol, exchange)
    vol = _get_volatility(symbol)

    base_slip = random.uniform(0.0001, 0.0005)
    size_ratio = order_size_usd / max(liquidity, 1.0)
    market_impact_pct = min(size_ratio ** 2 * 0.10, 0.05)
    vol_factor = vol / 0.60
    total_slip_pct = (base_slip + market_impact_pct) * vol_factor

    latency_ms = random.uniform(API_LATENCY_MIN_MS, API_LATENCY_MAX_MS)
    latency_sec = latency_ms / 1000.0
    drift_pct = PRICE_DRIFT_PER_SEC * latency_sec * vol_factor
    drift_direction = 1 if random.random() < 0.6 else -1
    drift_pct *= drift_direction

    fill_pct = 1.0
    if order_size_usd > liquidity * 0.5:
        fill_pct = min(1.0, liquidity * 0.5 / order_size_usd)
        fill_pct = max(fill_pct, 0.1)

    if side == "buy":
        fill_price = quoted_price * (1 + total_slip_pct + max(drift_pct, 0))
    else:
        fill_price = quoted_price * (1 - total_slip_pct - max(drift_pct, 0))

    fill_amount_usd = order_size_usd * fill_pct
    fill_amount = fill_amount_usd / fill_price if fill_price > 0 else 0

    return RealisticFill(
        fill_price=fill_price,
        fill_amount=fill_amount,
        fill_pct=fill_pct,
        slippage_cost_usd=round(order_size_usd * base_slip, 4),
        market_impact_usd=round(order_size_usd * market_impact_pct, 4),
        latency_drift_usd=round(order_size_usd * abs(drift_pct) if drift_direction == 1 else 0, 4),
        network_fee_usd=0.0,
        estimated_liquidity=liquidity,
        execution_latency_ms=round(latency_ms, 1),
        partial_fill=fill_pct < 1.0,
        blocked_by_mev=False,
    )


def simulate_flash_loan_execution(
    symbol: str,
    borrow_amount_usd: float,
    spread_pct: float,
    buy_price: float,
    sell_price: float,
) -> RealisticFill:
    if random.random() < FLASH_LOAN_MEV_RATE:
        return RealisticFill(
            fill_price=0.0, fill_amount=0.0, fill_pct=0.0,
            slippage_cost_usd=0.0, market_impact_usd=0.0,
            latency_drift_usd=0.0, network_fee_usd=0.0,
            estimated_liquidity=0.0, execution_latency_ms=0.0,
            partial_fill=False, blocked_by_mev=True,
        )

    gas_cost = FLASH_LOAN_GAS_BASE + random.uniform(0, FLASH_LOAN_GAS_VARIANCE)
    dex_slip_buy = random.uniform(0.002, 0.01)
    dex_slip_sell = random.uniform(0.002, 0.01)
    total_dex_slip = dex_slip_buy + dex_slip_sell

    estimated_pool_liq = _get_liquidity(symbol, "dex") * 5
    amm_impact = min(borrow_amount_usd / max(estimated_pool_liq * 2, 1.0), 0.05)

    effective_spread = spread_pct - total_dex_slip - amm_impact
    net_profit = borrow_amount_usd * effective_spread - gas_cost
    flash_fee = borrow_amount_usd * 0.0009
    net_profit -= flash_fee

    fill_price = buy_price * (1 + dex_slip_buy)
    fill_amount = borrow_amount_usd / fill_price if fill_price > 0 else 0

    return RealisticFill(
        fill_price=fill_price,
        fill_amount=fill_amount,
        fill_pct=1.0 if net_profit > 0 else 0.0,
        slippage_cost_usd=round(borrow_amount_usd * total_dex_slip, 4),
        market_impact_usd=round(borrow_amount_usd * amm_impact, 4),
        latency_drift_usd=0.0,
        network_fee_usd=round(gas_cost + flash_fee, 4),
        estimated_liquidity=estimated_pool_liq,
        execution_latency_ms=0.0,
        partial_fill=False,
        blocked_by_mev=False,
    )


def simulate_cross_exchange_execution(
    symbol: str,
    order_size_usd: float,
    buy_price: float,
    sell_price: float,
    buy_exchange: str,
    sell_exchange: str,
    pre_funded: bool = True,
) -> RealisticFill:
    base_sym = symbol.split("/")[0].split("-")[0].upper()
    buy_fill = simulate_spot_execution(symbol, "buy", order_size_usd, buy_price, buy_exchange)
    sell_fill = simulate_spot_execution(symbol, "sell", order_size_usd, sell_price, sell_exchange)

    inter_leg_delay_sec = random.uniform(0.05, 2.0)
    vol = _get_volatility(symbol)
    vol_factor = vol / 0.60
    inter_leg_drift_pct = PRICE_DRIFT_PER_SEC * inter_leg_delay_sec * vol_factor * 3
    if random.random() < 0.5:
        inter_leg_drift_pct *= -1
    drift_cost = order_size_usd * abs(inter_leg_drift_pct)

    net_fee = NETWORK_FEES.get(base_sym, NETWORK_FEES["default"])
    rebalance_fee = net_fee if not pre_funded else net_fee / 5.0

    return RealisticFill(
        fill_price=buy_fill.fill_price,
        fill_amount=buy_fill.fill_amount,
        fill_pct=min(buy_fill.fill_pct, sell_fill.fill_pct),
        slippage_cost_usd=round(buy_fill.slippage_cost_usd + sell_fill.slippage_cost_usd, 4),
        market_impact_usd=round(buy_fill.market_impact_usd + sell_fill.market_impact_usd, 4),
        latency_drift_usd=round(buy_fill.latency_drift_usd + sell_fill.latency_drift_usd + drift_cost, 4),
        network_fee_usd=round(rebalance_fee, 4),
        estimated_liquidity=min(buy_fill.estimated_liquidity, sell_fill.estimated_liquidity),
        execution_latency_ms=buy_fill.execution_latency_ms + sell_fill.execution_latency_ms,
        partial_fill=buy_fill.partial_fill or sell_fill.partial_fill,
        blocked_by_mev=False,
    )


def simulate_polymarket_execution(
    order_size_usd: float,
    mid_price: float,
    spread: float,
    bid_depth_usd: float,
    ask_depth_usd: float,
) -> RealisticFill:
    total_depth = bid_depth_usd + ask_depth_usd
    size_ratio = order_size_usd / total_depth if total_depth > 0 else 1.0
    impact_pct = min(size_ratio * 0.15, 0.10)
    base_slip = random.uniform(0.005, 0.02)
    polygon_gas = random.uniform(0.02, 0.10)

    fill_price = mid_price * (1 + base_slip + impact_pct)
    fill_pct = max(min(1.0, total_depth / max(order_size_usd, 1.0)), 0.1)

    return RealisticFill(
        fill_price=fill_price,
        fill_amount=order_size_usd * fill_pct / fill_price if fill_price > 0 else 0,
        fill_pct=fill_pct,
        slippage_cost_usd=round(order_size_usd * base_slip, 4),
        market_impact_usd=round(order_size_usd * impact_pct, 4),
        latency_drift_usd=0.0,
        network_fee_usd=round(polygon_gas, 4),
        estimated_liquidity=total_depth,
        execution_latency_ms=random.uniform(200, 1000),
        partial_fill=fill_pct < 1.0,
        blocked_by_mev=False,
    )
