"""
config.py
==========
Central configuration for the Arbitrage Engine paper trading system.
All monetary values in USD. All rates as decimals (0.001 = 0.1%).
"""

from __future__ import annotations
import logging
import os
from typing import Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# PAPER TRADING MODE
# ─────────────────────────────────────────────────────────────────────────────

PAPER_TRADING: bool = True          # ALWAYS True — no real money moves
DATA_DIR: str = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
TRADES_FILE: str = os.path.join(DATA_DIR, "paper_trades.json")
LOG_FILE: str = os.path.join(DATA_DIR, "engine.log")

# ─────────────────────────────────────────────────────────────────────────────
# CAPITAL ALLOCATION  (total = $10,000)
# ─────────────────────────────────────────────────────────────────────────────

CAPITAL: Dict[str, float] = {
    "engine1_funding_rate":  3000.00,
    "engine2_polymarket":    3000.00,
    "engine3_flash_loan":    1500.00,
    "engine4_triangular":    1500.00,
    "engine5_cross_exchange": 1000.00,
}

TOTAL_CAPITAL: float = sum(CAPITAL.values())   # $10,000.00

# ─────────────────────────────────────────────────────────────────────────────
# FEE STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

FEES: Dict[str, Dict[str, float]] = {
    "crypto_com": {
        "maker": 0.00075,   # 0.075%
        "taker": 0.00075,   # 0.075%
    },
    "binance": {
        "maker": 0.001,     # 0.10%
        "taker": 0.001,     # 0.10%
        "futures_maker": 0.0002,  # 0.02%
        "futures_taker": 0.0005,  # 0.05%
    },
    "coinbase": {
        "maker": 0.004,     # 0.40%
        "taker": 0.006,     # 0.60%
    },
    "kraken": {
        "maker": 0.0016,    # 0.16%
        "taker": 0.0026,    # 0.26%
    },
    "polymarket": {
        "maker": 0.00,      # no maker fee
        "taker": 0.00,      # no taker fee (rewards-based model)
    },
    "default": {
        "maker": 0.001,
        "taker": 0.002,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# SLIPPAGE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

SLIPPAGE: Dict[str, float] = {
    "min": 0.0005,    # 0.05%  — realistic minimum for taker orders
    "max": 0.002,     # 0.20%  — common for mid-cap tokens
    "default": 0.001, # 0.10%
    "large_order": 0.005,   # 0.50% for >$5k orders
}

LARGE_ORDER_THRESHOLD: float = 5000.0   # USD

# ─────────────────────────────────────────────────────────────────────────────
# REALISTIC EXECUTION MODE
# ─────────────────────────────────────────────────────────────────────────────

REALISTIC_MODE: bool = True  # Use execution_simulator for all trades

# ─────────────────────────────────────────────────────────────────────────────
# MINIMUM PROFIT THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

MIN_PROFIT: Dict[str, float] = {
    "engine1_funding_rate":   0.0001,    # 0.01% per 8hr funding cycle
    "engine2_polymarket":     0.002,     # 0.20% daily net
    "engine3_flash_loan":     0.003,     # 0.30% per trade (covers gas + fees)
    "engine4_triangular":     0.001,     # 0.10% per round-trip
    "engine5_cross_exchange": 0.002,     # 0.20% spread after fees
}

# Funding rate tiers (per 8-hour period)
FUNDING_RATE_TIERS: Dict[str, float] = {
    "baseline":    0.0001,   # 0.01%  — minimum worth tracking
    "moderate":    0.0003,   # 0.03%  — decent opportunity
    "strong":      0.0005,   # 0.05%  — strong signal
    "exceptional": 0.0010,   # 0.10%  — exceptional / act immediately
}

# ─────────────────────────────────────────────────────────────────────────────
# RISK PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

RISK: Dict[str, Any] = {
    "max_position_size_pct": 0.30,      # max 30% of engine capital per position
    "max_drawdown_pct": 0.10,           # halt engine at -10% drawdown
    "max_daily_loss_pct": 0.05,         # halt for day at -5% daily loss
    "max_open_positions": 10,           # per engine
    "max_leverage": 5,                  # global leverage cap
    "default_leverage": 3,             # default for funding rate engine
    "stop_loss_pct": 0.02,             # 2% stop-loss on delta-neutral leg
    "take_profit_pct": 0.05,           # 5% take-profit
    "min_order_size_usd": 10.0,        # minimum simulated order size
}

# ─────────────────────────────────────────────────────────────────────────────
# ENGINE 3 — FLASH LOAN ARBITRAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

FLASH_LOAN: Dict[str, Any] = {
    # Minimum price spread between sources before considering a flash loan
    "min_spread_pct": 0.005,          # 0.5%

    # Aave V3 flash loan fee (0.09%).  Uniswap v3 flash loans are 0.05%.
    "flash_fee_rate": 0.0009,

    # Simulated borrow sizes to test (USD)
    "borrow_amounts": [100000, 250000, 500000],

    # Gas cost estimates
    "gas_price_gwei_eth": 30,
    "gas_price_gwei_l2": 0.01,
    "gas_units_flash": 300000,
    "gas_estimate_eth_usd": 5.0,

    # Top-N tokens to scan from CoinGecko
    "top_n_tokens": 50,

    # Triangular DEX scan config
    "triangular_base_tokens": ["WETH", "USDC", "USDT"],
    "triangular_min_profit_pct": 0.001,
    "triangular_token_ids": [
        "ethereum", "usd-coin", "tether", "wrapped-bitcoin",
        "dai", "chainlink", "uniswap", "aave",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# ENGINE 4 — TRIANGULAR ARBITRAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TRIANGULAR: Dict[str, Any] = {
    "base_currencies": ["USD", "BTC", "ETH"],
    "min_profit_pct": 0.001,
    "min_volume_usd": 100000.0,
    "simulation_capital": 10000.0,
    "max_triangles_evaluated": 5000,
}

# ─────────────────────────────────────────────────────────────────────────────
# ENGINE 5 — CROSS-EXCHANGE SPREAD CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CROSS_EXCHANGE: Dict[str, Any] = {
    "token_list": [
        "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "DOT",
        "LINK", "MATIC", "UNI", "AAVE", "SUSHI", "CRV", "SNX",
        "COMP", "MKR", "YFI", "RUNE", "INJ", "FET", "RENDER",
        "AR", "TIA", "SEI", "SUI", "APT", "OP", "ARB", "NEAR",
    ],
    "min_spread_pct": 0.003,
    "simulation_capital_per_trade": 1000.0,
    "withdrawal_fees": {
        "BTC":  10.0,
        "ETH":  5.0,
        "USDT": 1.0,
        "default": 2.0,
    },
    "exchanges": ["kraken", "coingecko", "kucoin", "coinpaprika"],
}

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

API: Dict[str, str] = {
    "defillama_prices":          "https://coins.llama.fi/prices/current/",
    "dexscreener_tokens":        "https://api.dexscreener.com/latest/dex/tokens/",
    "kraken_ticker":             "https://api.kraken.com/0/public/Ticker",
    "kraken_asset_pairs":        "https://api.kraken.com/0/public/AssetPairs",
    "kraken_ohlc":               "https://api.kraken.com/0/public/OHLC",
    "kucoin_all_tickers":        "https://api.kucoin.com/api/v1/market/allTickers",
    "kucoin_ticker":             "https://api.kucoin.com/api/v1/market/orderbook/level1",
    "coinpaprika_ticker":        "https://api.coinpaprika.com/v1/tickers/{coin_id}",
    "coinpaprika_tickers":       "https://api.coinpaprika.com/v1/tickers",
    "coingecko_derivatives":     "https://api.coingecko.com/api/v3/derivatives",
    "coingecko_deriv_exchanges": "https://api.coingecko.com/api/v3/derivatives/exchanges",
    "coingecko_coins_markets":   "https://api.coingecko.com/api/v3/coins/markets",
    "coingecko_simple_price":    "https://api.coingecko.com/api/v3/simple/price",
    "polymarket_markets":        "https://gamma-api.polymarket.com/markets",
    "polymarket_clob_markets":   "https://clob.polymarket.com/markets",
    "polymarket_clob_book":      "https://clob.polymarket.com/book",
    "polymarket_events":         "https://gamma-api.polymarket.com/events",
    "bybit_funding_rate":        "https://api.bybit.com/v5/market/funding/history",
    "bybit_tickers":             "https://api.bybit.com/v5/market/tickers",
}

# ─────────────────────────────────────────────────────────────────────────────
# HTTP / RETRY SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

HTTP: Dict[str, Any] = {
    "timeout":           10,
    "max_retries":       3,
    "retry_delay":       1.5,
    "rate_limit_delay":  0.35,
    "user_agent":        "ArbitrageEngine/1.0 (paper-trading research)",
}

# ─────────────────────────────────────────────────────────────────────────────
# ENGINE DISPLAY NAMES
# ─────────────────────────────────────────────────────────────────────────────

ENGINE_NAMES: Dict[str, str] = {
    "engine1_funding_rate":   "Funding Rate Harvester",
    "engine2_polymarket":     "Polymarket Reward Farmer",
    "engine3_flash_loan":     "Flash Loan Arbitrage",
    "engine4_triangular":     "Triangular Arbitrage",
    "engine5_cross_exchange": "Cross-Exchange Spread",
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

import os as _os
_os.makedirs(DATA_DIR, exist_ok=True)

LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s [%(levelname)-8s] %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "brief": {
            "format": "[%(levelname)s] %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "brief",
            "level": "WARNING",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": LOG_FILE,
            "formatter": "detailed",
            "level": "DEBUG",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "DEBUG",
    },
}


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for a given module name."""
    import logging.config
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger(name)
