"""
paper_trader.py
================
Paper trading simulation layer for the Arbitrage Engine system.

All trades are SIMULATED — no real money moves. Live market prices are used
to assess P&L but orders are logged to a JSON ledger only.

State is persisted to: data/paper_trades.json
"""

from __future__ import annotations

import json
import os
import time
import uuid
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from config import (
    CAPITAL,
    ENGINE_NAMES,
    FEES,
    RISK,
    SLIPPAGE,
    TRADES_FILE,
    DATA_DIR,
    PAPER_TRADING,
    LARGE_ORDER_THRESHOLD,
    get_logger,
)

logger = get_logger(__name__)


@dataclass
class Position:
    """Represents one open simulated position."""
    position_id: str
    engine: str
    symbol: str
    side: str
    amount: float
    entry_price: float
    entry_fee: float
    slippage_cost: float
    leverage: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def notional_usd(self) -> float:
        return self.amount * self.entry_price

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Position":
        return cls(**d)


@dataclass
class Trade:
    """Immutable record of a completed (or opened) simulated trade."""
    trade_id: str
    position_id: str
    engine: str
    symbol: str
    side: str
    action: str
    amount: float
    price: float
    fee: float
    slippage_cost: float
    gross_pnl: float
    net_pnl: float
    timestamp: str
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trade":
        return cls(**d)


@dataclass
class EngineState:
    """Runtime state for a single engine."""
    engine: str
    starting_capital: float
    current_capital: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_fees_paid: float = 0.0
    total_slippage_paid: float = 0.0
    total_trades: int = 0
    open_positions: List[Dict[str, Any]] = field(default_factory=list)
    daily_pnl: Dict[str, float] = field(default_factory=dict)
    weekly_pnl: Dict[str, float] = field(default_factory=dict)
    monthly_pnl: Dict[str, float] = field(default_factory=dict)

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def return_pct(self) -> float:
        if self.starting_capital == 0:
            return 0.0
        return self.total_pnl / self.starting_capital

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineState":
        return cls(**d)


class PaperTrader:
    """
    Central paper trading ledger.
    All simulated orders pass through here.
    """

    def __init__(self, trades_file: str = TRADES_FILE) -> None:
        self.trades_file = trades_file
        os.makedirs(DATA_DIR, exist_ok=True)
        self._states: Dict[str, EngineState] = {}
        self._trade_history: List[Dict[str, Any]] = []
        self._load_or_init()

    def _load_or_init(self) -> None:
        if os.path.exists(self.trades_file):
            try:
                self.load_state()
                return
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("State file corrupt (%s); reinitialising.", exc)
        for engine_id, capital in CAPITAL.items():
            self._states[engine_id] = EngineState(
                engine=engine_id,
                starting_capital=capital,
                current_capital=capital,
            )
        self._trade_history = []
        self.save_state()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _today_str() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _week_str() -> str:
        dt = datetime.now(timezone.utc)
        return f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"

    @staticmethod
    def _month_str() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m")

    def _calc_fee(self, engine: str, amount_usd: float, order_type: str = "taker") -> float:
        exchange = "binance"
        rate = FEES.get(exchange, FEES["default"])[order_type]
        return amount_usd * rate

    def _calc_slippage(self, amount_usd: float, symbol: str = "") -> float:
        import random
        if amount_usd >= LARGE_ORDER_THRESHOLD:
            rate = SLIPPAGE["large_order"]
        else:
            rate = random.uniform(SLIPPAGE["min"], SLIPPAGE["max"])
        return amount_usd * rate

    def _get_state(self, engine: str) -> EngineState:
        if engine not in self._states:
            capital = CAPITAL.get(engine, 0.0)
            self._states[engine] = EngineState(
                engine=engine,
                starting_capital=capital,
                current_capital=capital,
            )
        return self._states[engine]

    def execute_trade(
        self,
        engine: str,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        leverage: float = 1.0,
        order_type: str = "taker",
        metadata: Optional[Dict[str, Any]] = None,
        latency_ms: float = 0.0,
    ) -> Trade:
        state = self._get_state(engine)
        notional = amount * price
        fee = self._calc_fee(engine, notional, order_type)
        slippage = self._calc_slippage(notional)
        total_cost = notional + fee + slippage

        if total_cost > state.current_capital:
            logger.warning(
                "Insufficient capital in %s (need $%.2f, have $%.2f)",
                engine, total_cost, state.current_capital,
            )

        position_id = str(uuid.uuid4())
        trade_id = str(uuid.uuid4())
        ts = self._now_iso()

        position = Position(
            position_id=position_id,
            engine=engine,
            symbol=symbol,
            side=side,
            amount=amount,
            entry_price=price,
            entry_fee=fee,
            slippage_cost=slippage,
            leverage=leverage,
            timestamp=ts,
            metadata=metadata or {},
        )

        trade = Trade(
            trade_id=trade_id,
            position_id=position_id,
            engine=engine,
            symbol=symbol,
            side=side,
            action="open",
            amount=amount,
            price=price,
            fee=fee,
            slippage_cost=slippage,
            gross_pnl=0.0,
            net_pnl=-(fee + slippage),
            timestamp=ts,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

        state.current_capital -= (notional + fee + slippage)
        state.total_fees_paid += fee
        state.total_slippage_paid += slippage
        state.total_trades += 1
        state.open_positions.append(position.to_dict())
        state.realized_pnl += trade.net_pnl
        self._record_pnl_snapshot(state)
        self._trade_history.append(trade.to_dict())
        self.save_state()
        return trade

    def close_position(
        self,
        engine: str,
        position_id: str,
        exit_price: float,
        order_type: str = "taker",
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Trade]:
        state = self._get_state(engine)
        pos_dict = next(
            (p for p in state.open_positions if p["position_id"] == position_id),
            None,
        )
        if pos_dict is None:
            return None

        pos = Position.from_dict(pos_dict)
        notional_exit = pos.amount * exit_price
        fee = self._calc_fee(engine, notional_exit, order_type)
        slippage = self._calc_slippage(notional_exit)

        if pos.side == "long":
            gross_pnl = (exit_price - pos.entry_price) * pos.amount * pos.leverage
        elif pos.side == "short":
            gross_pnl = (pos.entry_price - exit_price) * pos.amount * pos.leverage
        else:
            gross_pnl = pos.metadata.get("accumulated_funding", 0.0)

        net_pnl = gross_pnl - fee - slippage

        trade = Trade(
            trade_id=str(uuid.uuid4()),
            position_id=position_id,
            engine=engine,
            symbol=pos.symbol,
            side=pos.side,
            action="close",
            amount=pos.amount,
            price=exit_price,
            fee=fee,
            slippage_cost=slippage,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            timestamp=self._now_iso(),
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

        state.current_capital += notional_exit - fee - slippage
        state.total_fees_paid += fee
        state.total_slippage_paid += slippage
        state.total_trades += 1
        state.realized_pnl += net_pnl
        state.open_positions = [
            p for p in state.open_positions if p["position_id"] != position_id
        ]
        self._record_pnl_snapshot(state)
        self._trade_history.append(trade.to_dict())
        self.save_state()
        return trade

    def log_periodic_income(
        self,
        engine: str,
        position_id: str,
        amount_usd: float,
        income_type: str = "funding_payment",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Trade:
        state = self._get_state(engine)
        for pos in state.open_positions:
            if pos["position_id"] == position_id:
                pos.setdefault("metadata", {})
                pos["metadata"]["accumulated_funding"] = (
                    pos["metadata"].get("accumulated_funding", 0.0) + amount_usd
                )
                break

        trade = Trade(
            trade_id=str(uuid.uuid4()),
            position_id=position_id,
            engine=engine,
            symbol=metadata.get("symbol", "N/A") if metadata else "N/A",
            side="long_short",
            action=income_type,
            amount=0.0,
            price=0.0,
            fee=0.0,
            slippage_cost=0.0,
            gross_pnl=amount_usd,
            net_pnl=amount_usd,
            timestamp=self._now_iso(),
            latency_ms=0.0,
            metadata=metadata or {},
        )

        state.realized_pnl += amount_usd
        state.current_capital += amount_usd
        self._record_pnl_snapshot(state)
        self._trade_history.append(trade.to_dict())
        self.save_state()
        return trade

    def get_positions(self, engine: str) -> List[Dict[str, Any]]:
        return list(self._get_state(engine).open_positions)

    def get_pnl(self, engine: str) -> Dict[str, float]:
        state = self._get_state(engine)
        return {
            "realized_pnl": round(state.realized_pnl, 4),
            "unrealized_pnl": round(state.unrealized_pnl, 4),
            "total_pnl": round(state.total_pnl, 4),
            "return_pct": round(state.return_pct * 100, 4),
            "total_fees_paid": round(state.total_fees_paid, 4),
            "total_slippage_paid": round(state.total_slippage_paid, 4),
        }

    def get_portfolio_summary(self) -> Dict[str, Any]:
        total_realized = sum(s.realized_pnl for s in self._states.values())
        total_unrealized = sum(s.unrealized_pnl for s in self._states.values())
        total_capital = sum(CAPITAL.values())
        total_current = sum(s.current_capital for s in self._states.values())
        total_fees = sum(s.total_fees_paid for s in self._states.values())
        total_trades = sum(s.total_trades for s in self._states.values())

        engines_summary = {}
        for engine_id, state in self._states.items():
            engines_summary[engine_id] = {
                "name": ENGINE_NAMES.get(engine_id, engine_id),
                "capital_allocated": CAPITAL.get(engine_id, 0.0),
                "current_capital": round(state.current_capital, 2),
                "realized_pnl": round(state.realized_pnl, 2),
                "unrealized_pnl": round(state.unrealized_pnl, 2),
                "total_pnl": round(state.total_pnl, 2),
                "return_pct": round(state.return_pct * 100, 4),
                "open_positions": len(state.open_positions),
                "total_trades": state.total_trades,
            }

        return {
            "paper_trading": PAPER_TRADING,
            "timestamp": self._now_iso(),
            "total_capital_deployed": total_capital,
            "total_current_value": round(total_current, 2),
            "total_realized_pnl": round(total_realized, 4),
            "total_unrealized_pnl": round(total_unrealized, 4),
            "total_net_pnl": round(total_realized + total_unrealized, 4),
            "total_return_pct": round(
                (total_realized + total_unrealized) / total_capital * 100, 4
            ) if total_capital > 0 else 0.0,
            "total_fees_paid": round(total_fees, 4),
            "total_trades": total_trades,
            "engines": engines_summary,
            "trade_history_count": len(self._trade_history),
        }

    def get_trade_history(
        self, engine: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        history = self._trade_history
        if engine:
            history = [t for t in history if t["engine"] == engine]
        return history[-limit:]

    def _record_pnl_snapshot(self, state: EngineState) -> None:
        pnl = state.total_pnl
        state.daily_pnl[self._today_str()] = round(pnl, 4)
        state.weekly_pnl[self._week_str()] = round(pnl, 4)
        state.monthly_pnl[self._month_str()] = round(pnl, 4)

    def update_unrealized_pnl(self, engine: str, current_prices: Dict[str, float]) -> None:
        state = self._get_state(engine)
        total_unrealized = 0.0
        for pos_dict in state.open_positions:
            pos = Position.from_dict(pos_dict)
            cp = current_prices.get(pos.symbol)
            if cp is None:
                continue
            if pos.side == "long":
                upnl = (cp - pos.entry_price) * pos.amount * pos.leverage
            elif pos.side == "short":
                upnl = (pos.entry_price - cp) * pos.amount * pos.leverage
            else:
                upnl = pos.metadata.get("accumulated_funding", 0.0)
            total_unrealized += upnl
        state.unrealized_pnl = round(total_unrealized, 4)

    def save_state(self) -> None:
        payload: Dict[str, Any] = {
            "saved_at": self._now_iso(),
            "paper_trading": PAPER_TRADING,
            "states": {eid: s.to_dict() for eid, s in self._states.items()},
            "trade_history": self._trade_history,
        }
        tmp = self.trades_file + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp, self.trades_file)
        except OSError as exc:
            logger.error("Failed to save state: %s", exc)

    def load_state(self) -> None:
        with open(self.trades_file, "r", encoding="utf-8") as f:
            payload: Dict[str, Any] = json.load(f)
        self._states = {}
        for eid, sdict in payload.get("states", {}).items():
            self._states[eid] = EngineState.from_dict(sdict)
        self._trade_history = payload.get("trade_history", [])

    def reset_engine(self, engine: str) -> None:
        capital = CAPITAL.get(engine, 0.0)
        self._states[engine] = EngineState(
            engine=engine,
            starting_capital=capital,
            current_capital=capital,
        )
        self.save_state()

    def reset_all(self) -> None:
        for engine in CAPITAL:
            self.reset_engine(engine)


if __name__ == "__main__":
    pt = PaperTrader()
    summary = pt.get_portfolio_summary()
    print(json.dumps(summary, indent=2, default=str))
