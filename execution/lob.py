
# Two-sided Limit Order Book (LOB) with partial-fill tracking, liquidity replenishment, temporary price impact, and full snapshot history.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import numpy as np

Side = Literal["buy", "sell"]


@dataclass
class Fill:
    # A single fill against one price level
    price: float
    qty: float
    level_index: int  # 0 = best price, i.e. queue position / depth consumed


@dataclass
class ExecutionResult:
    side: Side
    requested_qty: float
    filled_qty: float
    total_cost: float
    fills: List[Fill] = field(default_factory=list)

    @property
    def avg_price(self) -> Optional[float]:
        return self.total_cost / self.filled_qty if self.filled_qty > 0 else None

    @property
    def fully_filled(self) -> bool:
        return self.filled_qty >= self.requested_qty


class LimitOrderBook:
    def __init__(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]):
    
        self.bids: List[Tuple[float, float]] = sorted(bids, key=lambda x: -x[0])
        self.asks: List[Tuple[float, float]] = sorted(asks, key=lambda x: x[0])
        self.trade_blotter: List[dict] = []
        self._snapshots: List[dict] = []

    
    # Basic book state
    
    def best_bid(self) -> Optional[float]:
        for price, qty in self.bids:
            if qty > 0:
                return price
        return None

    def best_ask(self) -> Optional[float]:
        for price, qty in self.asks:
            if qty > 0:
                return price
        return None

    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2

    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return ba - bb

    def depth(self, side: Side, n_levels: int = 5) -> List[Tuple[float, float]]:
        book = self.asks if side == "buy" else self.bids
        return book[:n_levels]

    
    # Execution
    
    def execute_market_order(self, side: Side, qty: float, t: Optional[float] = None) -> ExecutionResult:
        
        book = self.asks if side == "buy" else self.bids
        remaining = qty
        fills: List[Fill] = []
        total_cost = 0.0

        for i, (price, available) in enumerate(book):
            if remaining <= 0:
                break
            if available <= 0:
                continue

            traded = min(available, remaining)
            fills.append(Fill(price=price, qty=traded, level_index=i))
            total_cost += price * traded
            book[i] = (price, available - traded)
            remaining -= traded

        filled_qty = qty - remaining
        result = ExecutionResult(side=side, requested_qty=qty, filled_qty=filled_qty, total_cost=total_cost, fills=fills)

        if t is not None:
            for f in fills:
                self.trade_blotter.append({"time": t, "side": side, "price": f.price, "qty": f.qty, "level": f.level_index})

        return result

    def execute_market_buy(self, qty: float) -> float:
        
        result = self.execute_market_order("buy", qty)
        if not result.fully_filled:
            raise ValueError("Not enough liquidity in the order book to fulfill the order.")
        return result.total_cost

    
    # Liquidity dynamics
    
    def replenish(self, side: Optional[Side] = None, qty_range: Tuple[int, int] = (200, 800), rng=None) -> None:
        # Add stochastic liquidity back to one or both sides of the book
        rng = rng or np.random.default_rng()
        books = []
        if side in (None, "buy"):
            books.append(self.asks)
        if side in (None, "sell"):
            books.append(self.bids)

        for book in books:
            for i, (price, qty) in enumerate(book):
                book[i] = (price, qty + rng.integers(qty_range[0], qty_range[1]))

    def apply_price_impact(self, delta: float, side: Literal["buy", "sell", "both"] = "both") -> None:
        """Shift price levels by `delta` (temporary market impact)."""
        if side in ("buy", "both"):
            self.asks = [(p + delta, q) for p, q in self.asks]
        if side in ("sell", "both"):
            self.bids = [(p + delta, q) for p, q in self.bids]

    
    # History
    
    def snapshot(self, t: Optional[float] = None) -> dict:
        # Capture current book state. Call this at each simulation step to build up a history for a depth-chart / heatmap visualization
        snap = {
            "time": t,
            "bids": list(self.bids),
            "asks": list(self.asks),
            "mid_price": self.mid_price(),
            "spread": self.spread(),
        }
        self._snapshots.append(snap)
        return snap

    @property
    def snapshots(self) -> List[dict]:
        return self._snapshots
