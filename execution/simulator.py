
# MarketSimulator: the single object that ties order-flow generation, the
# limit order book, and execution together.



from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from config import SimulationConfig
from core.processes import hawkes_process, poisson_process
from execution.lob import LimitOrderBook


@dataclass
class SimulationResult:
    config: SimulationConfig
    filled_qty: float
    total_cost: float
    n_trades: int
    lob: LimitOrderBook
    trade_events: np.ndarray

    @property
    def avg_price(self) -> Optional[float]:
        return self.total_cost / self.filled_qty if self.filled_qty > 0 else None

    @property
    def slippage_bps(self) -> Optional[float]:
        # Execution slippage vs. arrival mid price, in basis points
        if self.avg_price is None:
            return None
        arrival_mid = self.config.mid_price
        return (self.avg_price - arrival_mid) / arrival_mid * 1e4

    @property
    def fully_filled(self) -> bool:
        return self.filled_qty >= self.config.order_qty


class MarketSimulator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def _build_book(self) -> LimitOrderBook:
        c = self.config
        asks = [(c.mid_price + c.level_spacing * (i + 1), c.level_qty) for i in range(c.n_levels)]
        bids = [(c.mid_price - c.level_spacing * (i + 1), c.level_qty) for i in range(c.n_levels)]
        return LimitOrderBook(bids, asks)

    def _generate_trade_events(self) -> np.ndarray:
        c = self.config
        if c.process_type == "poisson":
            return poisson_process(c.mu, c.T, rng=self.rng)
        elif c.process_type == "hawkes":
            events, _ = hawkes_process(c.mu, c.alpha, c.beta, c.T, rng=self.rng)
            return events
        raise ValueError(f"Unknown process_type: {c.process_type}")

    def run(self, record_history: bool = False) -> SimulationResult:
        c = self.config
        lob = self._build_book()

        trade_events = self._generate_trade_events()
        replenish_events = poisson_process(c.replenish_rate, c.T, rng=self.rng)

        merged = sorted(
            [(t, "trade") for t in trade_events] + [(t, "replenish") for t in replenish_events],
            key=lambda x: x[0],
        )

        remaining_qty = c.order_qty
        total_cost = 0.0
        n_trades = 0
        cumulative_impact = 0.0
        applied_impact = 0.0
        last_t = 0.0

        if record_history:
            lob.snapshot(t=0.0)

        for t, kind in merged:
            dt = t - last_t
            last_t = t

            
            cumulative_impact *= np.exp(-c.mean_reversion_speed * dt)
            delta = cumulative_impact - applied_impact
            if delta != 0:
                lob.apply_price_impact(delta, side="buy")
                applied_impact = cumulative_impact

            if kind == "replenish":
                lob.replenish(side="buy", qty_range=c.replenish_qty_range, rng=self.rng)
            else:
                if remaining_qty <= 0:
                    continue
                trade_size = min(remaining_qty, max(1.0, c.participation_rate * remaining_qty))
                result = lob.execute_market_order("buy", trade_size, t=t)

                if result.filled_qty > 0:
                    total_cost += result.total_cost
                    remaining_qty -= result.filled_qty
                    n_trades += 1

                    cumulative_impact += c.impact_factor * result.filled_qty
                    delta = cumulative_impact - applied_impact
                    lob.apply_price_impact(delta, side="buy")
                    applied_impact = cumulative_impact

            if record_history:
                lob.snapshot(t=t)

        return SimulationResult(
            config=c,
            filled_qty=c.order_qty - remaining_qty,
            total_cost=total_cost,
            n_trades=n_trades,
            lob=lob,
            trade_events=trade_events,
        )
