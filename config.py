

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

ProcessType = Literal["poisson", "hawkes"]


@dataclass
class SimulationConfig:
    # --- order flow process ---
    process_type: ProcessType = "poisson"
    T: float = 3600.0          # simulation horizon, seconds
    mu: float = 1.0            # baseline arrival rate (events/sec)
    alpha: float = 0.5         # Hawkes excitation strength (ignored if poisson)
    beta: float = 1.0          # Hawkes excitation decay (ignored if poisson)

    #  order book 
    mid_price: float = 100.0
    n_levels: int = 10
    level_spacing: float = 0.01
    level_qty: float = 10_000.0
    replenish_rate: float = 0.5   # liquidity-refresh events/sec
    replenish_qty_range: tuple[int, int] = (200, 800)

    # --- execution ---
    order_qty: float = 50_000.0     # total parent order size
    participation_rate: float = 0.1  # fraction of remaining qty per child trade
    impact_factor: float = 1e-6      # price impact per unit filled
    mean_reversion_speed: float = 0.05  # decay rate of temporary impact

    seed: Optional[int] = None

    def matched_hawkes(self) -> "SimulationConfig":
        
        from core.processes import mu_from_target_rate

        matched_mu = mu_from_target_rate(self.mu, self.alpha, self.beta)
        new_cfg = SimulationConfig(**{**self.__dict__, "process_type": "hawkes", "mu": matched_mu})
        return new_cfg
