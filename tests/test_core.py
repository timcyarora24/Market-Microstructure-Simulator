import numpy as np
import pytest

from core.processes import hawkes_process, poisson_process, mu_from_target_rate
from core.hawkes_mle import fit_hawkes
from execution.lob import LimitOrderBook
from execution.simulator import MarketSimulator
from config import SimulationConfig


def test_hawkes_rejects_unstable_params():
    with pytest.raises(ValueError):
        hawkes_process(mu=1.0, alpha=2.0, beta=1.0, T=10)


def test_hawkes_more_clustered_than_poisson_at_same_rate():
    T = 20000
    rate = 1.0
    alpha, beta = 0.5, 1.0
    matched_mu = mu_from_target_rate(rate, alpha, beta)

    poisson_events = poisson_process(rate, T)
    hawkes_events, _ = hawkes_process(matched_mu, alpha, beta, T)

    def cv(events, bin_width=5):
        n_bins = int(T / bin_width)
        counts, _ = np.histogram(events, bins=n_bins, range=(0, T))
        return counts.std() / counts.mean()

    assert cv(hawkes_events) > cv(poisson_events)


def test_lob_conserves_quantity():
    bids = [(99.0, 100)]
    asks = [(100.0, 50), (101.0, 50)]
    lob = LimitOrderBook(bids, asks)

    result = lob.execute_market_order("buy", 120)
    assert result.filled_qty == 100  # only 100 available across both ask levels
    assert not result.fully_filled
    remaining_ask_qty = sum(q for _, q in lob.asks)
    assert remaining_ask_qty == 0


def test_lob_partial_fill_breakdown_matches_total():
    lob = LimitOrderBook(bids=[(99.0, 1000)], asks=[(100.0, 10), (100.5, 10), (101.0, 10)])
    result = lob.execute_market_order("buy", 25)
    assert result.filled_qty == 25
    assert sum(f.qty for f in result.fills) == 25
    assert result.total_cost == pytest.approx(sum(f.price * f.qty for f in result.fills))


def test_two_sided_book_buy_and_sell_independent():
    lob = LimitOrderBook(bids=[(99.0, 100)], asks=[(101.0, 100)])
    buy_result = lob.execute_market_order("buy", 50)
    sell_result = lob.execute_market_order("sell", 50)
    assert buy_result.filled_qty == 50
    assert sell_result.filled_qty == 50
    # buy should not have touched the bid side and vice versa
    assert lob.bids[0][1] == 50
    assert lob.asks[0][1] == 50


def test_poisson_and_hawkes_book_depth_match_in_simulator():

    # Poisson and Hawkes runs must start from an identical book

    poisson_cfg = SimulationConfig(process_type="poisson", mu=1.0, n_levels=10, seed=1)
    hawkes_cfg = poisson_cfg.matched_hawkes()
    assert poisson_cfg.n_levels == hawkes_cfg.n_levels
    assert poisson_cfg.level_qty == hawkes_cfg.level_qty
    assert poisson_cfg.order_qty == hawkes_cfg.order_qty


def test_hawkes_mle_recovers_known_parameters():
    true_mu, true_alpha, true_beta = 0.4, 0.3, 0.8
    events, _ = hawkes_process(true_mu, true_alpha, true_beta, T=20000)
    fit = fit_hawkes(events, T=20000)

    assert fit["mu"] == pytest.approx(true_mu, rel=0.25)
    assert fit["alpha"] == pytest.approx(true_alpha, rel=0.3)
    assert fit["beta"] == pytest.approx(true_beta, rel=0.3)
    assert fit["converged"]


def test_market_simulator_end_to_end_fills_order():
    cfg = SimulationConfig(process_type="poisson", mu=2.0, order_qty=10_000, T=1800, seed=3)
    result = MarketSimulator(cfg).run()
    assert result.filled_qty > 0
    assert result.n_trades > 0
    assert result.slippage_bps is not None
