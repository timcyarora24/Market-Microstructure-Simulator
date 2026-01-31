

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dashboard.simulation import poisson_process, hawkes_process
from execution.lob import LimitOrderBook

def run_vwap_experiment():
    # Execution parameters
    t_quantity = 50000
    p = 0.01  # participation rate
    time_horizon = 390
    p_ref = 100.0
    
    # Market microstructure parameters
    impact_factor = 0.000001  # price impact per share
    mean_reversion_speed = 0.1  # how fast price reverts
    replenish_rate = 3.0  # liquidity replenishment events per minute
    
    vwap_poisson = []
    vwap_hawkes = []
    
    # === POISSON SIMULATIONS ===
    print("Running Poisson simulations...")
    for run in range(400):
        # Initialize LOB with dynamic mid-price
        mid_price = p_ref
        asks = [(mid_price + 0.01 + i*0.01, 10_000) for i in range(10)]
        lob = LimitOrderBook(asks.copy())
        
        # Generate events
        trade_events = poisson_process(rate=120/60, T=time_horizon)
        replenish_events = poisson_process(rate=replenish_rate, T=time_horizon)
        
        all_events = [(t, 'trade') for t in trade_events] + \
                     [(t, 'replenish') for t in replenish_events]
        all_events.sort()
        
        remaining_qty = t_quantity
        total_cost = 0.0
        last_time = 0
        cumulative_impact = 0.0
        
        for event_time, event_type in all_events:
            # Time decay of price impact (mean reversion)
            dt = event_time - last_time
            cumulative_impact *= np.exp(-mean_reversion_speed * dt)
            last_time = event_time
            
            if event_type == 'replenish':
                # Replenish liquidity
                for i in range(len(lob.asks)):
                    price, qty = lob.asks[i]
                    new_qty = qty + np.random.randint(200, 800)
                    lob.asks[i] = (price, new_qty)
                    
            elif event_type == 'trade':
                if remaining_qty <= 0:
                    continue
                
                trade_size = max(1, int(p * remaining_qty))
                trade_size = min(trade_size, remaining_qty)
                
                try:
                    # Execute trade
                    cost = lob.execute_market_buy(trade_size)
                    total_cost += cost
                    remaining_qty -= trade_size
                    
                    # Add temporary price impact
                    impact = impact_factor * trade_size
                    cumulative_impact += impact
                    
                    # Update LOB prices with impact
                    for i in range(len(lob.asks)):
                        old_price, qty = lob.asks[i]
                        new_price = old_price + cumulative_impact
                        lob.asks[i] = (new_price, qty)
                        
                except ValueError:
                    pass
        
        if remaining_qty < t_quantity:
            executed_qty = t_quantity - remaining_qty
            vwap_price = total_cost / executed_qty
            slippage_bps = 10_000 * (vwap_price - p_ref) / p_ref
            vwap_poisson.append(slippage_bps)
    
    # === HAWKES SIMULATIONS ===
    print("Running Hawkes simulations...")
    for run in range(400):
        # Initialize LOB
        mid_price = p_ref
        asks = [(mid_price + 0.01 + i*0.01, 10_000) for i in range(20)]
        lob = LimitOrderBook(asks.copy())
        
        # Generate events - Hawkes for trades
        rate = 120 / 60
        alpha = 0.8
        beta = 1.5
        mu = rate * (1 - alpha / beta)
        trade_events, _, _ = hawkes_process(mu=mu, alpha=alpha, beta=beta, T=time_horizon)
        replenish_events = poisson_process(rate=replenish_rate, T=time_horizon)
        
        all_events = [(t, 'trade') for t in trade_events] + \
                     [(t, 'replenish') for t in replenish_events]
        all_events.sort()
        
        remaining_qty = t_quantity
        total_cost = 0.0
        last_time = 0
        cumulative_impact = 0.0
        
        for event_time, event_type in all_events:
            # Time decay of price impact
            dt = event_time - last_time
            cumulative_impact *= np.exp(-mean_reversion_speed * dt)
            last_time = event_time
            
            if event_type == 'replenish':
                for i in range(len(lob.asks)):
                    price, qty = lob.asks[i]
                    new_qty = qty + np.random.randint(200, 800)
                    lob.asks[i] = (price, new_qty)
                    
            elif event_type == 'trade':
                if remaining_qty <= 0:
                    continue
                
                trade_size = max(1, int(p * remaining_qty))
                trade_size = min(trade_size, remaining_qty)
                
                try:
                    cost = lob.execute_market_buy(trade_size)
                    total_cost += cost
                    remaining_qty -= trade_size
                    
                    # Temporary price impact
                    impact = impact_factor * trade_size
                    cumulative_impact += impact
                    
                    # Update prices
                    for i in range(len(lob.asks)):
                        old_price, qty = lob.asks[i]
                        new_price = old_price + cumulative_impact
                        lob.asks[i] = (new_price, qty)
                        
                except ValueError:
                    pass
        
        if remaining_qty < t_quantity:
            executed_qty = t_quantity - remaining_qty
            vwap_price = total_cost / executed_qty
            slippage_bps = 10_000 * (vwap_price - p_ref) / p_ref
            vwap_hawkes.append(slippage_bps)
    
    # === VISUALIZATION ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot
    axes[0].boxplot([vwap_poisson, vwap_hawkes], 
                     labels=['Poisson', 'Hawkes'],
                     patch_artist=True,
                     boxprops=dict(facecolor='lightblue'))
    axes[0].set_ylabel('VWAP Slippage (bps)', fontsize=12)
    axes[0].set_title('VWAP Slippage Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram
    axes[1].hist(vwap_poisson, bins=30, alpha=0.6, label='Poisson', color='blue')
    axes[1].hist(vwap_hawkes, bins=30, alpha=0.6, label='Hawkes', color='red')
    axes[1].set_xlabel('VWAP Slippage (bps)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Slippage Distributions', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vwap_comparison_advanced.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # === STATISTICS ===
    print(f'\n{"="*60}')
    print(f'VWAP SLIPPAGE ANALYSIS')
    print(f'{"="*60}')
    print(f'\nSimulations completed: Poisson={len(vwap_poisson)}, Hawkes={len(vwap_hawkes)}')
    
    print(f'\n--- Central Tendency ---')
    print(f'Mean (Poisson):   {np.mean(vwap_poisson):6.2f} bps')
    print(f'Mean (Hawkes):    {np.mean(vwap_hawkes):6.2f} bps')
    print(f'Median (Poisson): {np.median(vwap_poisson):6.2f} bps')
    print(f'Median (Hawkes):  {np.median(vwap_hawkes):6.2f} bps')
    
    print(f'\n--- Dispersion ---')
    print(f'Std Dev (Poisson): {np.std(vwap_poisson):6.2f} bps')
    print(f'Std Dev (Hawkes):  {np.std(vwap_hawkes):6.2f} bps')
    
    print(f'\n--- Confidence Intervals (95%) ---')
    ci_poisson = np.percentile(vwap_poisson, [2.5, 97.5])
    ci_hawkes = np.percentile(vwap_hawkes, [2.5, 97.5])
    print(f'Poisson: [{ci_poisson[0]:6.2f}, {ci_poisson[1]:6.2f}] bps')
    print(f'Hawkes:  [{ci_hawkes[0]:6.2f}, {ci_hawkes[1]:6.2f}] bps')
    
    print(f'\n--- Difference (Hawkes - Poisson) ---')
    diff_mean = np.mean(vwap_hawkes) - np.mean(vwap_poisson)
    diff_median = np.median(vwap_hawkes) - np.median(vwap_poisson)
    print(f'Mean difference:   {diff_mean:6.2f} bps')
    print(f'Median difference: {diff_median:6.2f} bps')
    
    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(vwap_hawkes, vwap_poisson)
    print(f'\nT-test: t={t_stat:.3f}, p={p_value:.4f}')
    
    print(f'\n{"="*60}')
    
    return {
        'vwap_poisson': vwap_poisson,
        'vwap_hawkes': vwap_hawkes
    }


if __name__ == "__main__":
    results = run_vwap_experiment()