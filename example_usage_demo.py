#!/usr/bin/env python3
"""
Comprehensive demonstration of the market microstructure simulator
showing Hawkes processes, limit order book dynamics, and market impact models.

This script demonstrates:
1. Basic Hawkes process simulation
2. Limit order book operations
3. Market impact model comparisons
4. Integrated market simulation
5. Advanced analysis and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our market microstructure modules
from hawkes_process import HawkesProcess, MultivariatehawkesProcess, plot_hawkes_simulation
from limit_order_book import (
    LimitOrderBook, Order, OrderSide, OrderType, OrderStatus, 
    Trade, plot_order_book
)
from market_impact_models import (
    LinearImpactModel, SquareRootImpactModel, PowerLawImpactModel,
    AlmgrenChrisImpactModel, ImpactParameters, compare_impact_models,
    plot_impact_comparison, plot_impact_decay
)
from integrated_market_simulator import MarketSimulator, MarketConfig, plot_simulation_results

def demo_hawkes_process():
    """Demonstrate Hawkes process functionality"""
    print("\n" + "="*60)
    print("1. HAWKES PROCESS DEMONSTRATION")
    print("="*60)
    
    # Single Hawkes process
    print("\n--- Single Hawkes Process ---")
    hawkes = HawkesProcess(mu=1.0, alpha=0.8, beta=2.0)
    
    print(f"Parameters: μ={hawkes.mu}, α={hawkes.alpha}, β={hawkes.beta}")
    print(f"Branching ratio: {hawkes.branching_ratio():.3f}")
    print(f"Stable: {hawkes.stability_condition()}")
    
    # Simulate and plot
    T = 10
    events, time_grid, intensity = hawkes.simulate_with_intensities(T, dt=0.05, random_seed=42)
    print(f"Generated {len(events)} events in {T} time units")
    print(f"Average rate: {len(events)/T:.2f} events per unit time")
    
    # Plot results
    plot_hawkes_simulation(events, time_grid, intensity, "Single Hawkes Process")
    
    # Multivariate Hawkes process
    print("\n--- Multivariate Hawkes Process (Buy/Sell) ---")
    mu = np.array([1.5, 1.2])  # Buy and sell baseline intensities
    alpha = np.array([[0.8, 0.3], [0.3, 0.7]])  # Self and cross excitation
    beta = np.array([[2.0, 2.0], [2.0, 2.0]])   # Decay rates
    
    mv_hawkes = MultivariatehawkesProcess(mu, alpha, beta)
    mv_events = mv_hawkes.simulate(T=15, random_seed=123)
    
    print(f"Buy events: {len(mv_events[0])}")
    print(f"Sell events: {len(mv_events[1])}")
    
    # Plot multivariate events
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.eventplot([mv_events[0]], colors=['green'], linewidths=2, label='Buy Orders')
    ax1.set_ylabel('Buy Events')
    ax1.set_title('Multivariate Hawkes Process - Order Arrivals')
    ax1.grid(True, alpha=0.3)
    
    ax2.eventplot([mv_events[1]], colors=['red'], linewidths=2, label='Sell Orders')
    ax2.set_ylabel('Sell Events')
    ax2.set_xlabel('Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demo_limit_order_book():
    """Demonstrate limit order book functionality"""
    print("\n" + "="*60)
    print("2. LIMIT ORDER BOOK DEMONSTRATION")
    print("="*60)
    
    # Create order book
    lob = LimitOrderBook(tick_size=0.01)
    
    # Add initial orders to create market structure
    print("\n--- Building Initial Order Book ---")
    initial_orders = [
        Order("bid1", OrderSide.BUY, OrderType.LIMIT, 1000, 99.50, trader_id="mm1"),
        Order("bid2", OrderSide.BUY, OrderType.LIMIT, 800, 99.45, trader_id="mm2"),
        Order("bid3", OrderSide.BUY, OrderType.LIMIT, 1200, 99.40, trader_id="trader1"),
        Order("ask1", OrderSide.SELL, OrderType.LIMIT, 900, 100.50, trader_id="mm3"),
        Order("ask2", OrderSide.SELL, OrderType.LIMIT, 1100, 100.55, trader_id="mm4"),
        Order("ask3", OrderSide.SELL, OrderType.LIMIT, 700, 100.60, trader_id="trader2"),
    ]
    
    for order in initial_orders:
        trades = lob.add_order(order)
        print(f"Added {order.side.value} order: {order.quantity} @ {order.price}")
    
    # Show initial order book state
    snapshot = lob.get_order_book_snapshot()
    print(f"\nInitial Market State:")
    print(f"  Best Bid: {snapshot['best_bid']}")
    print(f"  Best Ask: {snapshot['best_ask']}")
    print(f"  Spread: {snapshot['spread']:.4f}")
    print(f"  Mid Price: {snapshot['mid_price']:.2f}")
    
    # Visualize initial order book
    plot_order_book(lob, max_levels=10)
    
    # Demonstrate order execution
    print("\n--- Order Execution Examples ---")
    
    # Market order execution
    print("\n1. Market Buy Order (500 shares)")
    market_buy = Order("market1", OrderSide.BUY, OrderType.MARKET, 500, trader_id="aggressive1")
    trades = lob.add_order(market_buy)
    
    print(f"Executed {len(trades)} trades:")
    for trade in trades:
        print(f"  {trade.quantity} shares @ {trade.price}")
    
    # Large limit order
    print("\n2. Large Limit Sell Order (1500 shares @ 100.00)")
    large_sell = Order("large1", OrderSide.SELL, OrderType.LIMIT, 1500, 100.00, trader_id="whale1")
    trades = lob.add_order(large_sell)
    
    print(f"Large order generated {len(trades)} immediate trades")
    for trade in trades:
        print(f"  {trade.quantity} shares @ {trade.price}")
    
    # Show updated order book
    snapshot = lob.get_order_book_snapshot()
    print(f"\nUpdated Market State:")
    print(f"  Best Bid: {snapshot['best_bid']}")
    print(f"  Best Ask: {snapshot['best_ask']}")
    print(f"  Spread: {snapshot['spread']:.4f}")
    print(f"  Last Price: {snapshot['last_price']}")
    print(f"  Total Volume: {snapshot['total_volume']:.0f}")
    
    # Final order book visualization
    plot_order_book(lob, max_levels=15)
    
    return lob

def demo_market_impact():
    """Demonstrate market impact models"""
    print("\n" + "="*60)
    print("3. MARKET IMPACT MODELS DEMONSTRATION")
    print("="*60)
    
    # Compare different impact models
    print("\n--- Impact Model Comparison ---")
    volume = 50000
    price = 100.0
    side = 'BUY'
    volatility = 0.02
    
    comparison = compare_impact_models(volume, price, side, volatility)
    print(f"\nImpact comparison for {volume} share buy order:")
    print(comparison.to_string(index=False))
    
    # Plot impact vs volume relationship
    print("\n--- Impact vs Volume Analysis ---")
    volumes = np.logspace(2, 6, 50)  # From 100 to 1,000,000 shares
    plot_impact_comparison(volumes, price, side, volatility)
    
    # Demonstrate Almgren-Chriss optimal execution
    print("\n--- Almgren-Chriss Optimal Execution ---")
    params = ImpactParameters()
    ac_model = AlmgrenChrisImpactModel(params, risk_aversion=1e-6)
    
    volume = 100000
    time_horizon = 3600  # 1 hour in seconds
    
    # Calculate optimal execution schedule
    t, x_t = ac_model.optimal_execution_schedule(volume, time_horizon, volatility)
    
    # Plot optimal trajectory
    plt.figure(figsize=(12, 6))
    plt.plot(t * 60, x_t, linewidth=3, label='Optimal Inventory Trajectory', color='blue')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Remaining Inventory')
    plt.title('Almgren-Chriss Optimal Execution Schedule')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add execution rate on secondary y-axis
    ax2 = plt.gca().twinx()
    execution_rate = -np.gradient(x_t, t)  # Negative gradient = execution rate
    ax2.plot(t * 60, execution_rate * 1440, 'r--', linewidth=2, label='Execution Rate', alpha=0.7)  # Convert to daily rate
    ax2.set_ylabel('Execution Rate (shares/day)', color='red')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate execution impact
    impact_result = ac_model.calculate_impact(volume, price, side, volatility, time_horizon)
    print(f"Optimal execution impact: {impact_result['total_impact'] * 100:.3f}%")
    print(f"Optimal execution rate: {impact_result['optimal_execution_rate']:.2f} shares/day")
    
    # Impact decay demonstration
    print("\n--- Impact Decay Models ---")
    plot_impact_decay(initial_impact=0.001, time_horizon=3600)

def demo_integrated_simulation():
    """Demonstrate the integrated market simulator"""
    print("\n" + "="*60)
    print("4. INTEGRATED MARKET SIMULATION")
    print("="*60)
    
    # Create multiple simulation scenarios
    scenarios = [
        {
            'name': 'Balanced Market',
            'config': MarketConfig(
                mu_buy=2.0, mu_sell=2.0,
                alpha_self=0.6, alpha_cross=0.2,
                beta=2.0, volatility=0.01
            )
        },
        {
            'name': 'High Frequency Market',
            'config': MarketConfig(
                mu_buy=5.0, mu_sell=5.0,
                alpha_self=0.9, alpha_cross=0.4,
                beta=3.0, volatility=0.015
            )
        },
        {
            'name': 'Volatile Market',
            'config': MarketConfig(
                mu_buy=1.5, mu_sell=1.8,
                alpha_self=0.8, alpha_cross=0.6,
                beta=1.5, volatility=0.03
            )
        }
    ]
    
    simulation_results = []
    
    for scenario in scenarios:
        print(f"\n--- Simulating {scenario['name']} ---")
        
        # Create and run simulator
        simulator = MarketSimulator(scenario['config'])
        
        # Run simulation
        T = 15.0  # Simulation time
        results = simulator.simulate(T, market_order_prob=0.15)
        
        # Store results
        results['scenario_name'] = scenario['name']
        simulation_results.append(results)
        
        print(f"Completed {scenario['name']} simulation:")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Total Volume: {results['total_volume']:.0f}")
        print(f"  Final Mid Price: {results['final_mid_price']:.2f}")
        print(f"  Final Spread: {results['final_spread']:.4f}")
    
    # Plot comparison of scenarios
    print("\n--- Scenario Comparison ---")
    plot_scenario_comparison(simulation_results)
    
    # Detailed analysis of best scenario
    best_scenario = simulation_results[0]  # Use first scenario for detailed analysis
    print(f"\n--- Detailed Analysis: {best_scenario['scenario_name']} ---")
    plot_simulation_results(best_scenario)
    
    return simulation_results

def plot_scenario_comparison(results_list):
    """Plot comparison of different simulation scenarios"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data for comparison
    scenario_names = [r['scenario_name'] for r in results_list]
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # 1. Price evolution comparison
    ax = axes[0, 0]
    for i, results in enumerate(results_list):
        price_df = results['price_history']
        if not price_df.empty:
            ax.plot(price_df['time'], price_df['mid_price'], 
                   color=colors[i], linewidth=2, label=results['scenario_name'])
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Mid Price')
    ax.set_title('Price Evolution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Hawkes intensity comparison
    ax = axes[0, 1]
    for i, results in enumerate(results_list):
        intensity_df = results['intensity_history']
        if not intensity_df.empty:
            total_intensity = intensity_df['buy_intensity'] + intensity_df['sell_intensity']
            ax.plot(intensity_df['time'], total_intensity, 
                   color=colors[i], linewidth=2, label=results['scenario_name'])
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Intensity')
    ax.set_title('Order Arrival Intensity Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Spread evolution comparison
    ax = axes[1, 0]
    for i, results in enumerate(results_list):
        price_df = results['price_history']
        if not price_df.empty and 'spread' in price_df.columns:
            ax.plot(price_df['time'], price_df['spread'], 
                   color=colors[i], linewidth=2, label=results['scenario_name'])
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Spread')
    ax.set_title('Bid-Ask Spread Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Trading volume comparison
    ax = axes[1, 1]
    volumes = [r['total_volume'] for r in results_list]
    trade_counts = [r['total_trades'] for r in results_list]
    
    ax.bar(scenario_names, volumes, alpha=0.7, color=colors[:len(scenario_names)], 
           label='Total Volume')
    ax.set_ylabel('Total Volume', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    
    # Secondary y-axis for trade count
    ax2 = ax.twinx()
    ax2.plot(scenario_names, trade_counts, 'ro-', linewidth=3, markersize=8, label='Trade Count')
    ax2.set_ylabel('Total Trades', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax.set_title('Volume and Trade Count Comparison')
    ax.set_xlabel('Scenario')
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

def demo_advanced_analysis():
    """Demonstrate advanced analysis capabilities"""
    print("\n" + "="*60)
    print("5. ADVANCED ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Run a detailed simulation for analysis
    config = MarketConfig(
        mu_buy=3.0, mu_sell=2.5,
        alpha_self=0.8, alpha_cross=0.4,
        beta=2.2, volatility=0.02,
        min_order_size=100, max_order_size=5000
    )
    
    simulator = MarketSimulator(config)
    results = simulator.simulate(T=25.0, market_order_prob=0.2)
    
    # Extract data
    price_df = results['price_history']
    trade_df = results['trade_history']
    order_flow_df = results['order_flow_history']
    
    if price_df.empty or trade_df.empty:
        print("Insufficient data for advanced analysis")
        return
    
    print("\n--- Market Microstructure Statistics ---")
    
    # 1. Calculate realized volatility
    if len(price_df) > 1:
        price_returns = np.diff(np.log(price_df['mid_price']))
        realized_vol = np.std(price_returns) * np.sqrt(252 * 24 * 60)  # Annualized
        print(f"Realized Volatility: {realized_vol:.4f}")
    
    # 2. Trade size analysis
    avg_trade_size = trade_df['quantity'].mean()
    median_trade_size = trade_df['quantity'].median()
    print(f"Average Trade Size: {avg_trade_size:.2f}")
    print(f"Median Trade Size: {median_trade_size:.2f}")
    
    # 3. Bid-ask spread statistics
    avg_spread = price_df['spread'].mean()
    spread_vol = price_df['spread'].std()
    print(f"Average Spread: {avg_spread:.4f}")
    print(f"Spread Volatility: {spread_vol:.4f}")
    
    # 4. Order flow imbalance
    buy_volume = order_flow_df[order_flow_df['side'] == 'BUY']['size'].sum()
    sell_volume = order_flow_df[order_flow_df['side'] == 'SELL']['size'].sum()
    imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
    print(f"Order Flow Imbalance: {imbalance:.4f}")
    
    # 5. Price impact analysis
    print("\n--- Price Impact Analysis ---")
    
    # Calculate price impact for different trade sizes
    trade_sizes = trade_df['quantity'].values
    price_changes = []
    
    for i in range(1, len(trade_df)):
        if i < len(price_df):
            price_change = (price_df.iloc[i]['mid_price'] - price_df.iloc[i-1]['mid_price']) / price_df.iloc[i-1]['mid_price']
            price_changes.append(price_change)
        else:
            price_changes.append(0)
    
    # Plot impact vs size
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Trade size vs price impact
    plt.subplot(2, 3, 1)
    if len(price_changes) > 0:
        plt.scatter(trade_sizes[1:len(price_changes)+1], np.abs(price_changes), alpha=0.6)
        plt.xlabel('Trade Size')
        plt.ylabel('Absolute Price Impact')
        plt.title('Trade Size vs Price Impact')
        plt.grid(True, alpha=0.3)
    
    # Subplot 2: Order flow autocorrelation
    plt.subplot(2, 3, 2)
    buy_indicator = (order_flow_df['side'] == 'BUY').astype(int)
    lags = range(1, min(21, len(buy_indicator)//2))
    autocorr = [np.corrcoef(buy_indicator[:-lag], buy_indicator[lag:])[0,1] for lag in lags]
    
    plt.plot(lags, autocorr, 'bo-')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Order Flow Autocorrelation')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Intraday patterns
    plt.subplot(2, 3, 3)
    if len(trade_df) > 0:
        time_bins = np.linspace(0, price_df['time'].max(), 20)
        trade_counts, _ = np.histogram(trade_df['time'], bins=time_bins)
        bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
        
        plt.plot(bin_centers, trade_counts, 'g-', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Trade Count')
        plt.title('Intraday Trading Pattern')
        plt.grid(True, alpha=0.3)
    
    # Subplot 4: Order size distribution (log-log)
    plt.subplot(2, 3, 4)
    sizes = order_flow_df['size'].values
    hist, bins = np.histogram(sizes, bins=50)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Remove zeros for log plot
    nonzero_mask = hist > 0
    plt.loglog(bin_centers[nonzero_mask], hist[nonzero_mask], 'bo-')
    plt.xlabel('Order Size')
    plt.ylabel('Frequency')
    plt.title('Order Size Distribution (Log-Log)')
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Hawkes intensity evolution
    plt.subplot(2, 3, 5)
    intensity_df = results['intensity_history']
    if not intensity_df.empty:
        plt.plot(intensity_df['time'], intensity_df['buy_intensity'], 'g-', label='Buy')
        plt.plot(intensity_df['time'], intensity_df['sell_intensity'], 'r-', label='Sell')
        plt.xlabel('Time')
        plt.ylabel('Intensity')
        plt.title('Hawkes Process Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Subplot 6: Price efficiency (price vs. value)
    plt.subplot(2, 3, 6)
    if len(trade_df) > 10:
        # Simple efficiency measure: how often prices move in same direction as trades
        trade_directions = ['BUY' if side == 'BUY' else 'SELL' for side in trade_df['aggressor_side']]
        
        efficiency_score = 0
        for i in range(1, min(len(price_changes), len(trade_directions)-1)):
            price_dir = 1 if price_changes[i] > 0 else -1
            trade_dir = 1 if trade_directions[i] == 'BUY' else -1
            if price_dir == trade_dir:
                efficiency_score += 1
        
        efficiency_ratio = efficiency_score / max(1, len(price_changes)-1)
        
        plt.bar(['Efficient', 'Inefficient'], [efficiency_ratio, 1-efficiency_ratio], 
                color=['green', 'red'], alpha=0.7)
        plt.ylabel('Ratio')
        plt.title(f'Price Efficiency: {efficiency_ratio:.2f}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nAdvanced analysis completed for {len(trade_df)} trades over {price_df['time'].max():.1f} time units")

def main():
    """Run comprehensive demonstration of market microstructure simulator"""
    print("MARKET MICROSTRUCTURE SIMULATOR - COMPREHENSIVE DEMO")
    print("="*65)
    print("This demonstration showcases:")
    print("• Hawkes processes for self-exciting order arrivals")
    print("• Limit order book dynamics with realistic order matching")
    print("• Market impact models for large order execution")
    print("• Integrated simulation combining all components")
    print("• Advanced market microstructure analysis")
    print("="*65)
    
    try:
        # Run all demonstrations
        demo_hawkes_process()
        order_book = demo_limit_order_book()
        demo_market_impact()
        simulation_results = demo_integrated_simulation()
        demo_advanced_analysis()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Components Demonstrated:")
        print("✓ Hawkes Process Simulation")
        print("✓ Limit Order Book Operations")
        print("✓ Market Impact Models")
        print("✓ Integrated Market Simulation")
        print("✓ Advanced Market Analysis")
        
        print("\nFiles Created:")
        print("• hawkes_process.py - Hawkes process implementation")
        print("• limit_order_book.py - Order book and matching engine")
        print("• market_impact_models.py - Impact and execution models")
        print("• integrated_market_simulator.py - Complete simulator")
        print("• example_usage_demo.py - This demonstration script")
        
        print("\nNext Steps:")
        print("• Modify MarketConfig parameters to explore different market regimes")
        print("• Extend the simulator with additional order types or market features")
        print("• Calibrate models to real market data")
        print("• Implement risk management and portfolio optimization")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("Please check that all required modules are properly installed.")

if __name__ == "__main__":
    main()