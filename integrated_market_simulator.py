import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import uuid
from dataclasses import dataclass, field
import warnings

# Import our custom modules
from hawkes_process import HawkesProcess, MultivariatehawkesProcess, plot_hawkes_simulation
from limit_order_book import LimitOrderBook, Order, OrderSide, OrderType, OrderStatus, Trade, plot_order_book
from market_impact_models import (
    MarketImpactModel, LinearImpactModel, SquareRootImpactModel, 
    ImpactParameters, ImpactDecayModel
)

@dataclass
class MarketConfig:
    """Configuration for the integrated market simulator"""
    # Hawkes process parameters
    mu_buy: float = 1.0          # Baseline intensity for buy orders
    mu_sell: float = 1.0         # Baseline intensity for sell orders
    alpha_self: float = 0.8      # Self-excitation coefficient
    alpha_cross: float = 0.3     # Cross-excitation coefficient
    beta: float = 2.0            # Decay rate
    
    # Order book parameters
    tick_size: float = 0.01      # Minimum price increment
    initial_mid_price: float = 100.0  # Starting mid price
    spread_half_width: float = 0.5    # Half spread around mid price
    
    # Order characteristics
    min_order_size: float = 100   # Minimum order size
    max_order_size: float = 10000 # Maximum order size
    order_size_alpha: float = 1.5 # Power law exponent for order sizes
    
    # Market maker parameters
    mm_intensity: float = 5.0     # Market maker order intensity
    mm_spread_factor: float = 1.2 # Market maker spread multiplier
    mm_order_size: float = 500    # Typical market maker order size
    
    # Volatility and impact
    volatility: float = 0.02      # Daily volatility
    impact_decay_rate: float = 0.1 # Impact decay rate

class MarketSimulator:
    """
    Integrated market microstructure simulator combining Hawkes processes,
    limit order book dynamics, and market impact models
    """
    
    def __init__(self, config: MarketConfig):
        self.config = config
        
        # Initialize components
        self._init_hawkes_processes()
        self._init_order_book()
        self._init_market_impact()
        self._init_market_makers()
        
        # Simulation state
        self.current_time = 0.0
        self.price_history = []
        self.trade_history = []
        self.order_flow_history = []
        self.intensity_history = []
        
        # Statistics
        self.total_trades = 0
        self.total_volume = 0.0
        
    def _init_hawkes_processes(self):
        """Initialize multivariate Hawkes process for buy/sell orders"""
        # Baseline intensities
        mu = np.array([self.config.mu_buy, self.config.mu_sell])
        
        # Excitation matrix (self and cross excitation)
        alpha = np.array([
            [self.config.alpha_self, self.config.alpha_cross],    # Buy -> (Buy, Sell)
            [self.config.alpha_cross, self.config.alpha_self]     # Sell -> (Buy, Sell)
        ])
        
        # Decay matrix
        beta = np.array([
            [self.config.beta, self.config.beta],
            [self.config.beta, self.config.beta]
        ])
        
        self.hawkes = MultivariatehawkesProcess(mu, alpha, beta)
        
        # Track event times
        self.buy_events = []
        self.sell_events = []
        
    def _init_order_book(self):
        """Initialize limit order book"""
        self.order_book = LimitOrderBook(tick_size=self.config.tick_size)
        
        # Add initial liquidity
        self._seed_initial_liquidity()
        
    def _init_market_impact(self):
        """Initialize market impact models"""
        impact_params = ImpactParameters(
            linear_coeff=0.01,
            sqrt_coeff=0.1,
            avg_daily_volume=500000,
            decay_rate=self.config.impact_decay_rate
        )
        
        self.impact_model = SquareRootImpactModel(impact_params)
        self.decay_model = ImpactDecayModel(decay_rate=self.config.impact_decay_rate)
        
        # Track current impact
        self.current_impact = 0.0
        self.last_impact_time = 0.0
        
    def _init_market_makers(self):
        """Initialize market maker behavior"""
        self.market_maker_last_update = 0.0
        self.mm_update_frequency = 10.0  # Update every 10 time units
        
    def _seed_initial_liquidity(self):
        """Add initial orders to create a realistic order book"""
        mid_price = self.config.initial_mid_price
        spread = self.config.spread_half_width
        
        # Add bid orders
        for i in range(10):
            price = mid_price - spread - i * self.config.tick_size
            size = np.random.uniform(self.config.min_order_size, 
                                   self.config.mm_order_size * 2)
            
            order = Order(
                order_id=f"init_bid_{i}",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=size,
                price=price,
                trader_id="market_maker"
            )
            self.order_book.add_order(order)
        
        # Add ask orders
        for i in range(10):
            price = mid_price + spread + i * self.config.tick_size
            size = np.random.uniform(self.config.min_order_size, 
                                   self.config.mm_order_size * 2)
            
            order = Order(
                order_id=f"init_ask_{i}",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=size,
                price=price,
                trader_id="market_maker"
            )
            self.order_book.add_order(order)
    
    def _generate_order_size(self) -> float:
        """Generate order size using power law distribution"""
        # Power law distribution for order sizes
        u = np.random.random()
        min_size = self.config.min_order_size
        max_size = self.config.max_order_size
        alpha = self.config.order_size_alpha
        
        size = min_size * ((1 - u) * (min_size / max_size)**(alpha - 1) + u)**(1 / (1 - alpha))
        return min(max(size, min_size), max_size)
    
    def _generate_limit_price(self, side: OrderSide, aggressive_prob: float = 0.3) -> float:
        """Generate limit order price"""
        if self.order_book.mid_price is None:
            return self.config.initial_mid_price
        
        mid_price = self.order_book.mid_price
        spread = self.order_book.spread if self.order_book.spread else self.config.spread_half_width * 2
        
        if np.random.random() < aggressive_prob:
            # Aggressive order (near mid price)
            if side == OrderSide.BUY:
                price_range = spread * 0.5
                price = mid_price - np.random.uniform(0, price_range)
            else:
                price_range = spread * 0.5
                price = mid_price + np.random.uniform(0, price_range)
        else:
            # Passive order (away from mid price)
            if side == OrderSide.BUY:
                price_range = spread * 2
                price = mid_price - spread * 0.5 - np.random.uniform(0, price_range)
            else:
                price_range = spread * 2
                price = mid_price + spread * 0.5 + np.random.uniform(0, price_range)
        
        # Round to tick size
        return round(price / self.config.tick_size) * self.config.tick_size
    
    def _update_market_makers(self):
        """Update market maker orders periodically"""
        if self.current_time - self.market_maker_last_update < self.mm_update_frequency:
            return
        
        self.market_maker_last_update = self.current_time
        
        if self.order_book.mid_price is None:
            return
        
        mid_price = self.order_book.mid_price
        target_spread = self.config.spread_half_width * 2 * self.config.mm_spread_factor
        
        # Add market maker orders if spread is wide
        current_spread = self.order_book.spread if self.order_book.spread else float('inf')
        
        if current_spread > target_spread:
            # Add bid
            bid_price = mid_price - target_spread / 2
            bid_order = Order(
                order_id=f"mm_bid_{self.current_time}",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=self.config.mm_order_size,
                price=bid_price,
                trader_id="market_maker"
            )
            self.order_book.add_order(bid_order)
            
            # Add ask
            ask_price = mid_price + target_spread / 2
            ask_order = Order(
                order_id=f"mm_ask_{self.current_time}",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=self.config.mm_order_size,
                price=ask_price,
                trader_id="market_maker"
            )
            self.order_book.add_order(ask_order)
    
    def _update_price_impact(self):
        """Update current price impact due to decay"""
        if self.current_impact != 0:
            time_elapsed = self.current_time - self.last_impact_time
            self.current_impact = self.decay_model.exponential_decay(
                self.current_impact, time_elapsed
            )
            
            if abs(self.current_impact) < 1e-6:
                self.current_impact = 0.0
    
    def _process_order_arrival(self, side: OrderSide, is_market_order: bool = False):
        """Process arrival of new order"""
        # Generate order characteristics
        size = self._generate_order_size()
        
        if is_market_order:
            order = Order(
                order_id=f"market_{self.current_time}_{np.random.randint(1000)}",
                side=side,
                order_type=OrderType.MARKET,
                quantity=size,
                trader_id=f"trader_{np.random.randint(1000)}"
            )
        else:
            price = self._generate_limit_price(side)
            order = Order(
                order_id=f"limit_{self.current_time}_{np.random.randint(1000)}",
                side=side,
                order_type=OrderType.LIMIT,
                quantity=size,
                price=price,
                trader_id=f"trader_{np.random.randint(1000)}"
            )
        
        # Calculate market impact
        if self.order_book.mid_price is not None:
            impact_result = self.impact_model.calculate_impact(
                size, self.order_book.mid_price, side.value, volatility=self.config.volatility
            )
            
            # Update current impact
            time_elapsed = self.current_time - self.last_impact_time
            self.current_impact = self.decay_model.exponential_decay(
                self.current_impact, time_elapsed
            )
            
            # Add new impact
            self.current_impact += impact_result['temporary_impact']
            self.last_impact_time = self.current_time
        
        # Execute order
        trades = self.order_book.add_order(order)
        
        # Record order flow
        self.order_flow_history.append({
            'time': self.current_time,
            'side': side.value,
            'type': order.order_type.value,
            'size': size,
            'price': order.price,
            'trades': len(trades)
        })
        
        # Process trades
        for trade in trades:
            self.trade_history.append({
                'time': self.current_time,
                'price': trade.price,
                'quantity': trade.quantity,
                'aggressor_side': trade.aggressor_side.value
            })
            
            self.total_trades += 1
            self.total_volume += trade.quantity
        
        # Record price and market state
        if self.order_book.mid_price is not None:
            self.price_history.append({
                'time': self.current_time,
                'mid_price': self.order_book.mid_price,
                'bid': self.order_book.best_bid,
                'ask': self.order_book.best_ask,
                'spread': self.order_book.spread,
                'last_price': self.order_book.last_price,
                'impact': self.current_impact
            })
    
    def simulate(self, T: float, dt: float = 0.1, market_order_prob: float = 0.1) -> Dict[str, Any]:
        """
        Run the integrated market simulation
        
        Args:
            T: Simulation time horizon
            dt: Time step for simulation
            market_order_prob: Probability of market vs limit orders
            
        Returns:
            Dictionary with simulation results
        """
        print(f"Starting market simulation for {T} time units...")
        
        # Generate Hawkes process events
        event_times = self.hawkes.simulate(T, random_seed=42)
        buy_times = event_times[0]
        sell_times = event_times[1]
        
        # Combine and sort all events
        all_events = []
        for t in buy_times:
            all_events.append((t, OrderSide.BUY))
        for t in sell_times:
            all_events.append((t, OrderSide.SELL))
        
        all_events.sort(key=lambda x: x[0])
        
        print(f"Generated {len(buy_times)} buy events and {len(sell_times)} sell events")
        
        # Process events
        for i, (event_time, side) in enumerate(all_events):
            self.current_time = event_time
            
            # Update market makers periodically
            self._update_market_makers()
            
            # Update price impact decay
            self._update_price_impact()
            
            # Determine if market or limit order
            is_market = np.random.random() < market_order_prob
            
            # Process order arrival
            self._process_order_arrival(side, is_market)
            
            # Record intensities
            buy_intensity = self.hawkes.intensity(event_time, [buy_times], 0)
            sell_intensity = self.hawkes.intensity(event_time, [sell_times], 1)
            
            self.intensity_history.append({
                'time': event_time,
                'buy_intensity': buy_intensity,
                'sell_intensity': sell_intensity
            })
            
            # Progress update
            if i % 100 == 0:
                print(f"Processed {i}/{len(all_events)} events...")
        
        print("Simulation completed!")
        
        # Compile results
        results = {
            'config': self.config,
            'order_book': self.order_book,
            'buy_events': buy_times,
            'sell_events': sell_times,
            'price_history': pd.DataFrame(self.price_history),
            'trade_history': pd.DataFrame(self.trade_history),
            'order_flow_history': pd.DataFrame(self.order_flow_history),
            'intensity_history': pd.DataFrame(self.intensity_history),
            'total_trades': self.total_trades,
            'total_volume': self.total_volume,
            'final_mid_price': self.order_book.mid_price,
            'final_spread': self.order_book.spread
        }
        
        return results

def plot_simulation_results(results: Dict[str, Any]):
    """
    Create comprehensive plots of simulation results
    """
    price_df = results['price_history']
    trade_df = results['trade_history']
    intensity_df = results['intensity_history']
    order_flow_df = results['order_flow_history']
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Price evolution
    ax1 = plt.subplot(3, 3, 1)
    if not price_df.empty:
        plt.plot(price_df['time'], price_df['mid_price'], 'b-', linewidth=1.5, label='Mid Price')
        plt.plot(price_df['time'], price_df['bid'], 'g--', alpha=0.7, label='Best Bid')
        plt.plot(price_df['time'], price_df['ask'], 'r--', alpha=0.7, label='Best Ask')
        if not trade_df.empty:
            plt.scatter(trade_df['time'], trade_df['price'], c='orange', s=10, alpha=0.6, label='Trades')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Price Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 2. Spread evolution
    ax2 = plt.subplot(3, 3, 2)
    if not price_df.empty and 'spread' in price_df.columns:
        plt.plot(price_df['time'], price_df['spread'], 'purple', linewidth=1.5)
        plt.xlabel('Time')
        plt.ylabel('Spread')
        plt.title('Bid-Ask Spread')
        plt.grid(True, alpha=0.3)
    
    # 3. Trade volume over time
    ax3 = plt.subplot(3, 3, 3)
    if not trade_df.empty:
        plt.scatter(trade_df['time'], trade_df['quantity'], 
                   c=['red' if side == 'SELL' else 'green' for side in trade_df['aggressor_side']], 
                   alpha=0.6, s=20)
        plt.xlabel('Time')
        plt.ylabel('Trade Size')
        plt.title('Trade Sizes Over Time')
        plt.grid(True, alpha=0.3)
    
    # 4. Hawkes intensities
    ax4 = plt.subplot(3, 3, 4)
    if not intensity_df.empty:
        plt.plot(intensity_df['time'], intensity_df['buy_intensity'], 'g-', linewidth=1.5, label='Buy Intensity')
        plt.plot(intensity_df['time'], intensity_df['sell_intensity'], 'r-', linewidth=1.5, label='Sell Intensity')
        plt.xlabel('Time')
        plt.ylabel('Intensity')
        plt.title('Hawkes Process Intensities')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 5. Order flow (cumulative)
    ax5 = plt.subplot(3, 3, 5)
    if not order_flow_df.empty:
        buy_orders = order_flow_df[order_flow_df['side'] == 'BUY']
        sell_orders = order_flow_df[order_flow_df['side'] == 'SELL']
        
        buy_cumsum = buy_orders['size'].cumsum()
        sell_cumsum = sell_orders['size'].cumsum()
        
        plt.plot(buy_orders['time'], buy_cumsum, 'g-', linewidth=2, label='Cumulative Buy Volume')
        plt.plot(sell_orders['time'], sell_cumsum, 'r-', linewidth=2, label='Cumulative Sell Volume')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Volume')
        plt.title('Cumulative Order Flow')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 6. Price impact
    ax6 = plt.subplot(3, 3, 6)
    if not price_df.empty and 'impact' in price_df.columns:
        plt.plot(price_df['time'], price_df['impact'] * 100, 'orange', linewidth=1.5)
        plt.xlabel('Time')
        plt.ylabel('Impact (%)')
        plt.title('Market Impact Over Time')
        plt.grid(True, alpha=0.3)
    
    # 7. Order size distribution
    ax7 = plt.subplot(3, 3, 7)
    if not order_flow_df.empty:
        plt.hist(order_flow_df['size'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Order Size')
        plt.ylabel('Frequency')
        plt.title('Order Size Distribution')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    # 8. Trade price vs mid price
    ax8 = plt.subplot(3, 3, 8)
    if not trade_df.empty and not price_df.empty:
        # Merge trade and price data
        trade_df_copy = trade_df.copy()
        trade_df_copy['mid_price'] = np.interp(trade_df['time'], price_df['time'], price_df['mid_price'])
        
        plt.scatter(trade_df_copy['mid_price'], trade_df_copy['price'], 
                   c=['red' if side == 'SELL' else 'green' for side in trade_df['aggressor_side']], 
                   alpha=0.6, s=20)
        plt.plot([trade_df_copy['mid_price'].min(), trade_df_copy['mid_price'].max()], 
                [trade_df_copy['mid_price'].min(), trade_df_copy['mid_price'].max()], 
                'k--', alpha=0.5)
        plt.xlabel('Mid Price')
        plt.ylabel('Trade Price')
        plt.title('Trade Price vs Mid Price')
        plt.grid(True, alpha=0.3)
    
    # 9. Order book visualization (final state)
    ax9 = plt.subplot(3, 3, 9)
    order_book = results['order_book']
    snapshot = order_book.get_order_book_snapshot(levels=10)
    
    if snapshot['bids'] and snapshot['asks']:
        bid_prices = [level['price'] for level in snapshot['bids']]
        bid_quantities = [level['quantity'] for level in snapshot['bids']]
        ask_prices = [level['price'] for level in snapshot['asks']]
        ask_quantities = [level['quantity'] for level in snapshot['asks']]
        
        plt.barh(bid_prices, bid_quantities, color='green', alpha=0.7, label='Bids')
        plt.barh(ask_prices, ask_quantities, color='red', alpha=0.7, label='Asks')
        plt.xlabel('Quantity')
        plt.ylabel('Price')
        plt.title('Final Order Book State')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n=== Simulation Summary ===")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Total Volume: {results['total_volume']:.0f}")
    print(f"Final Mid Price: {results['final_mid_price']:.2f}")
    print(f"Final Spread: {results['final_spread']:.4f}")
    
    if not trade_df.empty:
        print(f"Average Trade Size: {trade_df['quantity'].mean():.2f}")
        print(f"Trade Price Volatility: {trade_df['price'].std():.4f}")
    
    if not price_df.empty:
        print(f"Price Range: [{price_df['mid_price'].min():.2f}, {price_df['mid_price'].max():.2f}]")
        print(f"Average Spread: {price_df['spread'].mean():.4f}")

# Example usage and demonstration
if __name__ == "__main__":
    print("=== Integrated Market Microstructure Simulator ===")
    
    # Create configuration
    config = MarketConfig(
        mu_buy=2.0,
        mu_sell=2.0,
        alpha_self=0.8,
        alpha_cross=0.3,
        beta=2.5,
        tick_size=0.01,
        initial_mid_price=100.0,
        spread_half_width=0.05,
        volatility=0.02
    )
    
    # Create and run simulator
    simulator = MarketSimulator(config)
    
    # Run simulation
    T = 20.0  # Simulate for 20 time units
    results = simulator.simulate(T, market_order_prob=0.15)
    
    # Plot results
    plot_simulation_results(results)