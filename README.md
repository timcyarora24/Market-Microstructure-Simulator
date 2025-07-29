# Market Microstructure Simulator

A comprehensive Python implementation of market microstructure models featuring Hawkes processes, limit order book dynamics, and market impact models. This simulator provides realistic modeling of intraday trading behavior and order flow dynamics.

## 🚀 Features

### Core Components

1. **Hawkes Processes** (`hawkes_process.py`)
   - Self-exciting point processes for modeling order arrivals
   - Multivariate Hawkes processes for buy/sell cross-excitation
   - Parameter estimation using maximum likelihood
   - Stability analysis and branching ratio calculations

2. **Limit Order Book** (`limit_order_book.py`)
   - Complete order book implementation with price-time priority
   - Order types: limit, market, stop orders
   - Real-time order matching engine
   - Order cancellation and modification
   - Market depth visualization

3. **Market Impact Models** (`market_impact_models.py`)
   - Linear, square-root, and power-law impact models
   - Almgren-Chriss optimal execution framework
   - Temporary and permanent impact decomposition
   - Impact decay models and cross-asset effects

4. **Integrated Simulator** (`integrated_market_simulator.py`)
   - Combines all components into a unified simulation
   - Realistic market maker behavior
   - Configurable market regimes and parameters
   - Comprehensive output and analysis

5. **Demonstration & Analysis** (`example_usage_demo.py`)
   - Complete examples of all functionality
   - Advanced market microstructure analysis
   - Visualization and statistical tools

## 📊 Key Market Microstructure Features

- **Self-exciting order arrivals** using Hawkes processes
- **Realistic order book dynamics** with proper price-time priority
- **Market impact modeling** for large order execution
- **Cross-excitation** between buy and sell orders
- **Market maker behavior** and liquidity provision
- **Order flow imbalance** analysis
- **Price efficiency** measurements
- **Intraday patterns** and volatility clustering

## 🛠 Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

- `numpy>=1.21.0` - Numerical computations
- `pandas>=1.3.0` - Data manipulation and analysis
- `matplotlib>=3.5.0` - Plotting and visualization
- `scipy>=1.7.0` - Statistical functions and optimization
- `dataclasses>=0.6` - Data structures
- `typing-extensions>=4.0.0` - Type hints

### Quick Setup

```bash
# Clone or download the project files
git clone <repository-url>
cd market-microstructure-simulator

# Install dependencies
pip install -r requirements.txt

# Run the comprehensive demonstration
python example_usage_demo.py
```

## 📖 Usage Examples

### 1. Basic Hawkes Process

```python
from hawkes_process import HawkesProcess, plot_hawkes_simulation

# Create Hawkes process
hawkes = HawkesProcess(mu=1.0, alpha=0.8, beta=2.0)

# Simulate events
events, time_grid, intensity = hawkes.simulate_with_intensities(T=10, dt=0.05)

# Visualize results
plot_hawkes_simulation(events, time_grid, intensity)
```

### 2. Limit Order Book Operations

```python
from limit_order_book import LimitOrderBook, Order, OrderSide, OrderType

# Create order book
lob = LimitOrderBook(tick_size=0.01)

# Add orders
buy_order = Order("order1", OrderSide.BUY, OrderType.LIMIT, 1000, 99.50)
sell_order = Order("order2", OrderSide.SELL, OrderType.LIMIT, 800, 100.50)

trades = lob.add_order(buy_order)
trades = lob.add_order(sell_order)

# Get market snapshot
snapshot = lob.get_order_book_snapshot()
print(f"Best bid: {snapshot['best_bid']}, Best ask: {snapshot['best_ask']}")
```

### 3. Market Impact Analysis

```python
from market_impact_models import SquareRootImpactModel, ImpactParameters

# Create impact model
params = ImpactParameters(sqrt_coeff=0.5, avg_daily_volume=1000000)
model = SquareRootImpactModel(params)

# Calculate impact
impact = model.calculate_impact(volume=50000, price=100.0, side='BUY')
print(f"Total impact: {impact['total_impact']*100:.3f}%")
```

### 4. Integrated Market Simulation

```python
from integrated_market_simulator import MarketSimulator, MarketConfig

# Configure market
config = MarketConfig(
    mu_buy=2.0, mu_sell=2.0,
    alpha_self=0.8, alpha_cross=0.3,
    beta=2.0, volatility=0.02
)

# Run simulation
simulator = MarketSimulator(config)
results = simulator.simulate(T=20.0, market_order_prob=0.15)

# Analyze results
print(f"Total trades: {results['total_trades']}")
print(f"Final mid price: {results['final_mid_price']:.2f}")
```

## 🧮 Mathematical Models

### Hawkes Process Intensity
The intensity function for Hawkes processes is:

```
λ(t) = μ + Σ α·exp(-β·(t - tᵢ)) for all tᵢ < t
```

Where:
- `μ`: baseline intensity
- `α`: self-excitation parameter  
- `β`: decay parameter
- `tᵢ`: previous event times

### Market Impact Models

**Square-root Law:**
```
Impact = γ · Volume^0.5 · σ · sign(side)
```

**Almgren-Chriss Optimal Execution:**
```
x(t) = X · sinh(κ(T-t)) / sinh(κT)
```

Where `κ = √(γσ²/η)` optimizes the trade-off between market impact and timing risk.

## 📈 Analysis Capabilities

The simulator provides extensive analysis tools:

### Market Microstructure Metrics
- **Realized volatility** from high-frequency price changes
- **Bid-ask spread** statistics and dynamics
- **Order flow imbalance** and directional effects
- **Price impact** analysis across different order sizes
- **Market efficiency** measures

### Hawkes Process Analysis
- **Branching ratio** and stability conditions
- **Self and cross-excitation** parameter estimation
- **Intensity clustering** and burst behavior
- **Long-range dependence** in order arrivals

### Order Book Dynamics
- **Market depth** and liquidity analysis
- **Price level evolution** and order queue dynamics
- **Trade-through** and aggressive order patterns
- **Market maker behavior** and spread maintenance

## 🎯 Applications

### Academic Research
- Market microstructure modeling and analysis
- High-frequency trading strategy backtesting
- Liquidity and price discovery studies
- Optimal execution algorithm development

### Industry Applications
- Risk management and market impact assessment
- Trading algorithm development and testing
- Market making strategy optimization
- Regulatory impact analysis

### Educational Use
- Understanding market microstructure principles
- Learning about point processes and order flow
- Visualizing market dynamics and trading behavior
- Exploring the relationship between order flow and prices

## 📊 Visualization Features

The simulator includes comprehensive visualization tools:

- **Order book heatmaps** showing bid/ask depth
- **Price evolution** with trade overlays
- **Hawkes intensity** time series
- **Impact vs. volume** relationships
- **Order flow** and cumulative volume
- **Market efficiency** and price discovery metrics
- **Multi-scenario comparisons**

## 🔧 Customization

### Market Configuration
Easily customize market behavior through `MarketConfig`:

```python
config = MarketConfig(
    # Hawkes process parameters
    mu_buy=2.0, mu_sell=1.8,           # Baseline intensities
    alpha_self=0.8, alpha_cross=0.3,   # Self/cross excitation
    beta=2.5,                          # Decay rate
    
    # Order book parameters  
    tick_size=0.01,                    # Minimum price increment
    initial_mid_price=100.0,           # Starting price
    
    # Order characteristics
    min_order_size=100,                # Minimum order size
    max_order_size=10000,              # Maximum order size
    
    # Market dynamics
    volatility=0.02,                   # Daily volatility
    impact_decay_rate=0.1              # Impact decay
)
```

### Extending the Simulator
The modular design allows easy extension:

- **New order types** (e.g., iceberg, hidden orders)
- **Additional impact models** (e.g., regime-dependent)
- **Market maker strategies** (e.g., optimal bid-ask placement)
- **Multi-asset simulation** with correlation effects
- **Regime-switching models** for different market conditions

## 📚 References

The implementation is based on established academic literature:

1. **Hawkes Processes:**
   - Hawkes, A.G. (1971). "Spectra of some self-exciting and mutually exciting point processes"
   - Bacry, E. et al. (2015). "Hawkes processes in finance"

2. **Market Impact:**
   - Almgren, R. & Chriss, N. (2001). "Optimal execution of portfolio transactions"
   - Bouchaud, J.P. et al. (2009). "How markets slowly digest changes in supply and demand"

3. **Order Book Dynamics:**
   - Gould, M.D. et al. (2013). "Limit order books"
   - Cont, R. et al. (2010). "The price impact of order book events"

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional order types and market features
- More sophisticated market maker models  
- Machine learning integration for parameter estimation
- Real market data calibration tools
- Performance optimization for large-scale simulations

## 📄 License

This project is designed for educational and research purposes. Please cite appropriately if used in academic work.

## 📞 Support

For questions or issues:

1. Check the comprehensive examples in `example_usage_demo.py`
2. Review the documentation in each module
3. Run the demonstration script to ensure proper setup
4. Refer to the academic references for theoretical background

---

**Note:** This is a research and educational tool. For production trading applications, additional validation, testing, and risk management features would be required.
