import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import warnings

class ImpactType(Enum):
    TEMPORARY = "TEMPORARY"
    PERMANENT = "PERMANENT"
    TOTAL = "TOTAL"

@dataclass
class ImpactParameters:
    """Parameters for market impact models"""
    # Linear impact parameters
    linear_coeff: float = 0.1  # Price impact per unit volume
    
    # Square-root impact parameters
    sqrt_coeff: float = 0.5    # Coefficient for sqrt law
    sqrt_exponent: float = 0.5  # Exponent (usually 0.5)
    
    # Power law parameters
    power_coeff: float = 0.3
    power_exponent: float = 0.6
    
    # Volatility scaling
    volatility_scaling: float = 1.0
    
    # Liquidity parameters
    avg_daily_volume: float = 1000000  # Average daily volume
    participation_rate: float = 0.1    # Maximum participation rate
    
    # Decay parameters
    decay_rate: float = 0.1  # Rate of temporary impact decay
    half_life: float = 300   # Half-life of impact decay (seconds)

class MarketImpactModel(ABC):
    """Abstract base class for market impact models"""
    
    def __init__(self, parameters: ImpactParameters):
        self.params = parameters
    
    @abstractmethod
    def calculate_impact(self, volume: float, price: float, 
                        side: str, **kwargs) -> Dict[str, float]:
        """
        Calculate market impact for a given order
        
        Args:
            volume: Order volume
            price: Current price
            side: Order side ('BUY' or 'SELL')
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with impact components
        """
        pass
    
    def _get_direction_multiplier(self, side: str) -> int:
        """Get direction multiplier (+1 for buy, -1 for sell)"""
        return 1 if side.upper() == 'BUY' else -1

class LinearImpactModel(MarketImpactModel):
    """
    Linear market impact model: Impact ∝ Volume
    
    I = γ * V * σ * sign(side)
    
    Where:
    - γ: impact coefficient
    - V: volume
    - σ: volatility
    - sign: +1 for buy, -1 for sell
    """
    
    def calculate_impact(self, volume: float, price: float, 
                        side: str, volatility: float = 0.02) -> Dict[str, float]:
        """Calculate linear impact"""
        direction = self._get_direction_multiplier(side)
        
        # Temporary impact (immediate)
        temp_impact = (self.params.linear_coeff * volume * 
                      volatility * self.params.volatility_scaling * direction)
        
        # Permanent impact (fraction of temporary)
        perm_impact = temp_impact * 0.3  # Typically 30% of temporary
        
        # Total impact
        total_impact = temp_impact + perm_impact
        
        # Convert to price terms
        impact_price = price * (1 + total_impact)
        
        return {
            'temporary_impact': temp_impact,
            'permanent_impact': perm_impact,
            'total_impact': total_impact,
            'impact_price': impact_price,
            'original_price': price
        }

class SquareRootImpactModel(MarketImpactModel):
    """
    Square-root market impact model: Impact ∝ Volume^0.5
    
    This is based on empirical findings that impact scales with square root of volume
    """
    
    def calculate_impact(self, volume: float, price: float, 
                        side: str, volatility: float = 0.02,
                        time_horizon: float = 300) -> Dict[str, float]:
        """Calculate square-root impact"""
        direction = self._get_direction_multiplier(side)
        
        # Participation rate
        participation = volume / (self.params.avg_daily_volume / 24)  # Hourly volume
        participation = min(participation, self.params.participation_rate)
        
        # Square-root impact
        base_impact = (self.params.sqrt_coeff * 
                      np.power(volume, self.params.sqrt_exponent) *
                      volatility * self.params.volatility_scaling)
        
        # Adjust for participation rate
        if participation > 0.05:  # High participation penalty
            base_impact *= (1 + 2 * (participation - 0.05))
        
        # Temporary and permanent components
        temp_impact = base_impact * direction
        perm_impact = temp_impact * 0.25  # 25% permanent
        
        total_impact = temp_impact + perm_impact
        impact_price = price * (1 + total_impact)
        
        return {
            'temporary_impact': temp_impact,
            'permanent_impact': perm_impact,
            'total_impact': total_impact,
            'impact_price': impact_price,
            'original_price': price,
            'participation_rate': participation
        }

class PowerLawImpactModel(MarketImpactModel):
    """
    Power law impact model: Impact ∝ Volume^β
    
    More flexible than square-root, allows different exponents
    """
    
    def calculate_impact(self, volume: float, price: float, 
                        side: str, volatility: float = 0.02) -> Dict[str, float]:
        """Calculate power law impact"""
        direction = self._get_direction_multiplier(side)
        
        # Power law impact
        base_impact = (self.params.power_coeff * 
                      np.power(volume, self.params.power_exponent) *
                      volatility * self.params.volatility_scaling)
        
        temp_impact = base_impact * direction
        perm_impact = temp_impact * 0.2  # 20% permanent
        
        total_impact = temp_impact + perm_impact
        impact_price = price * (1 + total_impact)
        
        return {
            'temporary_impact': temp_impact,
            'permanent_impact': perm_impact,
            'total_impact': total_impact,
            'impact_price': impact_price,
            'original_price': price
        }

class AlmgrenChrisImpactModel(MarketImpactModel):
    """
    Almgren-Chriss impact model with optimal execution framework
    
    Includes both permanent and temporary impact with risk aversion
    """
    
    def __init__(self, parameters: ImpactParameters, risk_aversion: float = 1e-6):
        super().__init__(parameters)
        self.risk_aversion = risk_aversion
    
    def calculate_impact(self, volume: float, price: float, side: str,
                        volatility: float = 0.02, time_horizon: float = 300,
                        num_slices: int = 10) -> Dict[str, float]:
        """Calculate Almgren-Chriss impact with optimal slicing"""
        direction = self._get_direction_multiplier(side)
        
        # Optimal execution parameters
        gamma = self.risk_aversion
        sigma = volatility
        T = time_horizon / 86400  # Convert to days
        
        # Impact parameters
        eta = self.params.linear_coeff  # Temporary impact
        gamma_perm = eta * 0.3  # Permanent impact
        
        # Optimal trajectory parameter
        kappa = np.sqrt(gamma * sigma**2 / eta)
        
        # Calculate optimal execution rate
        if kappa * T > 1e-6:
            sinh_term = np.sinh(kappa * T)
            cosh_term = np.cosh(kappa * T)
            execution_rate = kappa * volume / (2 * sinh_term)
        else:
            execution_rate = volume / T  # Constant rate for small kappa*T
        
        # Calculate impacts
        temp_impact_rate = eta * execution_rate
        perm_impact_total = gamma_perm * volume
        
        # Scale by direction
        temp_impact = temp_impact_rate * direction
        perm_impact = perm_impact_total * direction
        total_impact = temp_impact + perm_impact
        
        impact_price = price * (1 + total_impact)
        
        return {
            'temporary_impact': temp_impact,
            'permanent_impact': perm_impact,
            'total_impact': total_impact,
            'impact_price': impact_price,
            'original_price': price,
            'optimal_execution_rate': execution_rate,
            'kappa': kappa,
            'num_slices': num_slices
        }
    
    def optimal_execution_schedule(self, volume: float, time_horizon: float,
                                 volatility: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """Generate optimal execution schedule"""
        gamma = self.risk_aversion
        sigma = volatility
        T = time_horizon / 86400
        eta = self.params.linear_coeff
        
        kappa = np.sqrt(gamma * sigma**2 / eta)
        
        # Time grid
        num_points = 100
        t = np.linspace(0, T, num_points)
        
        # Optimal inventory trajectory
        if kappa * T > 1e-6:
            sinh_kappa_T = np.sinh(kappa * T)
            x_t = volume * np.sinh(kappa * (T - t)) / sinh_kappa_T
        else:
            x_t = volume * (1 - t / T)
        
        return t, x_t

class LiquidityImpactModel(MarketImpactModel):
    """
    Market impact model based on order book liquidity
    
    Impact depends on available liquidity at different price levels
    """
    
    def __init__(self, parameters: ImpactParameters):
        super().__init__(parameters)
        self.liquidity_profile = self._generate_liquidity_profile()
    
    def _generate_liquidity_profile(self) -> Dict[float, float]:
        """Generate synthetic liquidity profile"""
        # Simple exponential decay from mid price
        price_levels = np.arange(-2.0, 2.01, 0.01)  # Price levels from mid
        liquidity = 1000 * np.exp(-np.abs(price_levels) * 2)  # Exponential decay
        
        return dict(zip(price_levels, liquidity))
    
    def calculate_impact(self, volume: float, price: float, side: str,
                        order_book_depth: Optional[Dict[float, float]] = None) -> Dict[str, float]:
        """Calculate liquidity-based impact"""
        direction = self._get_direction_multiplier(side)
        
        if order_book_depth is None:
            order_book_depth = self.liquidity_profile
        
        # Walk through order book to fill the order
        remaining_volume = volume
        total_cost = 0
        price_levels = sorted(order_book_depth.keys(), 
                            reverse=(side.upper() == 'BUY'))
        
        executed_price = price
        for level_offset in price_levels:
            if remaining_volume <= 0:
                break
            
            level_price = price + level_offset * direction
            available_liquidity = order_book_depth[level_offset]
            
            # Execute what we can at this level
            executed_volume = min(remaining_volume, available_liquidity)
            total_cost += executed_volume * level_price
            remaining_volume -= executed_volume
            
            if executed_volume > 0:
                executed_price = level_price
        
        # Calculate impact
        if volume > remaining_volume:
            avg_execution_price = total_cost / (volume - remaining_volume)
            temp_impact = (avg_execution_price - price) / price
        else:
            temp_impact = 0
        
        perm_impact = temp_impact * 0.2  # 20% permanent
        total_impact = temp_impact + perm_impact
        
        return {
            'temporary_impact': temp_impact,
            'permanent_impact': perm_impact,
            'total_impact': total_impact,
            'impact_price': price * (1 + total_impact),
            'original_price': price,
            'executed_volume': volume - remaining_volume,
            'unfilled_volume': remaining_volume
        }

class ImpactDecayModel:
    """
    Model for temporary impact decay over time
    """
    
    def __init__(self, decay_rate: float = 0.1, half_life: float = 300):
        self.decay_rate = decay_rate
        self.half_life = half_life
    
    def exponential_decay(self, initial_impact: float, time_elapsed: float) -> float:
        """Exponential decay of temporary impact"""
        return initial_impact * np.exp(-self.decay_rate * time_elapsed)
    
    def power_law_decay(self, initial_impact: float, time_elapsed: float, 
                       exponent: float = -0.5) -> float:
        """Power law decay of temporary impact"""
        if time_elapsed <= 0:
            return initial_impact
        return initial_impact * np.power(time_elapsed, exponent)
    
    def half_life_decay(self, initial_impact: float, time_elapsed: float) -> float:
        """Half-life based decay"""
        return initial_impact * np.power(0.5, time_elapsed / self.half_life)

class MultiAssetImpactModel:
    """
    Market impact model for multiple correlated assets
    """
    
    def __init__(self, correlation_matrix: np.ndarray, 
                 individual_models: List[MarketImpactModel]):
        self.correlation_matrix = correlation_matrix
        self.individual_models = individual_models
        self.n_assets = len(individual_models)
    
    def calculate_cross_impact(self, volumes: List[float], prices: List[float],
                             sides: List[str], **kwargs) -> Dict[str, Any]:
        """Calculate cross-asset impact effects"""
        # Individual impacts
        individual_impacts = []
        for i in range(self.n_assets):
            impact = self.individual_models[i].calculate_impact(
                volumes[i], prices[i], sides[i], **kwargs
            )
            individual_impacts.append(impact['total_impact'])
        
        # Cross impacts through correlation
        impact_vector = np.array(individual_impacts)
        cross_impacts = self.correlation_matrix @ impact_vector
        
        # Adjust for cross effects
        total_impacts = impact_vector + 0.1 * cross_impacts  # 10% cross effect
        
        return {
            'individual_impacts': individual_impacts,
            'cross_impacts': cross_impacts.tolist(),
            'total_impacts': total_impacts.tolist(),
            'correlation_matrix': self.correlation_matrix
        }

def compare_impact_models(volume: float, price: float, side: str,
                         volatility: float = 0.02) -> pd.DataFrame:
    """
    Compare different impact models for the same order
    """
    params = ImpactParameters()
    
    models = {
        'Linear': LinearImpactModel(params),
        'Square Root': SquareRootImpactModel(params),
        'Power Law': PowerLawImpactModel(params),
        'Almgren-Chriss': AlmgrenChrisImpactModel(params),
        'Liquidity': LiquidityImpactModel(params)
    }
    
    results = []
    for name, model in models.items():
        impact = model.calculate_impact(volume, price, side, volatility=volatility)
        results.append({
            'Model': name,
            'Temporary Impact (%)': impact['temporary_impact'] * 100,
            'Permanent Impact (%)': impact['permanent_impact'] * 100,
            'Total Impact (%)': impact['total_impact'] * 100,
            'Impact Price': impact['impact_price']
        })
    
    return pd.DataFrame(results)

def plot_impact_comparison(volumes: np.ndarray, price: float = 100, 
                          side: str = 'BUY', volatility: float = 0.02):
    """
    Plot impact comparison across different order sizes
    """
    params = ImpactParameters()
    models = {
        'Linear': LinearImpactModel(params),
        'Square Root': SquareRootImpactModel(params),
        'Power Law': PowerLawImpactModel(params)
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for name, model in models.items():
        impacts = []
        for volume in volumes:
            impact = model.calculate_impact(volume, price, side, volatility=volatility)
            impacts.append(impact['total_impact'] * 100)  # Convert to percentage
        
        ax1.plot(volumes, impacts, label=name, linewidth=2)
        ax2.loglog(volumes, impacts, label=name, linewidth=2)
    
    ax1.set_xlabel('Order Volume')
    ax1.set_ylabel('Total Impact (%)')
    ax1.set_title('Market Impact vs Order Volume')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Order Volume (log scale)')
    ax2.set_ylabel('Total Impact (%, log scale)')
    ax2.set_title('Market Impact vs Order Volume (Log-Log)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_impact_decay(initial_impact: float = 0.001, time_horizon: float = 3600):
    """
    Plot different impact decay models
    """
    decay_model = ImpactDecayModel()
    time_points = np.linspace(0, time_horizon, 1000)
    
    # Different decay models
    exponential = [decay_model.exponential_decay(initial_impact, t) for t in time_points]
    power_law = [decay_model.power_law_decay(initial_impact, t + 1) for t in time_points]
    half_life = [decay_model.half_life_decay(initial_impact, t) for t in time_points]
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_points / 60, np.array(exponential) * 100, 
             label='Exponential Decay', linewidth=2)
    plt.plot(time_points / 60, np.array(power_law) * 100, 
             label='Power Law Decay', linewidth=2)
    plt.plot(time_points / 60, np.array(half_life) * 100, 
             label='Half-life Decay', linewidth=2)
    
    plt.xlabel('Time (minutes)')
    plt.ylabel('Remaining Impact (%)')
    plt.title('Temporary Impact Decay Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()

# Example usage
if __name__ == "__main__":
    print("=== Market Impact Models Example ===")
    
    # Test parameters
    volume = 50000
    price = 100.0
    side = 'BUY'
    volatility = 0.02
    
    # Compare models
    print("\nComparing impact models for a large buy order:")
    comparison = compare_impact_models(volume, price, side, volatility)
    print(comparison.to_string(index=False))
    
    # Plot impact vs volume
    print("\nGenerating impact vs volume plots...")
    volumes = np.logspace(2, 6, 50)  # From 100 to 1,000,000
    plot_impact_comparison(volumes, price, side, volatility)
    
    # Plot decay models
    print("\nGenerating impact decay plots...")
    plot_impact_decay()
    
    # Almgren-Chriss optimal execution
    print("\n=== Almgren-Chriss Optimal Execution ===")
    params = ImpactParameters()
    ac_model = AlmgrenChrisImpactModel(params, risk_aversion=1e-6)
    
    # Calculate optimal execution
    volume = 100000
    time_horizon = 3600  # 1 hour
    
    t, x_t = ac_model.optimal_execution_schedule(volume, time_horizon, volatility)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t * 60, x_t, linewidth=2, label='Optimal Inventory')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Remaining Inventory')
    plt.title('Almgren-Chriss Optimal Execution Schedule')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    # Calculate execution impact
    impact_result = ac_model.calculate_impact(volume, price, side, volatility, time_horizon)
    print(f"Optimal execution impact: {impact_result['total_impact'] * 100:.3f}%")
    print(f"Optimal execution rate: {impact_result['optimal_execution_rate']:.2f} shares/day")