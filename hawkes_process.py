import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import factorial
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class HawkesProcess:
    """
    Implementation of Hawkes process for modeling self-exciting event arrivals
    in financial markets. The intensity function is:
    
    λ(t) = μ + Σ α * exp(-β * (t - tᵢ)) for all tᵢ < t
    
    Where:
    - μ: baseline intensity (background rate)
    - α: self-excitation parameter (jump size)
    - β: decay parameter (decay rate)
    """
    
    def __init__(self, mu: float = 1.0, alpha: float = 0.5, beta: float = 2.0):
        """
        Initialize Hawkes process parameters
        
        Args:
            mu: Baseline intensity
            alpha: Self-excitation parameter (must be < beta for stability)
            beta: Decay parameter
        """
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        
        # Check stability condition
        if alpha >= beta:
            warnings.warn("Warning: α ≥ β may lead to explosive behavior")
    
    def intensity(self, t: float, event_times: List[float]) -> float:
        """
        Calculate intensity at time t given previous events
        
        Args:
            t: Current time
            event_times: List of previous event times
            
        Returns:
            Intensity value at time t
        """
        base_intensity = self.mu
        
        # Sum contributions from all previous events
        excitation = 0
        for ti in event_times:
            if ti < t:
                excitation += self.alpha * np.exp(-self.beta * (t - ti))
        
        return base_intensity + excitation
    
    def simulate(self, T: float, random_seed: Optional[int] = None) -> List[float]:
        """
        Simulate Hawkes process using thinning algorithm
        
        Args:
            T: Simulation time horizon
            random_seed: Random seed for reproducibility
            
        Returns:
            List of event times
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        events = []
        t = 0
        
        # Upper bound for intensity (conservative estimate)
        lambda_max = self.mu + self.alpha / (1 - self.alpha / self.beta) if self.alpha < self.beta else self.mu * 10
        
        while t < T:
            # Generate candidate time
            u = np.random.exponential(1 / lambda_max)
            t += u
            
            if t >= T:
                break
            
            # Calculate actual intensity
            current_intensity = self.intensity(t, events)
            
            # Accept with probability intensity/lambda_max
            if np.random.random() < current_intensity / lambda_max:
                events.append(t)
        
        return events
    
    def simulate_with_intensities(self, T: float, dt: float = 0.01, 
                                 random_seed: Optional[int] = None) -> Tuple[List[float], np.ndarray, np.ndarray]:
        """
        Simulate Hawkes process and return intensity path
        
        Args:
            T: Simulation time horizon
            dt: Time step for intensity calculation
            random_seed: Random seed
            
        Returns:
            Tuple of (event_times, time_grid, intensity_path)
        """
        events = self.simulate(T, random_seed)
        
        # Calculate intensity path
        time_grid = np.arange(0, T, dt)
        intensity_path = np.array([self.intensity(t, events) for t in time_grid])
        
        return events, time_grid, intensity_path
    
    def log_likelihood(self, event_times: List[float], T: float) -> float:
        """
        Calculate log-likelihood of observed events
        
        Args:
            event_times: Observed event times
            T: Observation period
            
        Returns:
            Log-likelihood value
        """
        if len(event_times) == 0:
            return -self.mu * T
        
        log_lik = 0
        
        # Log of intensities at event times
        for i, ti in enumerate(event_times):
            intensity_i = self.intensity(ti, event_times[:i])
            log_lik += np.log(intensity_i)
        
        # Integral of intensity over [0, T]
        integral = self.mu * T
        
        for i, ti in enumerate(event_times):
            # Contribution from this event to all future times
            remaining_time = T - ti
            integral += (self.alpha / self.beta) * (1 - np.exp(-self.beta * remaining_time))
        
        log_lik -= integral
        
        return log_lik
    
    def fit(self, event_times: List[float], T: float, 
            method: str = 'L-BFGS-B') -> Dict[str, Any]:
        """
        Fit Hawkes process parameters to observed data
        
        Args:
            event_times: Observed event times
            T: Observation period
            method: Optimization method
            
        Returns:
            Dictionary with fitted parameters and optimization results
        """
        def negative_log_likelihood(params):
            mu, alpha, beta = params
            if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
                return 1e10
            
            # Temporarily set parameters
            old_params = (self.mu, self.alpha, self.beta)
            self.mu, self.alpha, self.beta = mu, alpha, beta
            
            nll = -self.log_likelihood(event_times, T)
            
            # Restore old parameters
            self.mu, self.alpha, self.beta = old_params
            
            return nll
        
        # Initial guess
        x0 = [self.mu, self.alpha, self.beta]
        
        # Bounds: all parameters positive, alpha < beta
        bounds = [(1e-6, None), (1e-6, None), (1e-6, None)]
        
        # Optimization
        result = minimize(negative_log_likelihood, x0, method=method, bounds=bounds)
        
        if result.success:
            self.mu, self.alpha, self.beta = result.x
        
        return {
            'success': result.success,
            'parameters': {'mu': self.mu, 'alpha': self.alpha, 'beta': self.beta},
            'log_likelihood': -result.fun,
            'optimization_result': result
        }
    
    def branching_ratio(self) -> float:
        """
        Calculate branching ratio (expected number of children per parent event)
        
        Returns:
            Branching ratio α/β
        """
        return self.alpha / self.beta
    
    def stability_condition(self) -> bool:
        """
        Check if process is stable (branching ratio < 1)
        
        Returns:
            True if stable, False otherwise
        """
        return self.branching_ratio() < 1
    
    def expected_number_events(self, T: float) -> float:
        """
        Calculate expected number of events in time interval [0, T]
        
        Args:
            T: Time horizon
            
        Returns:
            Expected number of events
        """
        if self.stability_condition():
            return self.mu * T / (1 - self.branching_ratio())
        else:
            return float('inf')

class MultivariatehawkesProcess:
    """
    Multivariate Hawkes process for modeling cross-excitation between different event types
    (e.g., buy/sell orders, different assets)
    """
    
    def __init__(self, mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray):
        """
        Initialize multivariate Hawkes process
        
        Args:
            mu: Vector of baseline intensities (n_dims,)
            alpha: Matrix of excitation parameters (n_dims, n_dims)
            beta: Matrix of decay parameters (n_dims, n_dims)
        """
        self.n_dims = len(mu)
        self.mu = np.array(mu)
        self.alpha = np.array(alpha)
        self.beta = np.array(beta)
    
    def intensity(self, t: float, event_times: List[List[float]], dim: int) -> float:
        """
        Calculate intensity for dimension dim at time t
        
        Args:
            t: Current time
            event_times: List of event time lists for each dimension
            dim: Dimension index
            
        Returns:
            Intensity value
        """
        base_intensity = self.mu[dim]
        excitation = 0
        
        for j in range(self.n_dims):
            for ti in event_times[j]:
                if ti < t:
                    excitation += self.alpha[dim, j] * np.exp(-self.beta[dim, j] * (t - ti))
        
        return base_intensity + excitation
    
    def simulate(self, T: float, random_seed: Optional[int] = None) -> List[List[float]]:
        """
        Simulate multivariate Hawkes process
        
        Args:
            T: Simulation time horizon
            random_seed: Random seed
            
        Returns:
            List of event time lists for each dimension
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        events = [[] for _ in range(self.n_dims)]
        t = 0
        
        # Conservative upper bound for total intensity
        lambda_max = np.sum(self.mu) * 10
        
        while t < T:
            # Generate candidate time
            u = np.random.exponential(1 / lambda_max)
            t += u
            
            if t >= T:
                break
            
            # Calculate intensities for all dimensions
            intensities = [self.intensity(t, events, dim) for dim in range(self.n_dims)]
            total_intensity = sum(intensities)
            
            # Accept event with probability total_intensity/lambda_max
            if np.random.random() < total_intensity / lambda_max:
                # Choose which dimension the event belongs to
                probs = np.array(intensities) / total_intensity
                dim = np.random.choice(self.n_dims, p=probs)
                events[dim].append(t)
        
        return events

def plot_hawkes_simulation(events: List[float], time_grid: np.ndarray, 
                          intensity_path: np.ndarray, title: str = "Hawkes Process Simulation"):
    """
    Plot Hawkes process simulation results
    
    Args:
        events: List of event times
        time_grid: Time grid for intensity
        intensity_path: Intensity values
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot events as vertical lines
    ax1.eventplot([events], colors=['red'], linewidths=1.5)
    ax1.set_ylabel('Events')
    ax1.set_title(f'{title} - Event Times')
    ax1.grid(True, alpha=0.3)
    
    # Plot intensity
    ax2.plot(time_grid, intensity_path, 'b-', linewidth=2, label='Intensity λ(t)')
    ax2.axhline(y=np.mean(intensity_path), color='g', linestyle='--', 
                label=f'Mean intensity: {np.mean(intensity_path):.2f}')
    
    # Mark events on intensity plot
    event_intensities = []
    for event in events:
        idx = np.argmin(np.abs(time_grid - event))
        event_intensities.append(intensity_path[idx])
    
    ax2.scatter(events, event_intensities, color='red', s=50, zorder=5, 
                label=f'{len(events)} events')
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Intensity')
    ax2.set_title(f'{title} - Intensity Process')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage and testing
if __name__ == "__main__":
    # Example 1: Basic Hawkes process
    print("=== Hawkes Process Example ===")
    
    # Create Hawkes process
    hawkes = HawkesProcess(mu=1.0, alpha=0.8, beta=2.0)
    
    print(f"Parameters: μ={hawkes.mu}, α={hawkes.alpha}, β={hawkes.beta}")
    print(f"Branching ratio: {hawkes.branching_ratio():.3f}")
    print(f"Stable: {hawkes.stability_condition()}")
    
    # Simulate process
    T = 10  # 10 time units
    events, time_grid, intensity = hawkes.simulate_with_intensities(T, dt=0.05, random_seed=42)
    
    print(f"Generated {len(events)} events in {T} time units")
    print(f"Average rate: {len(events)/T:.2f} events per unit time")
    
    # Plot results
    plot_hawkes_simulation(events, time_grid, intensity)
    
    # Example 2: Parameter estimation
    print("\n=== Parameter Estimation Example ===")
    
    # Generate synthetic data
    true_hawkes = HawkesProcess(mu=0.5, alpha=1.2, beta=3.0)
    true_events = true_hawkes.simulate(T=20, random_seed=123)
    
    # Fit parameters
    fitted_hawkes = HawkesProcess(mu=1.0, alpha=0.5, beta=2.0)  # Initial guess
    fit_result = fitted_hawkes.fit(true_events, T=20)
    
    print("True parameters:")
    print(f"  μ={true_hawkes.mu}, α={true_hawkes.alpha}, β={true_hawkes.beta}")
    print("Fitted parameters:")
    print(f"  μ={fitted_hawkes.mu:.3f}, α={fitted_hawkes.alpha:.3f}, β={fitted_hawkes.beta:.3f}")
    print(f"Log-likelihood: {fit_result['log_likelihood']:.2f}")
    print(f"Optimization success: {fit_result['success']}")