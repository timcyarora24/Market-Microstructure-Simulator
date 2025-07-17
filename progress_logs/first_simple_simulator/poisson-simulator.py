
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng
import yfinance as yf

rng=default_rng()
apple=yf.Ticker("APPL")
apple_data=apple.history(period='1y')

def simulate_trade_arrival(arrival_rate , time_period=1):

    lambda_rate=arrival_rate*time_period

    trade_time=[]
    current_time=0
    while current_time<time_period:
        inter_arrival_time=rng.exponential(1/lambda_rate)
        current_time+=inter_arrival_time
        if current_time < time_period:
            trade_time.append(current_time)

    return trade_time

lambda_rate = rng.integers(1,1000)
print('lambda_rate' , lambda_rate )

time_period = rng.integers(1,7)
print('time_period' , time_period)

trade_times = simulate_trade_arrival(lambda_rate, time_period)
print(f'Total trades: {len(trade_times)}')

# Calculate inter-arrival times
if len(trade_times) > 1:
    inter_arrival_times = np.diff([0] + trade_times)
else:
    inter_arrival_times = trade_times

# Create the visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Trade Arrival Simulation Analysis', fontsize=16, fontweight='bold')

# 1. Trade Timeline (Scatter Plot)
ax1.scatter(trade_times, range(1, len(trade_times) + 1), 
           alpha=0.7, s=50, c='blue', edgecolors='black')
ax1.set_xlabel('Time')
ax1.set_ylabel('Trade Number')
ax1.set_title('Trade Timeline - When Each Trade Occurs')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, time_period)

# 2. Inter-Arrival Time Distribution (Histogram)
ax2.hist(inter_arrival_times, bins=min(20, len(inter_arrival_times)), 
         alpha=0.7, color='green', edgecolor='black', density=True)
ax2.set_xlabel('Inter-Arrival Time')
ax2.set_ylabel('Density')
ax2.set_title('Distribution of Waiting Times Between Trades')
ax2.grid(True, alpha=0.3)

# Add theoretical exponential curve
if len(inter_arrival_times) > 0:
    x_theory = np.linspace(0, max(inter_arrival_times), 100)
    y_theory = (lambda_rate/time_period) * np.exp(-(lambda_rate/time_period) * x_theory)
    ax2.plot(x_theory, y_theory, 'r-', linewidth=2, 
             label=f'Theoretical Exponential (λ={lambda_rate/time_period:.2f})')
    ax2.legend()

# 3. Cumulative Trade Count Over Time
cumulative_trades = range(1, len(trade_times) + 1)
ax3.plot(trade_times, cumulative_trades, 'o-', linewidth=2, markersize=4, color='red')
ax3.set_xlabel('Time')
ax3.set_ylabel('Cumulative Number of Trades')
ax3.set_title('Cumulative Trade Count Over Time')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, time_period)

# Add theoretical line
theoretical_line = np.linspace(0, time_period, 100)
expected_trades = (lambda_rate/time_period) * theoretical_line
ax3.plot(theoretical_line, expected_trades, '--', color='orange', linewidth=2, 
         label=f'Expected Rate (λ={lambda_rate/time_period:.2f})')
ax3.legend()

# 4. Inter-Arrival Times as Time Series
if len(inter_arrival_times) > 0:
    ax4.plot(range(1, len(inter_arrival_times) + 1), inter_arrival_times, 
             'o-', linewidth=1, markersize=4, color='purple')
    ax4.axhline(y=np.mean(inter_arrival_times), color='orange', linestyle='--', 
                label=f'Mean = {np.mean(inter_arrival_times):.3f}')
    ax4.set_xlabel('Trade Sequence (nth waiting time)')
    ax4.set_ylabel('Inter-Arrival Time')
    ax4.set_title('Waiting Times Between Consecutive Trades')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*50)
print("SIMULATION SUMMARY STATISTICS")
print("="*50)
print(f"Time Period: {time_period}")
print(f"Lambda Rate: {lambda_rate}")
print(f"Expected Trades: {lambda_rate:.2f}")
print(f"Actual Trades: {len(trade_times)}")
print(f"Average Inter-Arrival Time: {np.mean(inter_arrival_times):.4f}")
print(f"Theoretical Mean: {time_period/lambda_rate:.4f}")
if len(inter_arrival_times) > 1:
    print(f"Std Dev of Inter-Arrival Times: {np.std(inter_arrival_times):.4f}")
print("="*50)






