import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm

def fetch_data(ticker: str, period="2y"):
    """Fetch daily data and calculate returns."""
    data = yf.download(ticker, period=period)

    # Calculate daily returns
    returns = data['Close'].pct_change().dropna()
    returns = returns.squeeze() # Convert to series if it's a dataframe

    # Calculate Mean (mu) and Volatility (sigma)
    mu = returns.mean()
    sigma = returns.std()

    # Get last price
    last_price = data['Close'].iloc[-1].item()

    return returns, mu, sigma, last_price

def calculate_var(returns, mu, sigma, confidence_level=0.95):
    """Calculate 95% Value at Risk (VaR) using historical and parametric methods."""
    # Historical VaR
    hist_var = np.percentile(returns, (1 - confidence_level) * 100)

    # Parametric VaR (assuming normal distribution)
    # Z-score for 95% is roughly 1.645
    z_score = norm.ppf(1 - confidence_level)
    param_var = mu + z_score * sigma

    return hist_var, param_var

def monte_carlo_simulation(last_price, mu, sigma, days=1008, num_simulations=100):
    """Perform a Monte Carlo Simulation to project a 4-year timeline of potential price paths."""
    # 4 years roughly equals 1008 trading days (252 * 4)
    dt = 1 # 1 day step

    # Generate random shocks
    shocks = np.random.normal(0, 1, (num_simulations, days))

    # Calculate daily returns using Geometric Brownian Motion (GBM)
    daily_returns = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks)

    # Create price paths
    price_paths = np.zeros_like(daily_returns)
    price_paths[:, 0] = last_price

    for t in range(1, days):
        price_paths[:, t] = price_paths[:, t-1] * daily_returns[:, t]

    return price_paths

def calculate_max_drawdown(price_paths):
    """Calculate Maximum Drawdown across all simulations."""
    # Calculate cumulative maximum for each path
    cum_max = np.maximum.accumulate(price_paths, axis=1)

    # Calculate drawdown
    drawdown = (price_paths - cum_max) / cum_max

    # Find max drawdown for each path and then take the average
    max_drawdowns = np.min(drawdown, axis=1)

    return np.mean(max_drawdowns)
