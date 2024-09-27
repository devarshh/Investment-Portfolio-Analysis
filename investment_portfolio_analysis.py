# investment_portfolio_analysis.py

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set Seaborn style
sns.set(style='darkgrid')

# Your Alpha Vantage API Key
API_KEY = 'YOUR_ALPHA_VANTAGE_API_KEY'

# List of stock symbols in your portfolio
portfolio_symbols = ['AAPL', 'GOOGL', 'NVDA']  # Example symbols
weights = np.array([0.4, 0.35, 0.25])  # Corresponding weights

# Market benchmark symbol (e.g., S&P 500)
benchmark_symbol = 'SPY'

# Function to fetch time series data for a symbol
def fetch_time_series(symbol, outputsize='compact'):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize={outputsize}&apikey={API_KEY}'
    while True:
        response = requests.get(url)
        data = response.json()
        if 'Time Series (Daily)' in data:
            break
        else:
            print(f"API call limit reached. Waiting for 60 seconds...")
            time.sleep(60)
    df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.rename(columns={'4. close': 'Close'}, inplace=True)
    return df[['Close']]

# Fetch data for each stock in the portfolio
portfolio_data = {}
for symbol in portfolio_symbols:
    print(f"Fetching data for {symbol}...")
    portfolio_data[symbol] = fetch_time_series(symbol, outputsize='full')

# Fetch data for the benchmark
print(f"Fetching data for benchmark {benchmark_symbol}...")
benchmark_data = fetch_time_series(benchmark_symbol, outputsize='full')

# Combine portfolio data into a single DataFrame
combined_df = pd.DataFrame()
for symbol in portfolio_symbols:
    combined_df[symbol] = portfolio_data[symbol]['Close']
combined_df.dropna(inplace=True)

# Calculate daily returns
daily_returns = combined_df.pct_change().dropna()

# Calculate portfolio daily returns
portfolio_daily_returns = daily_returns.dot(weights)

# Calculate cumulative returns
portfolio_cumulative_returns = (1 + portfolio_daily_returns).cumprod() - 1

# Calculate portfolio volatility
portfolio_volatility = portfolio_daily_returns.std() * np.sqrt(252)  # Annualized volatility

# Calculate ROI
initial_investment = 100000  # Example initial investment amount
portfolio_value = initial_investment * (1 + portfolio_cumulative_returns.iloc[-1])
roi = (portfolio_value - initial_investment) / initial_investment * 100

print(f"Portfolio ROI: {roi:.2f}%")
print(f"Portfolio Volatility: {portfolio_volatility:.2f}")

# Process benchmark data
benchmark_data = benchmark_data.loc[combined_df.index]
benchmark_returns = benchmark_data['Close'].pct_change().dropna()
benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() - 1

# Plot cumulative returns
plt.figure(figsize=(14, 7))
plt.plot(portfolio_cumulative_returns.index, portfolio_cumulative_returns, label='Portfolio')
plt.plot(benchmark_cumulative_returns.index, benchmark_cumulative_returns, label=benchmark_symbol)
plt.title('Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()

# Plot portfolio allocation
plt.figure(figsize=(7, 7))
plt.pie(weights, labels=portfolio_symbols, autopct='%1.1f%%', startangle=140)
plt.title('Portfolio Allocation')
plt.show()

# Optimize asset allocation (simple simulation)
def optimize_portfolio(returns, num_portfolios=5000):
    np.random.seed(42)
    num_assets = len(portfolio_symbols)
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        # Randomly assign weights
        weights_sim = np.random.random(num_assets)
        weights_sim /= np.sum(weights_sim)
        # Calculate portfolio return and volatility
        portfolio_return = np.sum(returns.mean() * weights_sim) * 252
        portfolio_volatility = np.sqrt(np.dot(weights_sim.T, np.dot(returns.cov() * 252, weights_sim)))
        # Sharpe Ratio (assuming risk-free rate is 0)
        sharpe_ratio = portfolio_return / portfolio_volatility
        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = sharpe_ratio
    return results

# Perform optimization
print("Optimizing portfolio...")
results = optimize_portfolio(daily_returns)

# Plot Efficient Frontier
plt.figure(figsize=(10, 7))
plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis', marker='o', s=10, alpha=0.3)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(portfolio_volatility, portfolio_daily_returns.mean() * 252, color='red', marker='*', s=500, label='Current Portfolio')
plt.title('Efficient Frontier')
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.legend()
plt.show()
