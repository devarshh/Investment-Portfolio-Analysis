import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.stats import norm

# Set Seaborn style
sns.set(style='darkgrid')

# Your Alpha Vantage API Key
API_KEY = 'YOUR_ALPHA_VANTAGE_API_KEY'

# Greet the user
print("Welcome to the Investment Portfolio Analyzer!")
print("We'll help you analyze your portfolio with advanced metrics and visuals.")

# CLI for dynamic user input
portfolio_symbols = input("Please enter the stock symbols in your portfolio (separated by commas): ").strip().split(',')
weights = np.array(list(map(float, input(f"Enter weights for these stocks {portfolio_symbols} (comma-separated, must sum to 1): ").strip().split(','))))
initial_investment = float(input("What is your initial investment amount? "))
benchmark_symbol = input("Which benchmark symbol would you like to compare to (e.g., SPY)? ").strip()

# Function to fetch time series data for a symbol
def fetch_time_series(symbol, outputsize='compact'):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize={outputsize}&apikey={API_KEY}'
    while True:
        response = requests.get(url)
        data = response.json()
        if 'Time Series (Daily)' in data:
            print(f"Data successfully fetched for {symbol}.")
            break
        elif 'Note' in data:
            print(f"API limit reached for today. Retrying in 60 seconds...")
            time.sleep(60)
        else:
            print(f"There was an issue fetching data for {symbol}: {data.get('Error Message', 'Unknown error')}. Skipping this symbol.")
            return pd.DataFrame()
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
    data = fetch_time_series(symbol, outputsize='full')
    if not data.empty:
        portfolio_data[symbol] = data

# Fetch data for the benchmark
print(f"Fetching data for benchmark {benchmark_symbol}...")
benchmark_data = fetch_time_series(benchmark_symbol, outputsize='full')

# Combine portfolio data into a single DataFrame
combined_df = pd.DataFrame()
for symbol in portfolio_data:
    combined_df[symbol] = portfolio_data[symbol]['Close']
combined_df.dropna(inplace=True)

if combined_df.empty:
    print("No valid data available for the provided symbols. Please try again.")
    exit()

# Calculate daily returns
daily_returns = combined_df.pct_change().dropna()

# Calculate portfolio daily returns
portfolio_daily_returns = daily_returns.dot(weights)

# Calculate cumulative returns
portfolio_cumulative_returns = (1 + portfolio_daily_returns).cumprod() - 1

# Calculate portfolio volatility and Sharpe Ratio
portfolio_volatility = portfolio_daily_returns.std() * np.sqrt(252)  # Annualized volatility
portfolio_return = portfolio_daily_returns.mean() * 252
sharpe_ratio = portfolio_return / portfolio_volatility

# Calculate Maximum Drawdown
rolling_max = (1 + portfolio_cumulative_returns).cummax()
drawdown = (1 + portfolio_cumulative_returns) / rolling_max - 1
max_drawdown = drawdown.min()

# Calculate CAGR
years = (portfolio_cumulative_returns.index[-1] - portfolio_cumulative_returns.index[0]).days / 365.25
cagr = ((1 + portfolio_cumulative_returns.iloc[-1]) ** (1 / years)) - 1

# Calculate Beta
benchmark_daily_returns = benchmark_data['Close'].pct_change().dropna()
common_dates = daily_returns.index.intersection(benchmark_daily_returns.index)
portfolio_returns_aligned = portfolio_daily_returns.loc[common_dates]
benchmark_returns_aligned = benchmark_daily_returns.loc[common_dates]
beta = np.cov(portfolio_returns_aligned, benchmark_returns_aligned)[0, 1] / np.var(benchmark_returns_aligned)

# Calculate Sortino Ratio
negative_returns = portfolio_daily_returns[portfolio_daily_returns < 0]
portfolio_downside_deviation = np.sqrt((negative_returns ** 2).mean()) * np.sqrt(252)
sortino_ratio = portfolio_return / portfolio_downside_deviation

# Print friendly results
print("\n--- Portfolio Analysis Summary ---")
print(f"ROI: {portfolio_cumulative_returns.iloc[-1] * 100:.2f}%")
print(f"Annualized Volatility: {portfolio_volatility:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Sortino Ratio: {sortino_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")
print(f"CAGR: {cagr * 100:.2f}%")
print(f"Portfolio Beta: {beta:.2f}")

# Correlation Heatmap
print("\nGenerating correlation heatmap...")
plt.figure(figsize=(10, 8))
sns.heatmap(daily_returns.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Portfolio Assets')
plt.show()

# Monte Carlo Simulation
def monte_carlo_simulation(returns, num_simulations=1000, num_days=252):
    np.random.seed(42)
    simulations = np.zeros((num_simulations, num_days))
    for i in range(num_simulations):
        simulated_returns = np.random.choice(returns, size=num_days, replace=True)
        simulations[i, :] = np.cumprod(1 + simulated_returns)
    return simulations

print("\nRunning Monte Carlo Simulation...")
simulations = monte_carlo_simulation(portfolio_daily_returns)
plt.figure(figsize=(10, 6))
plt.plot(simulations.T, color='lightblue', alpha=0.1)
plt.title('Monte Carlo Simulation of Portfolio Value')
plt.xlabel('Days')
plt.ylabel('Portfolio Value Growth')
plt.show()

# Generate Report
def generate_report():
    with open("portfolio_report.txt", "w") as file:
        file.write("Portfolio Analysis Report\n")
        file.write("==========================\n")
        file.write(f"ROI: {portfolio_cumulative_returns.iloc[-1] * 100:.2f}%\n")
        file.write(f"Volatility: {portfolio_volatility:.2f}\n")
        file.write(f"Sharpe Ratio: {sharpe_ratio:.2f}\n")
        file.write(f"Sortino Ratio: {sortino_ratio:.2f}\n")
        file.write(f"Max Drawdown: {max_drawdown * 100:.2f}%\n")
        file.write(f"CAGR: {cagr * 100:.2f}%\n")
        file.write(f"Beta: {beta:.2f}\n")

print("\nGenerating a detailed report...")
generate_report()
print("Report successfully saved as 'portfolio_report.txt'.")

print("\nThank you for using the Investment Portfolio Analyzer. Happy investing!")
