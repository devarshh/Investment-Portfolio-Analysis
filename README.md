# Investment Portfolio Analyzer

## Overview

The **Investment Portfolio Analyzer** is a Python-based tool that helps investors evaluate their portfolio's performance. It provides key metrics, risk analysis, and visual insights, making it a comprehensive solution for portfolio management.

### Key Features
- **Dynamic User Input**: Enter your portfolio stocks, weights, and initial investment dynamically via the CLI.
- **Portfolio Metrics**:
  - ROI (Return on Investment)
  - Annualized Volatility
  - Sharpe Ratio
  - Sortino Ratio
  - Maximum Drawdown
  - CAGR (Compound Annual Growth Rate)
  - Beta (Portfolio vs. Benchmark)
- **Visualizations**:
  - Cumulative Returns
  - Correlation Heatmap
  - Monte Carlo Simulation
- **Detailed Report**: Generates a text-based summary (`portfolio_report.txt`) of key performance metrics.

---

## Requirements

### Dependencies
- Python 3.7+
- Libraries:
  - `requests`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scipy`

Install the dependencies using pip:
```bash
pip install requests pandas numpy matplotlib seaborn scipy
```
## API Key

Obtain a free API key from [Alpha Vantage](https://www.alphavantage.co/) and replace `YOUR_ALPHA_VANTAGE_API_KEY` in the script.

---

## How to Use

### Run the Script:
```bash
python investment_portfolio_analyzer.py
```

## How to Use

### Provide Inputs:
- **Enter Stock Symbols**: Input stock symbols (e.g., `AAPL, GOOGL, NVDA`).
- **Provide Weights**: Specify weights as decimals that sum to 1 (e.g., `0.4, 0.35, 0.25`).
- **Specify Investment**: Enter your initial investment amount (e.g., `100000`).
- **Enter Benchmark**: Provide the benchmark symbol for comparison (e.g., `SPY`).

### View Results:
- The script will calculate various metrics, such as ROI, Sharpe Ratio, and CAGR, and display insights directly in the terminal.
- Visualizations like cumulative returns and Monte Carlo simulations will be displayed for further insights.

### Generated Report:
- A detailed report summarizing key metrics will be saved as `portfolio_report.txt`.

---

## Output Examples

### 1. Portfolio Analysis Summary:
```yaml
--- Portfolio Analysis Summary ---
ROI: 15.25%
Annualized Volatility: 0.18
Sharpe Ratio: 0.85
Sortino Ratio: 1.20
Maximum Drawdown: -12.35%
CAGR: 13.42%
Portfolio Beta: 1.05
```
## Visualizations

### Cumulative Returns:
Compares portfolio growth with the benchmark over time.

### Correlation Heatmap:
Illustrates relationships and dependencies between portfolio assets.

### Monte Carlo Simulations:
Forecasts potential portfolio performance over a year, showcasing risk and return possibilities.

---

## Improvements

### Error Handling:
- Gracefully handles invalid symbols or weights provided by the user.
- Incorporates retry logic to manage API limitations, ensuring uninterrupted operations.

### Performance:
- Optimized for efficient data fetching and analysis processes.
- Supports dynamic portfolio changes without requiring modifications to the script.

---

## Future Enhancements

- **Integration**: Add support for other data sources (e.g., Yahoo Finance API) to increase flexibility.
- **Rebalancing Suggestions**: Provide recommendations for portfolio rebalancing based on market conditions.
- **Advanced Risk Metrics**: Include Value at Risk (VaR) and Conditional VaR for deeper risk analysis.

---
