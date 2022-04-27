# Crypto Web Application

This is an interactive web application hosted on Streamlit that performs statistical analysis on cryptocurrency prices based on user inputs. It connects to the Messari API, which aggreggates data from crypto exchanges, protocols, and analytics firms. The app runs a series of statistical models on timeseries price data and displays key insights as data visualizations.


## Technologies

```python
The program uses Pandas, NumPy, FinancialAnalysis, Messari, Scikit-learn, hvPlot, Matplotlib, and sevaral custom built functions. 
```
---

## Installation Guide

FinancialAnalysis and Messari.Messari are required to run the Jupyter Notebook locally on your computer. There are four additional modules in the "formulas" folder that the application also depends on.

---

## Crypto Assets

The application aggregates, cleans, and runs models on timeseries price data collected on twelve (12) Layer One blockchain protocols.

* Bitcoin (BTC)
* Ethereum (ETH)
* BNB Chain (BNB)
* Solana (SOL)
* Cardano (ADA)
* Terra (LUNA)
* Avalanche (AVAX)
* Polygon (MATIC)
* Polkadot (DOT)
* NEAR Protocol (NEAR)
* Cosmos (ATOM)
* Algorand (ALGO)

---

## Usage

There are two user inputs: the timeframe and cryptocurrency. Users select one of the twelve currencies to generate insights on in addition to the time period. For the time period, the user selects the number of months to look back. The app then displays the following insights:

1) Linear Regressions: performs a linear regression on time and price that produces a parellel channel that consists of the mean and the standard deviations above and below the one. This is used to guide a "mean reversion" trading strategy.

2) Financial Ratios & Performance Statistics: shows the price peak-to-trough and Sharpe, Sortino, and Calmar ratios over the last 12 months

3) Assets Correlations: shows the correlation of the crypto assets on a rolling 12 month basis, displayed as a heatmap

---

## Usage

MIT License
