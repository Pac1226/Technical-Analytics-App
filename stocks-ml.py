"""Crypto Linear Regression App by Peter Lieberman"""

# Loads basic libraries and dependencies
from PIL import Image
import pandas as pd
import numpy as np
import datetime as dt
import os
import financialanalysis as fa
import streamlit as st
import yfinance as yf
import warnings
from pathlib import Path

# Imports the data visualization libraries
import matplotlib.pyplot as plt
import hvplot.pandas
import holoviews as hv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
hv.extension('bokeh')


# Application Page Configuration: Headers & Sidebar
st.set_option('deprecation.showPyplotGlobalUse', False)
st.header('Stock Price Technical Indicators App')

# Description of App in header
st.markdown("""
This app analyzes price data for any stock. It builds a linear regression parallel channel, 
returns asset performance statistics, and builds an algo-trading strategy based on different technical indicators.
* **Python Libraries:** pandas, numpy, streamlit, yfinance, financialanalysis, scikit-learn, plotly
* **Data Source::** [YFinance](https://finance.yahoo.com/)
* **Models:** linear regression and risk/return ratios
* **Charts:** all charts are interactive, expandable and downloadable
""")

# Sidebar widgets: stock and time period selection
st.sidebar.header('User Input Features')
st.sidebar.caption('Input any stock ticker and timeframe.')


# Widget to select stock ticker
selected_asset = st.sidebar.text_input('Stock Ticker', value="SPY")


# Widget to select timeperiod
start_date = st.sidebar.date_input("Start Date", value = dt.date(1993, 1, 22), min_value = dt.date(1990, 1, 31)).strftime('%Y-%m-%d')
end_date = st.sidebar.date_input("End Date").strftime('%Y-%m-%d')


# Analytics Section 1: Function for Linear Regressions #
st.markdown("""**Linear Regression Parallell Channel**""")
st.markdown("""Regression line of time and price with standard deviation channels and Simple Moving Averages.""")
st.caption(f"Ticker: {selected_asset}")

# YFinance API Pull and DataFrame cleaning
pull = yf.Ticker(f"{selected_asset}")
data = pull.history(start = start_date, end = end_date)
data = pd.DataFrame(data["Close"])       
data = data.rename(columns={"Close": "Price"})
data["Daily Returns"] = data["Price"].pct_change().dropna()
data["Cumulative Returns"] = (1 + data["Daily Returns"]).cumprod()
data = data.dropna()
data = data.round(2)

# Calculates 200-Day / 50-Day Simple Moving Averages(SMAs)
sma200 = data["Price"].rolling(window=200).mean()
sma50 = data["Price"].rolling(window=50).mean()

# Calculates standad deviation of cumulative returns    
std = data["Cumulative Returns"].std()
    
# Copies DataFrame for caching   
linear_regression_df = data
linear_regression_df.reset_index(inplace=True)
linear_regression_df.dropna(inplace=True)

# Utilizes financialanalysis (fa) module to build linear regression channels
X = linear_regression_df["Date"].to_list() # converts Series to list
X = fa.datetimeToFloatyear(X) # for example, 2020-07-01 becomes 2020.49589041
X = np.array(X) # converts list to a numpy array
X = X[::,None] # converts row vector to column vector (just column vector is acceptable)
y = linear_regression_df["Cumulative Returns"] # get y data (relative price)
y = y.values # converts Series to numpy
y = y[::,None] # row vector to column vector (just column vector is acceptable)
 
slope, intercept, x, fittedline = fa.timeseriesLinearRegression(linear_regression_df["Date"], linear_regression_df["Cumulative Returns"])

# Trendlines for standard deviation parallel channels
fittedline_upper_1 = fittedline + std
fittedline_lower_1 = fittedline - std
fittedline_upper_2 = fittedline + (std*2)
fittedline_lower_2 = fittedline - (std*2)
    
# Plotly chart with subplots (secondary y-axis)    
chart = make_subplots(specs=[[{"secondary_y" : True}]])
chart.add_trace(go.Scatter(x=linear_regression_df["Date"], y=linear_regression_df["Price"], name="Price", line_color="black"), secondary_y=True)
#chart.add_trace(go.Scatter(x=linear_regression_df["Date"], y=linear_regression_df["Cumulative Returns"], line_color="yellow", showlegend=False, hoverinfo='none'), secondary_y=True,)
chart.add_trace(go.Scatter(x=linear_regression_df["Date"], y=fittedline, name="Trendline", line_color="lightslategray", hoverinfo='none'), secondary_y=False,)
chart.add_trace(go.Scatter(x=linear_regression_df["Date"], y=fittedline_lower_1, name="Standard Deviation", line_color="forestgreen", hoverinfo='none'), secondary_y=False,)
chart.add_trace(go.Scatter(x=linear_regression_df["Date"], y=fittedline_upper_1, line_color="forestgreen", showlegend=False, hoverinfo='none'), secondary_y=False,)
chart.add_trace(go.Scatter(x=linear_regression_df["Date"], y=fittedline_lower_2, name="2 Standard Deviations", line_color="rosybrown", hoverinfo='none'), secondary_y=False,)
chart.add_trace(go.Scatter(x=linear_regression_df["Date"], y=fittedline_upper_2, name="2 Standard Deviations", line_color="rosybrown", showlegend=False, hoverinfo='none'), secondary_y=False,)
chart.add_trace(go.Scatter(x=linear_regression_df["Date"], y=sma200, name="200-Day SMA", line_color="gray"), secondary_y=True,)
chart.add_trace(go.Scatter(x=linear_regression_df["Date"], y=sma50, name="50-Day SMA", line_color="lightgray"), secondary_y=True,)

chart.update_xaxes(title_text = "Date", showline=False)
chart.update_yaxes(title_text="Actual Price", range=[linear_regression_df["Price"].min() * .6, linear_regression_df["Price"].max() * 1.2], zeroline = True, tickformat = '$', showgrid=True, tick0 = 0, secondary_y=True)
chart.update_yaxes(showticklabels = False, range=[linear_regression_df["Cumulative Returns"].min() * .6, linear_regression_df["Cumulative Returns"].max()* 1.2], tick0 = 0, secondary_y=False)
chart.update_layout(template="simple_white")
chart.update_traces(marker_colorscale="Earth", selector=dict(type='scatter'))
chart.update_traces(fill="none")
chart.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="left", x=.01, font = dict(size = 10, color = "black")))
chart.update_layout(plot_bgcolor='white')
chart.update_layout(margin=dict(l=0, r=0, t=55))
chart.update_yaxes(nticks = 10, secondary_y=True)

st.plotly_chart(chart)


risk_free_rate = .03 # necessary for Sortino/Sharpe Ratio calculations

# Function to display summary statistics and financial ratios
def get_token_statistics(data, start, end):

    # Calculates average daily returns and cumulative returns of the asset
    daily_returns = data["Daily Returns"]
    cumulative_returns = data["Cumulative Returns"]
    total_return = data["Cumulative Returns"].iloc[-1]
    annualized_return = data["Daily Returns"].mean() * 252
    peak = data["Cumulative Returns"].expanding(min_periods=1).max()
    ath = peak.max()

    # Calculates annualized returns / standard deviation, the variance, and max drawdown
    standard_deviation = data["Daily Returns"].std() * np.sqrt(252)
    max_drawdown = (data["Cumulative Returns"]/peak-1).min()
    negative_standard_deviation = data["Daily Returns"][data["Daily Returns"]<0].std() * np.sqrt(252)

    # Calculates the Sharpe, Sortino, & Calmar Ratios. Negative Annualized Standard Deviation is used for Sortino Ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / standard_deviation
    sortino_ratio = (annualized_return - risk_free_rate) / negative_standard_deviation
    calmar_ratio = (annualized_return - risk_free_rate) / (abs(max_drawdown))

    # Combines three metrics into a single DataFrame
    alist = []
    alist.append(calmar_ratio)
    alist.append(sortino_ratio)
    alist.append(sharpe_ratio)
    alist.append(max_drawdown)
    alist.append(ath)
    alist.append(standard_deviation)
    alist.append(total_return)
    token_statistics = pd.DataFrame(alist).T
    token_statistics.columns = ["Calmar Ratio", "Sortino Ratio", "Sharpe Ratio", "Max Drawdown", "Peak", "Annual Volatity", "Total Return"]
    token_statistics.index = [f"{selected_asset}"]
    token_statistics = token_statistics.round(2)

    return token_statistics

bar_chart = get_token_statistics(data, start_date, end_date).hvplot.bar(color="black", hover_color="green", rot=45)

st.markdown("""**Financial Ratios & Statistics**""")
st.markdown("""Risk/return metrics and performance ratios over selected time period.""")
st.caption(f"Ticker: {selected_asset}")
st.bokeh_chart(hv.render(bar_chart, backend="bokeh"))


# New section to backtest an algo-trading strategy leveraging the linear regression channel
st.markdown("""**Algo-Trading: Strategy Backtesting**""")
st.markdown("""Select a buy/sell signal and see how the strategy would have performed over the last year.
* **Buy Signal:** the algorithm BUYS a stock when it goes BELOW the signal
* **Sell Signal:** the algorithm SELLS a stock when it goes ABOVE the signal
The signals are created from a parallel linear regression channel for the last 12-months.
""")

# Widget for users to select buy/sell signals
buy_signal = st.selectbox('Buy Signal', ("Trendline", "1 Standard Deviation (Above)", "1 Standard Deviation (Below)", "2 Standard Deviations (Above)", "2 Standard Deviations (Below)"))
sell_signal = st.selectbox('Sell Signal', ("Trendline", "1 Standard Deviation (Above)", "1 Standard Deviation (Below)", "2 Standard Deviations (Above)", "2 Standard Deviations (Below)"))

# Sets dates for testing the strategy over 1-year rolling period
today = pd.to_datetime("today").strftime('%Y-%m-%d')
one_year_ago = (pd.to_datetime("today") - pd.DateOffset(years=1)).strftime('%Y-%m-%d')

# YFinance API pull
pull = yf.Ticker(f"{selected_asset}")
data = pull.history(start = one_year_ago, end = today)
data = pd.DataFrame(data["Close"])       
data = data.rename(columns={"Close": "Price"})

# Calculates daily returns and cumulative returns
data["Daily Returns"] = data["Price"].pct_change().dropna()
data["Cumulative Returns"] = (1 + data["Daily Returns"]).cumprod()
data = data.dropna()

# Standard deviation of cumulative returns
std = data["Cumulative Returns"].std()

# Copies DataFrame for the machine learning models
model_df = data
model_df.reset_index(inplace=True)
model_df.dropna(inplace=True)

# Builds the timeseries linear regression trendlinea
X = model_df["Date"].to_list() # converts Series to list
X = fa.datetimeToFloatyear(X) # for example, 2020-07-01 becomes 2020.49589041
X = np.array(X) # converts list to a numpy array
X = X[::,None] # converts row vector to column vector (just column vector is acceptable)
y = model_df["Cumulative Returns"] # get y data (relative price)
y = y.values # converts Series to numpy
y = y[::,None] # row vector to column vector (just column vector is acceptable)
 
slope, intercept, x, fittedline = fa.timeseriesLinearRegression(model_df["Date"], model_df["Cumulative Returns"])

# Creates standard deviation parallel channels from trendline
std_1_upper = fittedline + std
std_1_lower = fittedline - std
std_2_upper = fittedline + (std*2)
std_2_lower = fittedline - (std*2)

# Adds the trendlines to the model DataFrame
model_df["Trendline"] = fittedline
model_df["1 Standard Deviation (Above)"] = std_1_upper
model_df["1 Standard Deviation (Below)"] = std_1_lower
model_df["2 Standard Deviations (Above)"] = std_2_upper
model_df["2 Standard Deviations (Below)"] = std_2_lower

model_df.index = model_df["Date"]
model_df = model_df.drop(columns="Date")

# Initializes the trade Signal column
model_df['Signal'] = 0.0

# When Price is less than or equal to the mean, generates signal to buy stock long
model_df.loc[(model_df['Cumulative Returns'] < model_df[f"{buy_signal}"]), 'Signal'] = 1

# When Price is one standard deviation above the mean, generates signal to sell stock short
model_df.loc[(model_df['Cumulative Returns'] > model_df[f"{sell_signal}"]), 'Signal'] = -1

# Calculates the strategy returns and add them to the signals_df DataFrame
model_df['Strategy Returns'] = (1 + (model_df['Daily Returns'] * model_df['Signal'].shift().dropna())).cumprod()

model_df = model_df.dropna()
model_df["Actual Returns"] = model_df['Cumulative Returns']


# Plots strategies on a graph
actual_returns_graph = model_df["Actual Returns"].hvplot.line(hover_color="red", line_color = "black", legend="left", rot=45)
strategy_returns_graph = model_df["Strategy Returns"].hvplot.line(hover_color="black", line_color = "green", rot=45)
graph = actual_returns_graph * strategy_returns_graph
st.bokeh_chart(hv.render(graph, backend="bokeh"))   


# YFinance API Pull and DataFrame cleaning
stock_pull = yf.Ticker(f"{selected_asset}")
stock_data = stock_pull.history(start = pd.to_datetime("today") - pd.DateOffset(months=12), end = pd.to_datetime("today"))
stock_data = pd.DataFrame(stock_data["Close"])       
stock_data = stock_data.rename(columns={"Close": "Price"})
stock_data["Daily Returns"] = stock_data["Price"].pct_change().dropna()
stock_data["Cumulative Returns"] = (1 + stock_data["Daily Returns"]).cumprod()
stock_data = stock_data.dropna()
stock_data = stock_data.round(2)

spy_pull = yf.Ticker("SPY")
spy_data = spy_pull.history(start = pd.to_datetime("today") - pd.DateOffset(months=12), end = pd.to_datetime("today"))
spy_data = pd.DataFrame(spy_data["Close"])       
spy_data = spy_data.rename(columns={"Close": "Price"})
spy_data["Daily Returns"] = spy_data["Price"].pct_change().dropna()
spy_data["Cumulative Returns"] = (1 + spy_data["Daily Returns"]).cumprod()
spy_data = spy_data.dropna()
spy_data = spy_data.round(2)

beta_90d = (stock_data["Daily Returns"].tail(90).cov(spy_data["Daily Returns"].tail(90))) / spy_data["Daily Returns"].tail(90).var()
beta_180d = (stock_data["Daily Returns"].tail(180).cov(spy_data["Daily Returns"].tail(180))) / spy_data["Daily Returns"].tail(180).var()
beta_1year = (stock_data["Daily Returns"].tail(360).cov(spy_data["Daily Returns"].tail(360))) / spy_data["Daily Returns"].tail(360).var()

beta_90d = beta_90d.round(2)
beta_180d = beta_180d.round(2)
beta_1year = beta_1year.round(2)


st.sidebar.header('Asset Beta')
st.sidebar.caption(f"Real-time beta for {selected_asset} relative to the S&P 500.")
st.sidebar.metric("90-Day Beta", beta_90d, delta_color="off")
st.sidebar.metric("180-Day Beta", beta_180d, delta_color="off")
st.sidebar.metric("360-Day Beta", beta_1year, delta_color="off")
