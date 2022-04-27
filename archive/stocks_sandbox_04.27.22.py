"""Crypto Linear Regression App by Peter Lieberman"""

# Loads basic libraries and dependencies
from PIL import Image
import pandas as pd
import numpy as np
import datetime as dt
import os
import financialanalysis as fa
import streamlit as st
import alpaca_trade_api as tradeapi
import matplotlib.pyplot as plt
import hvplot.pandas
import holoviews as hv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
hv.extension('bokeh')

# API keys & Streamlit secrerts
alpaca_api_key = st.secrets["ALPACA_API_KEY"]
alpaca_secret_key = st.secrets["ALPACA_SECRET_KEY"]

# Create the Alpaca API object
alpaca = tradeapi.REST(alpaca_api_key, alpaca_secret_key, api_version="v3")

# Application Page Configuration: Headers & Sidebar #

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header('Stock Linear Regression')

st.markdown("""
This app connects to crypto APIs and runs a series of models 
to assess past performance and predict future price trends!
* **Python libraries:** pandas, numpy, os, streamlit, messari.messari, financialanalysis, scikit-learn
* **Data source:** [Messari.io](https://messari.io/api)
* **Models:** linear regression, risk/return analysis, and statistical correlations
* **Charts:** all charts are interactive and can be saved as images
""")


# Sidebar widgets: cryptocurrency and time period selection
st.sidebar.header('User Input Features')
st.sidebar.caption('Input any ticker symbol and select timeframe.')


# Widget to select stock ticker
selected_asset = st.sidebar.text_input('Stock Ticker')

# Widget to select timeperiod
start_date = st.sidebar.date_input("Start Date").strftime("%Y-%m-%d")
end_date = st.sidebar.date_input("End Date").strftime("%Y-%m-%d")
timeframe = "1D"

# Analytics Section 1: Function for Linear Regressions #

st.markdown("""**Linear Regression Channel**""")
st.markdown("""Regression line of time and price with standard deviation channels and Simple Moving Averages.""")

df = alpaca.get_bars(selected_asset, timeframe="1D", start = start_date, end = end_date).df

ticker_df = pd.DataFrame(df["close"])
ticker_df = ticker_df.rename(columns={"close": "Price"})
ticker_df.reset_index(inplace=True)
ticker_df = ticker_df.rename(columns={"timestamp": "Date"})
ticker_df["Date"] = ticker_df["Date"].dt.strftime('%Y-%m-%d')
ticker_df["Date"] = pd.to_datetime(ticker_df["Date"], infer_datetime_format=True)
ticker_df["Date"] = ticker_df["Date"].dt.date
ticker_df = ticker_df.set_index("Date")
ticker_df["Daily Returns"] = ticker_df["Price"].pct_change().dropna()
ticker_df["Cumulative Returns"] = (1 + ticker_df["Daily Returns"]).cumprod()
ticker_df.dropna()
ticker_df = ticker_df.round(2)

sma200 = ticker_df["Price"].rolling(window=200).mean()
sma50 = ticker_df["Price"].rolling(window=50).mean()
    
std = ticker_df["Cumulative Returns"].std()
    
linear_regression_df = ticker_df
linear_regression_df.reset_index(inplace=True)
linear_regression_df.dropna(inplace=True)
st.dataframe(linear_regression_df)
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
    
chart = make_subplots(specs=[[{"secondary_y" : True}]])
chart.add_trace(go.Scatter(x=linear_regression_df["Date"], y=linear_regression_df["Price"], name="Price", line_color="black"), secondary_y=True,)
#chart.add_trace(go.Scatter(x=linear_regression_df["Date"], y=linear_regression_df["Cumulative Returns"], line_color="white", showlegend=False, hoverinfo='none'), secondary_y=True,)
chart.add_trace(go.Scatter(x=linear_regression_df["Date"], y=fittedline, name="Prediction", line_color="lightslategray", hoverinfo='none'), secondary_y=False,)
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


risk_free_rate = .025 # necessary for Sortino/Sharpe Ratio calculations

# Function to display summary statistics and financial ratios
def get_token_statistics(ticker_df, start, end):

    # Calculates average daily returns and cumulative returns of the asset
    daily_returns = ticker_df["Daily Returns"]
    cumulative_returns = ticker_df["Cumulative Returns"]
    total_return = ticker_df["Cumulative Returns"].iloc[-1]
    peak = ticker_df["Cumulative Returns"].expanding(min_periods=1).max()
    ath = peak.max()

    # Calculates annualized returns / standard deviation, the variance, and max drawdown
    standard_deviation = ticker_df["Daily Returns"].std() * np.sqrt(365)
    max_drawdown = (ticker_df["Cumulative Returns"]/peak-1).min()
    negative_standard_deviation = ticker_df["Daily Returns"][ticker_df["Daily Returns"]<0].std() * np.sqrt(366)

    # Calculates the Sharpe, Sortino, & Calmar Ratios. Negative Annualized Standard Deviation is used for Sortino Ratio
    sharpe_ratio = (total_return - risk_free_rate) / standard_deviation
    sortino_ratio = (total_return - risk_free_rate) / negative_standard_deviation
    calmar_ratio = (total_return - risk_free_rate) / (abs(max_drawdown))

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
    token_statistics.columns = ["Calmar Ratio", "Sortino Ratio", "Sharpe Ratio", "Max Drawdown", "Peak", "Annual Volatity", "Return"]
    token_statistics = token_statistics.round(2)

    return token_statistics

bar_chart = get_token_statistics(ticker_df, start_date, end_date).hvplot.bar(color="black", hover_color="green", rot=45)

st.markdown("""**Financial Ratios & Statistics**""")
st.markdown("""Risk/return metrics and performance ratios over selected time period.""")

st.bokeh_chart(hv.render(bar_chart, backend="bokeh"))


# Calculating correlations with SPY, QQQ, ARKK over time period selected by user

tickers = ["SPY", "QQQ", "ARKK"]

indices_df = alpaca.get_bars(tickers, timeframe, start = start_date, end = end_date ).df

spy_df = indices_df[indices_df['symbol']=='SPY'].drop('symbol', axis=1)
spy_df = pd.DataFrame(spy_df["close"])
spy_df = spy_df.rename(columns={"close": "SPY"})

qqq_df = indices_df[indices_df['symbol']=='QQQ'].drop('symbol', axis=1)
qqq_df = pd.DataFrame(qqq_df["close"])
qqq_df = qqq_df.rename(columns={"close": "QQQ"})

arkk_df = indices_df[indices_df['symbol']=='ARKK'].drop('symbol', axis=1)
arkk_df = pd.DataFrame(arkk_df["close"])
arkk_df = arkk_df.rename(columns={"close": "ARKK"})

stock_prices = pd.concat([spy_df, qqq_df , arkk_df],axis="columns", join="inner")
stock_prices.reset_index(inplace=True)
stock_prices = stock_prices.rename(columns={"timestamp": "Date"})
stock_prices["Date"] = stock_prices["Date"].dt.strftime('%Y-%m-%d')
stock_prices["Date"] = pd.to_datetime(stock_prices["Date"], infer_datetime_format=True)
stock_prices = stock_prices.set_index("Date")

daily_returns = stock_prices.pct_change().dropna()
cumulative_returns = (1 + daily_returns).cumprod()

spy_correlation = ticker_df["Cumulative Returns"].corr(cumulative_returns["SPY"]) #* ticker_df["Cumulative Returns"].corr(cumulative_returns["SPY"])
#spy_correlation = spy_correlation.round(2)
qqq_correlation = ticker_df["Cumulative Returns"].corr(cumulative_returns["QQQ"]) #* ticker_df["Cumulative Returns"].corr(cumulative_returns["QQQ"])
#qqq_correlation = qqq_correlation.round(2)
arkk_correlation = ticker_df["Cumulative Returns"].corr(cumulative_returns["ARKK"]) #* ticker_df["Cumulative Returns"].corr(cumulative_returns["ARKK"])
#arkk_correlation = arkk_correlation.round(2)

st.sidebar.header('Stock Market Correlation')
st.sidebar.caption("Correlation with market indices over time period.")
#col1, col2, col3 = st.columns(3) # code to move indice correlation into main body of application
st.sidebar.metric("S&P 500 (SPY)", spy_correlation, delta_color="off")
st.sidebar.metric("NASDAQ (QQQ)", qqq_correlation, delta_color="off")
st.sidebar.metric("Ark Innovation Fund (ARKK)", arkk_correlation, delta_color="off")
