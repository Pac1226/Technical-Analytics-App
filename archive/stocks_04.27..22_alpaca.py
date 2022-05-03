"""Crypto Linear Regression App by Peter Lieberman"""

# Loads basic libraries and dependencies
from PIL import Image
import pandas as pd
import numpy as np
import datetime as dt
import os
import financialanalysis as fa
import streamlit as st
import quandl
import matplotlib.pyplot as plt
import hvplot.pandas
import holoviews as hv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
hv.extension('bokeh')

# API keys & Streamlit secrerts
quandl_apy_key = "PPbpavuDMKGWYMGS8rrB"
quandl.ApiConfig.api_key = quandl_apy_key

# Create the Alpaca API object

# Application Page Configuration: Headers & Sidebar #

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header('Stock Linear Regression')

st.markdown("""
This app pulls price data for any stock and runs a series of models 
to assess past performance and predict future price trends!
* **Python libraries:** pandas, numpy, os, streamlit, messari.messari, financialanalysis, scikit-learn
* **Data Source::** [Alpaca API](https://alpaca.markets/data)
* **Historical Data:** only 5-6 years of historical data
* **Models:** linear regression and risk/return ratios
* **Charts:** all charts are interactive and can be saved as images
""")


# Sidebar widgets: cryptocurrency and time period selection
st.sidebar.header('User Input Features')
st.sidebar.caption('Input any ticker symbol and select timeframe. There is only 5 years of historical data.')


# Widget to select stock ticker
selected_asset = st.sidebar.text_input('Stock Ticker', value="AAPL")

# Widget to select timeperiod
start_date = "2022-01-31" #st.sidebar.date_input("Start Date", value = dt.date(2022, 1, 31), min_value = dt.date(1950, 1, 31)).strftime('%Y-%m-%d')
end_date = "2022-04-27" # st.sidebar.date_input("End Date").strftime('%Y-%m-%d')
st.write(start_date)
st.write(end_date)
# Analytics Section 1: Function for Linear Regressions #

st.markdown("""**Linear Regression Channel**""")
st.markdown("""Regression line of time and price with standard deviation channels and Simple Moving Averages.""")
st.caption(selected_asset)

data = quandl.get_table('WIKI/PRICES', ticker = "MSFT", 
                        qopts = { 'columns': ['ticker', 'date', 'close'] }, 
                        date = { 'gte': f'{start_date}', 'lte': f'{end_date}' }, 
                        paginate=True)

st.dataframe(data)

data = data.drop(columns={"ticker"})
data = data.set_index("date")
data = data.rename(columns={"close": "Price"})
data["Daily Returns"] = data["Price"].pct_change().dropna()
data["Cumulative Returns"] = (1 + data["Daily Returns"]).cumprod()
data.dropna()

st.dataframe(data)



sma200 = data["Price"].rolling(window=200).mean()
sma50 = data["Price"].rolling(window=50).mean()
    
std = data["Cumulative Returns"].std()
    
linear_regression_df = data
linear_regression_df.reset_index(inplace=True)
linear_regression_df.dropna(inplace=True)

# Utilizes financialanalysis (fa) module to build linear regression channels
X = linear_regression_df["date"].to_list() # converts Series to list
X = fa.datetimeToFloatyear(X) # for example, 2020-07-01 becomes 2020.49589041
X = np.array(X) # converts list to a numpy array
X = X[::,None] # converts row vector to column vector (just column vector is acceptable)
y = linear_regression_df["Cumulative Returns"] # get y data (relative price)
y = y.values # converts Series to numpy
y = y[::,None] # row vector to column vector (just column vector is acceptable)
 
slope, intercept, x, fittedline = fa.timeseriesLinearRegression(linear_regression_df["date"], linear_regression_df["Cumulative Returns"])

    # Trendlines for standard deviation parallel channels
fittedline_upper_1 = fittedline + std
fittedline_lower_1 = fittedline - std
fittedline_upper_2 = fittedline + (std*2)
fittedline_lower_2 = fittedline - (std*2)
    
chart = make_subplots(specs=[[{"secondary_y" : True}]])
chart.add_trace(go.Scatter(x=linear_regression_df["date"], y=linear_regression_df["Price"], name="Price", line_color="black"), secondary_y=True,)
#chart.add_trace(go.Scatter(x=linear_regression_df["Date"], y=linear_regression_df["Cumulative Returns"], line_color="white", showlegend=False, hoverinfo='none'), secondary_y=True,)
chart.add_trace(go.Scatter(x=linear_regression_df["date"], y=fittedline, name="Prediction", line_color="lightslategray", hoverinfo='none'), secondary_y=False,)
chart.add_trace(go.Scatter(x=linear_regression_df["date"], y=fittedline_lower_1, name="Standard Deviation", line_color="forestgreen", hoverinfo='none'), secondary_y=False,)
chart.add_trace(go.Scatter(x=linear_regression_df["date"], y=fittedline_upper_1, line_color="forestgreen", showlegend=False, hoverinfo='none'), secondary_y=False,)
chart.add_trace(go.Scatter(x=linear_regression_df["date"], y=fittedline_lower_2, name="2 Standard Deviations", line_color="rosybrown", hoverinfo='none'), secondary_y=False,)
chart.add_trace(go.Scatter(x=linear_regression_df["date"], y=fittedline_upper_2, name="2 Standard Deviations", line_color="rosybrown", showlegend=False, hoverinfo='none'), secondary_y=False,)
chart.add_trace(go.Scatter(x=linear_regression_df["date"], y=sma200, name="200-Day SMA", line_color="gray"), secondary_y=True,)
chart.add_trace(go.Scatter(x=linear_regression_df["date"], y=sma50, name="50-Day SMA", line_color="lightgray"), secondary_y=True,)

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
    annualized_return = ticker_df["Daily Returns"].mean() * 252
    peak = ticker_df["Cumulative Returns"].expanding(min_periods=1).max()
    ath = peak.max()

    # Calculates annualized returns / standard deviation, the variance, and max drawdown
    standard_deviation = ticker_df["Daily Returns"].std() * np.sqrt(252)
    max_drawdown = (ticker_df["Cumulative Returns"]/peak-1).min()
    negative_standard_deviation = ticker_df["Daily Returns"][ticker_df["Daily Returns"]<0].std() * np.sqrt(252)

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

bar_chart = get_token_statistics(ticker_df, start_date, end_date).hvplot.bar(color="black", hover_color="green", rot=45)

st.markdown("""**Financial Ratios & Statistics**""")
st.markdown("""Risk/return metrics and performance ratios over selected time period.""")
st.caption(selected_asset)
st.bokeh_chart(hv.render(bar_chart, backend="bokeh"))