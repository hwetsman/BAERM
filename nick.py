# (c)2020-2023 Nick Phraudsta (btconometrics/codeorange)
import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime as dt
from dateutil.relativedelta import relativedelta
from scipy.stats import norm
from scipy import stats
import seaborn as sns
import matplotlib.widgets as widgets
from matplotlib.widgets import Button
from datetime import datetime, timedelta
from math import exp, cos, log
from scipy import stats

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st. set_page_config(layout="wide")
st.header('(c)2020-2023 Nick Phraudsta (btconometrics/codeorange)')
# create end date of projection
max_projection = dt.datetime.now() + relativedelta(years=20)
proj_date = st.sidebar.date_input('Projection to date', dt.datetime.now(), max_value=max_projection,
                                  min_value=dt.datetime.now())
time_delta = proj_date - pd.to_datetime(dt.datetime.now()).date()
days_to_add = time_delta.days

# Import data
url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
df = pd.read_csv(url)
# df.to_csv('3_22_23_data.csv', index=False)

# Fix date formats and tsset the data
df['date'] = pd.to_datetime(df['time'], format='%Y-%m-%d')
df.drop('time', axis=1, inplace=True)

# Extend the range
extend_df = pd.DataFrame({'date': pd.date_range(
    df['date'].max() + pd.Timedelta(days=1), periods=days_to_add, freq='D')})
df = pd.concat([df,extend_df]).reset_index(drop=True)
# df = df.append(pd.DataFrame({'date': pd.date_range(
#     df['date'].max() + pd.Timedelta(days=1), periods=days_to_add, freq='D')})).reset_index(drop=True)

# Replace missing blkcnt values
df.loc[df['BlkCnt'].isnull(), 'BlkCnt'] = 6 * 24

# Generate sum_blocks and hp_blocks
df['sum_blocks'] = df['BlkCnt'].cumsum()
df['hp_blocks'] = df['sum_blocks'] % 210001

# Generate hindicator and epoch
df['hindicator'] = (df['hp_blocks'] < 200) & (df['hp_blocks'].shift(1) > 209000)
df['epoch'] = df['hindicator'].cumsum()

# Generate reward and daily_reward
df['reward'] = 50 / (2 ** df['epoch'].astype(float))
df.loc[df['epoch'] >= 33, 'reward'] = 0
df['daily_reward'] = df['BlkCnt'] * df['reward']

# Generate tsupply
df['tsupply'] = df['daily_reward'].cumsum()

# Generate logprice
df['logprice'] = np.log(df['PriceUSD'])

# Calculate phaseplus variable
start_date = pd.to_datetime('2008-10-31')
df['days_since_start'] = (df['date'] - start_date).dt.days
df['phaseplus'] = df['reward'] - (df['epoch'] + 1) ** 2

# Drop rows with date < 2010-07-18
df = df[df['date'] >= dt.datetime.strptime('2010-07-18', '%Y-%m-%d')].reset_index(drop=True)
# st.write(df)

# get dates of halvings
halvings = df[df.hindicator == True].date.tolist()

# Run the regression
mask = df['epoch'] < 2
reg_X = df.loc[mask, ['logprice', 'phaseplus']].shift(1).iloc[1:]
reg_y = df.loc[mask, 'logprice'].iloc[1:]
reg_X = sm.add_constant(reg_X)
ols = sm.OLS(reg_y, reg_X).fit()
coefs = ols.params.values


# Step 1: Calculate AR + phase OLS
start_date = pd.to_datetime('2010-07-30')
mask = df['date'] > start_date
indices = df[mask].index

# Initialize YHAT with 0
df['YHAT'] = df['logprice']

# Calculate YHAT one row at a time
for i in range(indices[0] + 1, indices[-1] + 1):
    df.at[i, 'YHAT'] = coefs[0] + coefs[1] * df.at[i - 1, 'YHAT'] + coefs[2] * df.at[i, 'phaseplus']

# Step 2: Calculate decayfunc
n = df.index.to_numpy()
df['decayfunc'] = 3 * np.exp(-0.0004 * n) * np.cos(0.005 * n - 1)

# Step 3: Calculate prediction from Step 1 (already in YHAT column)

# Step 4: Add decay to Step 3
df['YHAT'] = df['YHAT'] + df['decayfunc']

# Step 5: Exponentiate
df['eYHAT'] = np.exp(df['YHAT'])


# Define function to format y-axis ticks as dollars
def format_dollars(y, pos, is_minor=False):
    return f'${y:.0f}'


# Plot the results
date_20190101 = dt.datetime.strptime('2011-01-01', '%Y-%m-%d')
# date_20240101 = dt.datetime.strptime('2028-01-01', '%Y-%m-%d')
date_20240101 = proj_date.strftime('%m-%d-%Y')
plot_df = df[(df['date'] > date_20190101) & (df['date'] < date_20240101)]

# calculate today's baerm and display
today_baerm = plot_df[plot_df['date'] <= dt.datetime.now()].eYHAT.tolist()[-1]
st.sidebar.write(f"Today's model value is ${round(today_baerm,2)}")


# Calculate residuals
# residuals = plot_df['logprice'] - plot_df['YHAT']


# prepare plotly graph
fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(
    f'BTC/USD Exchange Rate and Model from 2011-01-01 to {proj_date}',
    'Residuals vs Fitted Values',
    'Histogram of Residuals and Standard Normal Distribution'))

# Plot the first trace on the first subplot
fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['PriceUSD'], name='PriceUSD'), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['eYHAT'], name='BAERM'), row=1, col=1)
training_df = plot_df[plot_df.epoch < 2]
fig.add_trace(go.Scatter(x=training_df['date'], y=training_df['PriceUSD'],
              name='Model Training'), row=1, col=1)
for date in halvings:
    fig.add_vline(x=date, line_width=1, line_dash="dash", line_color="green")
fig.update_yaxes(type='log', row=1, col=1)
fig.update_yaxes(tickformat='$,.0f', row=1, col=1)
fig.update_layout(showlegend=True, width=1100, height=700)


st.plotly_chart(fig)
#
#
#
#
# Uncomment below to engage DCA model
# st.sidebar.write('________')
# st.sidebar.write('Buy/Hold model calculations begin 7/9/16')
# # calculate buy/hold
# start_date = '2016-03-23'
# end_date = datetime.now().date()
#
# # Generate the date range
# date_range = pd.date_range(start_date, end_date, freq='D')
#
# # Create the DataFrame
# hold_df = pd.DataFrame(date_range, columns=['date'])
# hold_df['upslope'] = 0
# build_dates = {1: ['3/23/16', '1/28/18'], 2: ['8/19/19', '8/16/21'], 3: ['1/13/23', datetime.now()]}
# for i in build_dates:
#     start_date = build_dates[i][0]
#     end_date = build_dates[i][1]
#     hold_df.loc[(hold_df['date'] >= start_date) & (hold_df['date'] <= end_date), 'upslope'] = 1
# min_date = hold_df.date.min()
# max_date = hold_df.date.max()
# daily = st.sidebar.slider('$$ per day DCA', min_value=1, max_value=100, value=5)
#
# # add prices
# price_df = plot_df[['date', 'eYHAT', 'PriceUSD']]
# hold_df = pd.merge(hold_df, price_df, on='date', how='left')
# # dca
# hold_df['daily_dca'] = daily/hold_df.PriceUSD
# st.sidebar.write(
#     f"Total BTC buying daily is {round(hold_df.daily_dca.sum(),2)} at a cost of ${round(hold_df.shape[0]*daily,2)}")
# # dca with model
# st.sidebar.write('Only buy when price<model')
# up_only = st.sidebar.radio('Buy only on upslope', ['Yes', 'No'])
# for i, r in hold_df.iterrows():
#     if i == 0:
#         hold_df.loc[i, 'balance'] = daily
#     else:
#         hold_df.loc[i, 'balance'] = daily+hold_df.loc[i-1, 'end_balance']
#     # calculate if deploy
#     if up_only == 'Yes':
#         if hold_df.loc[i, 'eYHAT'] > hold_df.loc[i, 'PriceUSD'] and hold_df.loc[i, 'upslope'] == 1:
#             hold_df.loc[i, 'deploy'] = 1
#         else:
#             hold_df.loc[i, 'deploy'] = 0
#     else:
#         if hold_df.loc[i, 'eYHAT'] > hold_df.loc[i, 'PriceUSD']:
#             hold_df.loc[i, 'deploy'] = 1
#         else:
#             hold_df.loc[i, 'deploy'] = 0
#
#     # calculate $ to deploy
#     if hold_df.loc[i, 'deploy'] == 1:
#         hold_df.loc[i, '$deploy'] = round(max(
#             daily, hold_df.loc[i, 'balance']*((hold_df.loc[i, 'eYHAT']-hold_df.loc[i, 'PriceUSD'])/hold_df.loc[i, 'eYHAT'])), 2)
#     else:
#         hold_df.loc[i, '$deploy'] = round(0, 2)
#     hold_df.loc[i, 'end_balance'] = hold_df.loc[i, 'balance']-hold_df.loc[i, '$deploy']
#     hold_df.loc[i, 'btc_in'] = hold_df.loc[i, '$deploy']/hold_df.loc[i, 'PriceUSD']
#     if i == 0:
#         hold_df.loc[i, 'btc_bal'] = hold_df.loc[i, 'btc_in']
#     else:
#         hold_df.loc[i, 'btc_bal'] = hold_df.loc[i-1, 'btc_bal']+hold_df.loc[i, 'btc_in']
#     hold_df.loc[i, 'worth'] = hold_df.loc[i, 'btc_bal']*hold_df.loc[i, 'PriceUSD']
#     hold_df['total_spent'] = hold_df['$deploy'].cumsum()
#     hold_df['loss'] = hold_df['worth'] < hold_df['total_spent']
# total_bought = hold_df.btc_bal.max()
# total_spent = hold_df["$deploy"].sum()
# st.sidebar.write(f'Purchased {round(total_bought,2)}')
# st.sidebar.write(f'For ${round(total_spent,2)}')
# st.sidebar.write(f'Average: ${round(total_spent/total_bought,2)}')
# st.sidebar.write(f'Days of loss: {hold_df.loss.sum()}')
# st.write(hold_df)
# create daily DCA
