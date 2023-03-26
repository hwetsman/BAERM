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
proj_date = st.sidebar.date_input('Projection to date', dt.datetime.now(), max_value=max_projection)
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
# df = df.append(pd.DataFrame({'date': pd.date_range(
#     df['date'].max() + pd.Timedelta(days=1), periods=2000, freq='D')})).reset_index(drop=True)
df = df.append(pd.DataFrame({'date': pd.date_range(
    df['date'].max() + pd.Timedelta(days=1), periods=days_to_add, freq='D')})).reset_index(drop=True)

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
st.write(plot_df)
dt.datetime.now()
short_list = plot_df[plot_df['date'] <= dt.datetime.now()].PriceUSD.tolist()[-2:]

st.write(bool(short_list[-1]))
if short_list[-1] == pd.nonull():
    today_baerm = short_list[-1]
else:
    today_baerm = short_list[-2]
today_baerm = plot_df.PriceUSD.tolist()[-1]


st.sidebar.write(f"Today's model value is ${round(today_baerm,2)}")
# Calculate residuals
residuals = plot_df['logprice'] - plot_df['YHAT']


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
