# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 18:15:52 2021

@author: andre
"""

from plotting_functions import series_plot, pred_truth_vis
from beta_functions import beta_generator
from models import model_generator
from rolling_functions import Rolling_LR_OneD
from trading_strats import MeanReversion
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
#%% Data load in

# Create list of currency names
currency_list = ['CME Australian Dollar AUD',
                 'CME Mexican Peso',
                 'CME Canadian Dollar CAD']

# Initialise empty dataframes with full indexing
currency_dict = {currency:  pd.DataFrame([], index = pd.bdate_range(start = '1/1/1980', end = '7/31/2020')) for currency in currency_list}

# Loop through each currency and load the data into the waiting dataframne
for currency in currency_list:
    
    current_df = pd.read_csv('Data/Continuous Futures Series/{}.csv'.format(currency), index_col = 0, skiprows = 0, skipfooter = 1, header = 1, engine = 'python')
    current_df.index = pd.to_datetime(current_df.index)
    
    currency_dict[currency] = currency_dict[currency].join(current_df)
    
# Create empty dataframe of currency close prices to be filled in
full_close_df = pd.DataFrame([], index = currency_dict['CME Canadian Dollar CAD'].index)

# Loop through each currency dataframe and pull its close price data
for currency in currency_list:
    
    full_close_df[currency] = currency_dict[currency]['Close']
#%%
# Create list of currency names
commodity_list = pd.read_csv('Data/Commodities_List_Updated.csv')['NAME']

# Initialise empty dataframes with hourly indexing
commodity_dict = {commodity:  pd.DataFrame([], index = pd.bdate_range(start = '1/1/1980', end = '7/31/2020')) for commodity in commodity_list}

# Loop through each currency and load the data into the waiting dataframne
for commodity in commodity_list:
    
    current_df = pd.read_csv('Data/Continuous Futures Series/{}.csv'.format(commodity), index_col = 0, skiprows = 0, skipfooter = 1, header = 1, engine = 'python')
    current_df.index = pd.to_datetime(current_df.index)
    
    commodity_dict[commodity] = commodity_dict[commodity].join(current_df)
#%%
# Loop through each currency dataframe and pull its close price data
for commodity in commodity_list:
    
    full_close_df[commodity] = commodity_dict[commodity]['Close']
 
#%% Cleaning full data
full_close_df_copy = full_close_df.copy()
full_close_df_copy['Avg'] = full_close_df[['CME Canadian Dollar CAD',
                                           'CME Australian Dollar AUD',
                                           'CME Mexican Peso']].copy().mean(axis=1)

full_close_df_copy = full_close_df_copy.drop(['CME Australian Dollar AUD',
                                    'CME Mexican Peso',
                                    'CME Canadian Dollar CAD',
                                    'ICE Heating Oil', 
                                    'ICE WTI Crude Oil',
                                    'CME Class III Milk'], axis=1)

full_close_df_copy = full_close_df_copy.fillna(method='ffill')


full_close_df_unstd = full_close_df_copy.dropna()
# # Truncating currency data frame to first AUD date
# mask = currency_close_df['CME Mexican Peso'].isnull()

# currency_close_df = currency_close_df.loc[np.logical_not(mask)]


#%%

# Normalised return dataframe
norm_commodity_returns_df = pd.DataFrame([]).reindex_like(full_close_df_unstd)

# Fill dataframe
for commodity in full_close_df_unstd:
    
    commodity_returns_series = np.log(full_close_df_unstd[commodity] / full_close_df_unstd[commodity].shift(1))
    
    commodity_std_series = commodity_returns_series.rolling(15, min_periods=1).std()
    
    norm_commodity_returns_df[commodity] = commodity_returns_series / commodity_std_series
norm_commodity_returns_df.dropna(inplace=True)

#%%
trade_commod = norm_commodity_returns_df.iloc[10:,:-1]
trade_cur = norm_commodity_returns_df.iloc[10:,-1]

trade_prices = full_close_df_copy.loc[trade_commod.index]
# trade_commod = norm_commodity_returns.dropna()
# trade_cur = norm_cur_avg.loc[trade_commod.index]
# trade_prices = commodity_close_df[trade_commod.columns].loc[trade_commod.index]


#%%
from trading_strats_real_data import MeanReversion
mean_rev = MeanReversion()
test = mean_rev.back_test(trade_cur,
                          trade_commod,
                          trade_prices,
                          chunk_size=25,
                          lookback=50,
                          pos_ratio=1/3)
#%%
# performance_graphs=[]
# performance_graphs.append(mean_rev.LR_performance)
# performance_graphs.append(mean_rev.LSTM_performance)
performance_graphs[1] = mean_rev.LR_performance

#%%
plt.figure(figsize=(20,10))
plt.title('Performance of Strategies on Real Commodities Data', fontsize=27)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.xlabel('Date', fontsize=22)
plt.ylabel('Profit/Loss (Dollars)', fontsize=22)

perf_labels=['Rolling Linear Regression',
             'Random Signals']

for i, performance in enumerate(performance_graphs):
    plt.plot(performance.iloc[3900:], label=perf_labels[i], lw=4)
    
# plt.plot([chunk[-1] for chunk in PL_chunk_list])
plt.legend(fontsize=22)
#%%

# Grab us treasury yield price
# treas_bond = pd.read_csv('Data/Continuous Futures Series/CBOT 10-year US Treasury Note.csv'.format(currency), index_col = 0, skiprows = 0, skipfooter = 1, header = 1, engine = 'python')['Close']
treas_bond_df = pd.DataFrame([], index=mean_rev.LR_performance.index)

treas_bond_series = pd.read_csv('Data/Continuous Futures Series/Treasury Yields.csv'.format(currency), index_col = 0, engine = 'python')['RF'][:-2]

treas_bond_series.index = pd.to_datetime(treas_bond_series.index)
treas_bond_df['10-Year'] = treas_bond_series
treas_bond_df = treas_bond_df.fillna(method='ffill')

def sharpe_ratio(PnL_curve):
    
    # Split PL curve into trading chunks
    PL_chunk_list = np.array_split(PnL_curve, np.floor(len(PnL_curve) / 25))
    PL_chunk_list = PL_chunk_list[160:]
    #Set Burn-In
    # PL_chunk_list = PL_chunk_list[10:]
    
    # Compute balance list
    returns_df = pd.DataFrame([], index=[chunk.index[-1] for chunk in PL_chunk_list])
    returns_df['Balance'] = [chunk[-1] for chunk in PL_chunk_list]
    
    # Compute Returns
    returns_df['Returns'] = returns_df['Balance'] / returns_df['Balance'].shift(1)
    
    # Compute Risk Adjusted returns
    returns_df['Treasury Yield'] = treas_bond_df['10-Year']
    returns_df['Risk Adjusted Returns'] = returns_df['Returns'] - returns_df['Treasury Yield']
    
    sharpe = returns_df['Risk Adjusted Returns'].mean()
    sigma = returns_df['Risk Adjusted Returns'].std()
    # sigma = returns_df['Risk Adjusted Returns'].std()
    
    return sharpe, sigma, returns_df

test = sharpe_ratio(mean_rev.LR_performance)


print(test[0] / test[1])
# print(test[0])

