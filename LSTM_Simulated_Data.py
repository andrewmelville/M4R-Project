# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 13:44:58 2021

@author: andre
"""

import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

from models import model_generator
from plotting_functions import series_plot, signal_plot


import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

#%% Simulating Data
from models import model_generator

# Define a generative model to simulate 10000 days of data for 1 currency basket and 30 commodities
model = model_generator()
sim_currency = model.linear_model(num_obs=10000, 
                                       num_covariates=2, 
                                       beta_type='cor_bb', 
                                       beta_sigma=0.0035, 
                                       noise=0.0001,
                                       t=0.6)
sim_betas = model.params
sim_commods = model.covariates()['Noisy']
# model.model_plot()

next_day_returns = pd.DataFrame(sim_commods[1])
def noise_vis(returns, noise):
    
    plt.figure(figsize=(20,10))
    plt.title('Visualisation of the Noise Against the True Commodity Returns')
    plt.xlabel('Index')
    plt.ylabel('Value')
    
    plt.plot(returns, label='True Commodity Returns') 
    plt.plot(noise, label='Noise')

    plt.legend()
    
noise_vis(model.covariates()['True'][1:], model.noise[1:])
# series_plot(next_day_returns.cumsum(),'')
# CBOT_OATS_df = pd.read_csv('Data/Continuous Futures Series/CBOT Rough Rice.csv',
#                            index_col = 0, skiprows = 0, skipfooter = 1, header = 1, engine = 'python')
# next_day_returns = pd.DataFrame(CBOT_OATS_df['Close'].diff(1))
#%% Train (70%) / Validate (20%) / Test (10%)
def makeXy(comm_df, cur_df, nb_timesteps):
    """
    Input: 
            ts: original time series
            nb_timesteps: number of time steps in the regressors
    Output: 
            X: 2-D array of regressors
            y: 1-D array of target 
    """
    # Split data into train/val/test sets
    n = len(comm_df)
    
    # Grab column names
    comm_col, cur_col = comm_df.columns[0], cur_df.columns[0]
    
    # Split full data into train, validation, and test sets
    comm_train_unscaled, cur_train_unscaled = pd.DataFrame(comm_df[0:int(n*0.7)]).reset_index(drop=True), pd.DataFrame(cur_df[0:int(n*0.7)]).reset_index(drop=True)
    comm_val_unscaled, cur_val_unscaled = pd.DataFrame(comm_df[int(n*0.7):int(n*0.9)]).reset_index(drop=True), pd.DataFrame(cur_df[int(n*0.7):int(n*0.9)]).reset_index(drop=True)
    comm_test_unscaled, cur_test_unscaled = pd.DataFrame(comm_df[int(n*0.9):]).reset_index(drop=True), pd.DataFrame(cur_df[int(n*0.9):]).reset_index(drop=True)
    
    # Standardise data with respect to the training data
    # (would expect data very close to a normal given the method of data simulation)
    # scaler = StandardScaler()
    # comm_scaler, cur_scaler = MinMaxScaler(), MinMaxScaler()
    # comm_scaler.fit(comm_train_unscaled)
    # cur_scaler.fit(cur_train_unscaled)
    
    # comm_train_scaled, cur_train_scaled = pd.DataFrame(comm_scaler.transform(comm_train_unscaled[comm_col].values.reshape(-1,1))), pd.DataFrame(cur_scaler.transform(cur_train_unscaled[cur_col].values.reshape(-1,1)))
    # comm_val_scaled, cur_val_scaled = pd.DataFrame(comm_scaler.transform(comm_val_unscaled[comm_col].values.reshape(-1,1))), pd.DataFrame(cur_scaler.transform(cur_val_unscaled[cur_col].values.reshape(-1,1)))
    # comm_test_scaled, cur_test_scaled = pd.DataFrame(comm_scaler.transform(comm_test_unscaled[comm_col].values.reshape(-1,1))), pd.DataFrame(cur_scaler.transform(cur_test_unscaled[cur_col].values.reshape(-1,1)))
    
    # Reshape data to be vectors of length nb_timesteps and labels
    train_X, train_y, val_X, val_y, test_X, test_y = [], [], [], [], [], []
    
    # Train
    for i in range(nb_timesteps, comm_train_unscaled[comm_col].shape[0]-1):
        train_X.append(np.array(cur_train_unscaled[cur_col].loc[i-nb_timesteps:i]))
        train_y.append(comm_train_unscaled[comm_col].loc[i-nb_timesteps:i])
    train_X = np.array(train_X, dtype=object)
    
    # Validation
    for i in range(nb_timesteps, comm_val_unscaled[comm_col].shape[0]-1):
        val_X.append(cur_val_unscaled[cur_col].loc[i-nb_timesteps:i])
        val_y.append(comm_val_unscaled[comm_col].loc[i-nb_timesteps:i])
    val_X = np.array(val_X, dtype=object)

    # Test
    for i in range(nb_timesteps, comm_test_unscaled[comm_col].shape[0]-1):
        test_X.append(cur_test_unscaled[cur_col].loc[i-nb_timesteps:i])
        test_y.append(comm_test_unscaled[comm_col].loc[i-nb_timesteps:i])
    test_X = np.array(test_X, dtype=object)
    
    # prepare data
    train_X = tf.convert_to_tensor(train_X, dtype='float64')
    train_y =  tf.convert_to_tensor(train_y, dtype='float64')
    val_X =  tf.convert_to_tensor(val_X, dtype='float64')
    val_y =  tf.convert_to_tensor(val_y, dtype='float64')
    test_X =  tf.convert_to_tensor(test_X, dtype='float64')
    test_y =  tf.convert_to_tensor(test_y, dtype='float64')
    
    return train_X, train_y, val_X, val_y, test_X, test_y
lookback = 120
X_train, y_train, X_val, y_val, X_test, y_test = makeXy(next_day_returns[1:], sim_currency[1:], lookback)
print('Shape of train arrays:', X_train.shape, y_train.shape)
#%%

import keras
from keras.layers import Dense, Input, Dropout
from keras.layers.recurrent import LSTM

from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

input_layer = Input(shape=(lookback+1,1), dtype='float32')

lstm_layer = LSTM(10, input_shape=(lookback+1,1),
                  return_sequences=True)(input_layer)

# dropout_layer = Dropout(0)(lstm_layer) 

# dense_layer = Dense(5, activation='tanh')(lstm_layer)

output_layer = Dense(1, activation='linear')(lstm_layer)
#%%
opt = tf.keras.optimizers.Adam()

ts_model = Model(inputs=input_layer,
                 outputs=output_layer)

ts_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                 optimizer=opt)

ts_model.summary()

save_weights_at = os.path.join('keras_models', 'Sim_Data_LSTM_weights')
# save_weights_at = os.path.join('keras_models', 'Sim_Data_LSTM_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
# save_weights_at = os.path.join('keras_models', 'PRSA_data_Air_Pressure_LSTM_weights.01.hdf5')

save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min',
                            period=1)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
ts_model.fit(x=X_train, y=y_train, 
             batch_size=32, epochs=20,
             verbose=True, callbacks=[callback], validation_data=(X_val, y_val),
             shuffle=False)

   # In[ ]:
    
# best_model = load_model(os.path.join('keras_models', 'Sim_Data_LSTM_weights'))
# best_model = load_model(os.path.join('keras_models', 'Sim_Data_LSTM_weights.01-0.9380.hdf5'), custom_objects={'profit_loss':profit_loss})
# best_model = load_model(os.path.join('keras_models', 'PRSA_data_Air_Pressure_LSTM_weights.02-0.0128.hdf5'))
preds = ts_model.predict(X_train)
# pred_PRES = scaler.inverse_transform(preds)
pred_PRES = np.squeeze(preds)

y_train_hat = np.array([pred[-1] for pred in pred_PRES])
y_test_hat = np.array([pred[-1] for pred in np.squeeze(ts_model.predict(X_test))])
# y_train_series = np.exp([i for i in next_day_returns[1][lookback+1:int(0.7*10000)]]).cumprod()

# def lstm_pred_price_vis(y_pred, y_true):
    
#     plt.figure(figsize=(20,10))    
#     plt.title('Visualisation of LSTM Predictions')
#     plt.xlabel('Index')
#     plt.ylabel('Value')
    
#     plt.plot(np.exp([i for i in y_true]).cumprod(), label='True Series')
#     plt.plot(np.exp(y_pred)[1:] * np.exp([i for i in y_true]).cumprod()[:-1], label='Predicted Series')
    
#     plt.legend()
    
# lstm_pred_price_vis(y_train_hat, next_day_returns[1][501:7000])

##%% 
from plotting_functions import pred_truth_vis, return_series_vis    
pred_truth_vis(model.covariates()['True'][lookback+1:6999][1].reset_index(drop=True), y_train_hat)
# %%
# return_series_vis(model.covariates()['True'][lookback+1:6999].reset_index(drop=True), y_train_hat)
return_series_vis(model.covariates()['True'][lookback+9000:-1][1].reset_index(drop=True), y_test_hat)
pred_truth_vis(model.covariates()['True'][lookback+9000:-1][1].reset_index(drop=True), y_test_hat)

#%%
second_commods = pd.DataFrame(sim_commods[2])
X_train, y_train, X_val, y_val, X_test, y_test = makeXy(second_commods[1:], sim_currency[1:], lookback)
print('Shape of train arrays:', X_train.shape, y_train.shape)


preds = ts_model.predict(X_train)
# pred_PRES = scaler.inverse_transform(preds)
pred_PRES = np.squeeze(preds)

y_train_hat = np.array([pred[-1] for pred in pred_PRES])
y_test_hat = np.array([pred[-1] for pred in np.squeeze(ts_model.predict(X_test))])
# y_train_series = np.exp([i for i in next_day_returns[1][lookback+1:int(0.7*10000)]]).cumprod()

# def lstm_pred_price_vis(y_pred, y_true):
    
#     plt.figure(figsize=(20,10))    
#     plt.title('Visualisation of LSTM Predictions')
#     plt.xlabel('Index')
#     plt.ylabel('Value')
    
#     plt.plot(np.exp([i for i in y_true]).cumprod(), label='True Series')
#     plt.plot(np.exp(y_pred)[1:] * np.exp([i for i in y_true]).cumprod()[:-1], label='Predicted Series')
    
#     plt.legend()
    
# lstm_pred_price_vis(y_train_hat, next_day_returns[1][501:7000])

##%% 
from plotting_functions import pred_truth_vis, return_series_vis    
pred_truth_vis(model.covariates()['True'][lookback+1:6999][2].reset_index(drop=True), y_train_hat)

return_series_vis(model.covariates()['True'][lookback+9000:-1][2].reset_index(drop=True), y_test_hat)
pred_truth_vis(model.covariates()['True'][lookback+9000:-1][2].reset_index(drop=True), y_test_hat)