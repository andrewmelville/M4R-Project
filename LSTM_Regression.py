# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 19:02:43 2021

@author: andre
"""

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
sim_currency = model.linear_model(num_obs=10000, num_covariates=30, beta_type='bm_std', noise=0.0002)
sim_betas = model.params
sim_commods = model.covariates()['Noisy']
model.model_plot()

next_day_returns = pd.DataFrame(sim_commods[1])
series_plot(next_day_returns.cumsum(),'')
# CBOT_OATS_df = pd.read_csv('Data/Continuous Futures Series/CBOT Rough Rice.csv',
#                            index_col = 0, skiprows = 0, skipfooter = 1, header = 1, engine = 'python')
# next_day_returns = pd.DataFrame(CBOT_OATS_df['Close'].diff(1))
#%% Train (70%) / Validate (20%) / Test (10%)
def makeXy(series_df, nb_timesteps):
    """
    Input: 
           ts: original time series
           nb_timesteps: number of time steps in the regressors
    Output: 
           X: 2-D array of regressors
           y: 1-D array of target 
    """
    # Split data into train/val/test sets
    n = len(series_df)
    
    # Grab column name
    col_name = series_df.columns[0]
    
    # Split full data into train, validation, and test sets
    train_unscaled = pd.DataFrame(series_df[0:int(n*0.7)]).reset_index(drop=True)
    val_unscaled = pd.DataFrame(series_df[int(n*0.7):int(n*0.9)]).reset_index(drop=True)
    test_unscaled = pd.DataFrame(series_df[int(n*0.9):]).reset_index(drop=True)
    
    # Standardise data with respect to the training data
    # (would expect data very close to a normal given the method of data simulation)
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    scaler.fit(train_unscaled)
    
    train_scaled = pd.DataFrame(scaler.transform(train_unscaled[col_name].values.reshape(-1,1)))
    val_scaled = pd.DataFrame(scaler.transform(val_unscaled[col_name].values.reshape(-1,1)))
    test_scaled = pd.DataFrame(scaler.transform(test_unscaled[col_name].values.reshape(-1,1)))
    
    # Reshape data to be vectors of length nb_timesteps and labels
    train_X, train_y = [], np.array([returns for returns in train_unscaled[col_name].shift(-1)])[nb_timesteps:-1]
    val_X, val_y = [], np.array([returns for returns in val_unscaled[col_name].shift(-1)])[nb_timesteps:-1]
    test_X, test_y = [], np.array([returns for returns in test_unscaled[col_name].shift(-1)])[nb_timesteps:-1]
    
    # Train
    for i in range(nb_timesteps, train_scaled[0].shape[0]-1):
        train_X.append(list(train_scaled[0].loc[i-nb_timesteps:i-1]))
    train_X = np.array(train_X)
    
    # Validation
    for i in range(nb_timesteps, val_scaled[0].shape[0]-1):
        val_X.append(list(val_scaled[0].loc[i-nb_timesteps:i-1]))
    val_X = np.array(val_X)
    
    # Test
    for i in range(nb_timesteps, test_scaled[0].shape[0]-1):
        test_X.append(list(test_scaled[0].loc[i-nb_timesteps:i-1]))
    test_X = np.array(test_X)

    # prepare data
    train_X = tf.convert_to_tensor(train_X, dtype="float32") 
    train_y =  tf.convert_to_tensor(train_y, dtype="float32")
    val_X =  tf.convert_to_tensor(val_X, dtype="float32")
    val_y =  tf.convert_to_tensor(val_y, dtype="float32")
    test_X =  tf.convert_to_tensor(test_X, dtype="float32")
    test_y =  tf.convert_to_tensor(test_y, dtype="float32")
    
    return train_X, train_y, val_X, val_y, test_X, test_y, scaler


        
lookback = 600
X_train, y_train, X_val, y_val, X_test, y_test, scaler = makeXy(next_day_returns.dropna(), lookback)
print('Shape of train arrays:', X_train.shape, y_train.shape)
#%%

# Now we define the neural network topology using the Keras Functional API. In this approach a layer can be declared as the input of the following layer at the time of defining the next layer. 
import keras
from keras.layers import Dense, Input, Dropout
from keras.layers.recurrent import LSTM
# from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
# from keras.layers import Bidirectional, RNN, LSTMCell

# Define input layer which has shape (None, 7) and of type float32. None indicates the number of instances in the mini-batch used for training.
input_layer = Input(shape=(lookback,1), dtype='float32')

# The LSTM layer is defined for seven timesteps
lstm_layer = LSTM(10, input_shape=(lookback,1), return_sequences=False)(input_layer)

# Use Dropout regularization - the parameter chosen here can in principle be optimized.
# dropout_layer = Dropout(0.8)(dense_layer2)

# Finally, the output layer gives prediction for the next day's air pressure. This should be a real number value, and so no activation function is used (i.e. linear activation).
# dense_layer = Dense(5, activation='tanh')(dropout_layer)
output_layer = Dense(1, activation='relu')(lstm_layer)

# #%% Defining custom loss function to be used in model training

# def profit_loss(y_true, y_pred):
#     profit = tf.math.multiply(y_true, -100*y_pred) 
#     return profit

# keras.losses.profit_loss = profit_loss
#%%
# loss_bce = keras.losses.BinaryCrossentropy()
loss_profit = profit_loss
opt = tf.keras.optimizers.Adam()

ts_model = Model(inputs=input_layer,
                 outputs=output_layer)
# ts_model.compile(loss=loss_profit, optimizer=opt)
ts_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                 optimizer=opt)

ts_model.summary()

save_weights_at = os.path.join('keras_models', 'Sim_Data_LSTM_weights')
# save_weights_at = os.path.join('keras_models', 'Sim_Data_LSTM_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
# save_weights_at = os.path.join('keras_models', 'PRSA_data_Air_Pressure_LSTM_weights.01.hdf5')

save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min',
                            period=1)
ts_model.fit(x=X_train, y=y_train, 
             batch_size=32, epochs=4,
             verbose=True, callbacks=[save_best], validation_data=(X_val, y_val),
             shuffle=False)

# In[ ]:
    
best_model = load_model(os.path.join('keras_models', 'Sim_Data_LSTM_weights'))
# best_model = load_model(os.path.join('keras_models', 'Sim_Data_LSTM_weights.01-0.9380.hdf5'), custom_objects={'profit_loss':profit_loss})
# best_model = load_model(os.path.join('keras_models', 'PRSA_data_Air_Pressure_LSTM_weights.02-0.0128.hdf5'))
preds = best_model.predict(X_train)
# pred_PRES = scaler.inverse_transform(preds)
pred_PRES = np.squeeze(pred_PRES)
pred_PRES = [-1 if np.argmax(prediction) == 0 else 1 for prediction in pred_PRES]
# In[ ]:
from sklearn.metrics import r2_score
r2 = r2_score(y_val, pred_PRES)
print('R-squared on validation set of the returns:', r2)
# In[ ]:
import matplotlib.pyplot as plt    

n = next_day_returns.shape[0]

signal_plot(next_day_returns, , lookback)
# plt.hist(pred_PRES, bins=100)
# def pred_plot(actual_y, pred_y):

#     plt.figure(figsize=(10, 10))
#     plt.plot(actual_y[2:], marker='*', color='r',lw=0.1)
#     plt.plot(pred_y[2:], marker='.', color='b', lw=0.1)
#     plt.legend(['Actual','Predicted'], loc=2)
#     plt.title('Actual vs Predicted Air Pressure')
#     plt.ylabel('Air Pressure')
#     plt.xlabel('Index')
# pred_plot(y_val, pred_PRES)