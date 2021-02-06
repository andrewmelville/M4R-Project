# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 17:28:34 2021

@author: andre

This is the first test of LSTM on data simulated using the genertive methods outlined in models.py.
The workflow and many functions in this file are taken from the tensorflow RNN tutorial 
which can be accessed from: https://www.tensorflow.org/tutorials/structured_data/time_series

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
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

from models import model_generator
from plotting_functions import series_plot

#%% Simulating Data


# Define a generative model to simulate 10000 days of data for 1 currency basket and 30 commodities
model = model_generator()
sim_currency = model.linear_model(num_obs=10000, num_covariates=30, beta_type='brownian', noise=0.001)
sim_betas = model.params
sim_commods = model.covariates()['Noisy']
model.model_plot()

next_day_returns = sim_commods.shift(1)

#%% Train (70%) / Validate (20%) / Test (10%)

## This workflow should be turned into a function ASAP


column_indices = {name: i for i, name in enumerate(sim_commods.columns)}

n = len(sim_commods)
train_df = pd.DataFrame(sim_commods[1][0:int(n*0.7)])
val_df = pd.DataFrame(sim_commods[1][int(n*0.7):int(n*0.9)])
test_df = pd.DataFrame(sim_commods[1][int(n*0.9):])

num_features = sim_commods.shape[1]

# Standardise data (would expect data very close to a normal given the method of data simulation)
scaler = StandardScaler()
scaler.fit(train_df)

# Define data to be scaled current day returns concatenated with unstandardised next-day returns
train_std = pd.DataFrame(scaler.transform(train_df))
val_std =  pd.DataFrame(scaler.transform(val_df))
test_std =  pd.DataFrame(scaler.transform(test_df))

train_std['Next Day Returns'] = pd.DataFrame(next_day_returns[1][0:int(n*0.7)])
val_std['Next Day Returns'] = pd.DataFrame(next_day_returns[1][int(n*0.7):int(n*0.9)])
test_std['Next Day Returns'] = pd.DataFrame(next_day_returns[1][int(n*0.9):])

# Violin plot to visualise the distribution of standardised data
plt.figure(figsize=(12, 6))
ax = sns.violinplot(data=train_std)
_ = ax.set_xticklabels(train_std.keys(), rotation=90)

#%% Defining window generator class

class WindowGenerator():
    
    
    def __init__(self, input_width, label_width, shift,
               train_df=train_std, val_df=val_std, test_df=test_std,
               label_columns='Next Day Returns'):
      
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}
    
        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
    
        self.total_window_size = input_width + shift
    
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
    
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    
    
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels
    

    
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)


    @property
    def val(self):
        return self.make_dataset(self.val_df)


    @property
    def test(self):
        return self.make_dataset(self.test_df)


    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        
        return result


    def __repr__(self):
        return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
#%%
single_step_window = WindowGenerator(
    input_width=60, label_width=1, shift=1,
    label_columns=['Next Day Returns'])
single_step_window

for example_inputs, example_labels in single_step_window.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')
  
  
  
#%% LSTM Model Architecture

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])
  

lstm_model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
#%% Defining a custom loss fuction and a method for compiling and training a model


def profit_loss(y_true, y_predicted):
    
    ## This function calculates the profit/loss incurred by the decision made by the LSTM model
    ## with outputs in (-1,1). This relies on the labels in the training data to be the difference
    ## in price between the current day and the day chosen to sell.
    ## e.g. investment horizon of 1 day would require labels = (price on day t) - (price on day t+1).
    
    prof_func = - y_predicted * y_true
    
    return prof_func

MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')


    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])


    history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
    return history

#%% Fitting naive model

history = compile_and_fit(lstm_model, single_step_window)