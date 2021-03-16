# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 12:57:24 2021

@author: andre
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from models import model_generator
from plotting_functions import series_plot, signal_plot


import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
import keras
from keras.layers import Dense, Input, Dropout
from keras.layers.recurrent import LSTM

from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
class LSTM_trainer:
    
    def __init__(self, lkback_vec, noise_vec):
        
        self.lkback_vec = lkback_vec
        self.noise_vec = noise_vec
        
        self.curr_df = pd.DataFrame([], index=range(10000), columns=[noise for noise in noise_vec])
        self.commod_df = pd.DataFrame([], index=range(10000), columns=[noise for noise in noise_vec])
        self.commod_df_clean = pd.DataFrame([], index=range(10000), columns=[noise for noise in noise_vec])
        
    def __call__(self):
        
        # Create df of data with different noise levels
        for noise in self.noise_vec:
            
            model = model_generator()
            self.curr_df[noise] = model.linear_model(num_obs=10000, num_covariates=1, beta_type='bm_std', noise=noise)
            self.commod_df[noise] = model.covariates()['Noisy']
            self.commod_df_clean[noise] = model.covariates()['True']
            
        # Create training data tensors from data
        ## X_train = self.param_training_dict[lkback][noise][0]
        ## y_train = self.param_training_dict[lkback][noise][1]
        ## X_val = self.param_training_dict[lkback][noise][2]
        ## y_val = self.param_training_dict[lkback][noise][3]
        ## X_test = self.param_training_dict[lkback][noise][4]
        ## y_test = self.param_training_dict[lkback][noise][5]
        self.param_training_dict = {lkback : {noise: [] for noise in self.noise_vec} for lkback in self.lkback_vec}
        
        for noise in self.noise_vec:
            for lkback in self.lkback_vec:
                self.param_training_dict[lkback][noise] = self.makeXy(self.commod_df[noise], self.curr_df[noise], lkback)
            
            
    def train(self):
        # Initialise plotting
        fig, axs = plt.subplots(len(self.noise_vec), len(self.lkback_vec), figsize=(20,20), sharey=True, sharex=True)
        fig.suptitle('LSTM Performance on Data with Varied Noise and Lookback')
        
        # Print progress statistics
        model_num = 1
        num_of_models = len(self.lkback_vec) * len(self.noise_vec)
        
        for i, lkback in enumerate(self.lkback_vec):
            for j, noise in enumerate(self.noise_vec):
                
                # Define model
                input_layer = Input(shape=(lkback+1,1), dtype='float32')
                lstm_layer = LSTM(1, input_shape=(lkback+1,1), return_sequences=True)(input_layer)
                output_layer = Dense(1, activation='linear')(lstm_layer)
        
                # Prepare for trainning
                opt = tf.keras.optimizers.Adam()
                ts_model = Model(inputs=input_layer,
                                 outputs=output_layer)
                ts_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                                 optimizer=opt)
                # ts_model.summary()
                save_weights_at = os.path.join('keras_models', 'Sim_Data_LSTM_weights')
                save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                                            save_best_only=True, save_weights_only=False, mode='min',
                                            period=1)
                
                # Fit model
                ts_model.fit(x=self.param_training_dict[lkback][noise][0], 
                             y=self.param_training_dict[lkback][noise][1], 
                             batch_size=32, epochs=5,
                             verbose=False, callbacks=[save_best], validation_data=(self.param_training_dict[lkback][noise][2], self.param_training_dict[lkback][noise][3]),
                             shuffle=False)
                
                # Retrieve model
                # best_model = load_model(os.path.join('keras_models', 'Sim_Data_LSTM_weights'))
                preds = ts_model.predict(self.param_training_dict[lkback][noise][0])
                pred_PRES = np.squeeze(preds)
                
                # Clean output for visualisation
                y_train_hat = np.array([pred[-1] for pred in pred_PRES])
                
                # Visualise residuals
                axs[i,j].set_title(f"LB:{lkback}-N:{noise}")
                axs[i,j].set_xlabel('True Return')
                axs[i,j].set_ylabel('Predicted Return')
                axs[i,j].set_xlim([0.996, 1.004])
                axs[i,j].set_ylim([0.996, 1.004])

                axs[i,j].scatter(np.exp([i for i in self.commod_df_clean[noise][lkback+1:7000]]), np.exp(y_train_hat), s=1)
                # axs[i,j].scatter(np.exp([i[-1] for i in self.param_training_dict[lkback][noise][1]]), np.exp(y_train_hat), s=1)
                
                # Print progress statistics
                print(f"Completed {model_num}/{num_of_models}")
                model_num += 1
    

    def makeXy(self, comm_df, cur_df, nb_timesteps):
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
        
        # Split full data into train, validation, and test sets
        comm_train_unscaled, cur_train_unscaled = pd.DataFrame(comm_df[0:int(0.7*n)]).reset_index(drop=True), pd.DataFrame(cur_df[0:int(0.7*n)]).reset_index(drop=True)
        comm_val_unscaled, cur_val_unscaled = pd.DataFrame(comm_df[int(0.7*n):int(0.9*n)]).reset_index(drop=True), pd.DataFrame(cur_df[int(0.7*n):int(0.9*n)]).reset_index(drop=True)
        comm_test_unscaled, cur_test_unscaled = pd.DataFrame(comm_df[int(0.9*n):]).reset_index(drop=True), pd.DataFrame(cur_df[int(0.9*n):]).reset_index(drop=True)
           
        # Reshape data to be vectors of length nb_timesteps and labels
        train_X, train_y, val_X, val_y, test_X, test_y = [], [], [], [], [], []
        
        # Train
        for i in range(nb_timesteps, comm_train_unscaled.shape[0]-1):
            train_X.append(np.array(cur_train_unscaled.loc[i-nb_timesteps:i]))
            train_y.append(np.array(comm_train_unscaled.loc[i-nb_timesteps:i]))
        train_X, train_y = np.array(train_X, dtype=object), np.array(train_y, dtype=object)

        # Validate
        for i in range(nb_timesteps, comm_val_unscaled.shape[0]-1):
            val_X.append(np.array(cur_val_unscaled.loc[i-nb_timesteps:i]))
            val_y.append(np.array(comm_val_unscaled.loc[i-nb_timesteps:i]))
        val_X, val_y = np.array(val_X, dtype=object), np.array(val_y, dtype=object)

        # Test
        for i in range(nb_timesteps, comm_test_unscaled.shape[0]-1):
            test_X.append(np.array(cur_test_unscaled.loc[i-nb_timesteps:i]))
            test_y.append(np.array(comm_test_unscaled.loc[i-nb_timesteps:i]))
        test_X, test_y = np.array(test_X, dtype=object), np.array(test_y, dtype=object)
        
        # Prepare data
        train_X = tf.convert_to_tensor(train_X, dtype='float64')
        train_y = tf.convert_to_tensor(train_y, dtype='float64')
        val_X = tf.convert_to_tensor(train_y, dtype='float64')
        val_y = tf.convert_to_tensor(train_y, dtype='float64')
        test_X = tf.convert_to_tensor(train_y, dtype='float64')
        test_y = tf.convert_to_tensor(train_y, dtype='float64')
        
        return train_X, train_y, val_X, val_y, test_X, test_y
    
#%%    
# test = LSTM_trainer([50,500,1000,2000],[0.0001,0.0005,0.001,0.005])
test = LSTM_trainer([500,1500,3000],[0,0.0002,0.0005])
test()
test.train()
