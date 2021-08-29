# This file contains functions for rolling analyses
# (linear/lasso/ridge regressions, PCA variance explained ratios)

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings("ignore")

import keras
from keras.layers import Dense, Input, Dropout
from keras.layers.recurrent import LSTM

from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

import tensorflow as tf

#%%

class Rolling_LR():
    
    # This class makes use of sci-kit-learn's linear regression function to
    # perform a rolling linear regression on a dataframe of any size.
    #
    # Analytics are available in the form of plots and dataframes,
    # to account for the time-course nature of the regression.
    
    
    def __init__(self):
        
        ## Initialise check that regression has not been fitted yet
        
        self.fitted = False
        self.outcome = []
        
     
        
    def fit(self, outcome, predictors, lookback, intercept = False, true_betas = []):
        
        ## Regress outcome series on predictors on a rolling window that is 'lookback' long.
        ## Intercept is not fitted by default.
         
        
        # Save inputs of regression for later analytics
        self.outcome = outcome
        self.predictors = predictors
        self.lookback = lookback
        self.true_betas = true_betas
        
         # Initialise empty array for beta coefficients
        self.beta_df = pd.DataFrame([[np.nan]*predictors.shape[1]]*predictors.shape[0], 
                                    columns = predictors.columns, 
                                    index = predictors.index)
    
        # Initialise empty array for R^2
        self.r_df = pd.DataFrame([[np.nan]*1]*predictors.shape[0], 
                                 index = predictors.index)
        
        # Initialise empty array for MSE
        self.mse_df = pd.DataFrame([[np.nan]*1]*predictors.shape[0], 
                                   index = predictors.index)
        
        # Initialise empty array for prediction
        self.pred_ts = pd.DataFrame([[np.nan]*1]*predictors.shape[0], 
                                    index = predictors.index, 
                                    columns = ['Prediction'])

        # Merge outcome and predictor series into a single dataframe
        full_df = predictors.copy()
        full_df['Y'] = outcome
        
        # Foward fill all na entries in full_df
        full_df = full_df.fillna(method='ffill')
        
        # Roll through each day
        for t in range(full_df.shape[0]-lookback): 
            
            # Splice data frame to the last lookback-1 days
            regression_window = full_df.iloc[t:t+lookback-1,:]
            
            # Perform linear regression
            cur_lr = LinearRegression(fit_intercept=intercept)
            cur_lr.fit(regression_window.iloc[:,:-1], regression_window.iloc[:,-1])
        
            # Save beta values for current day
            self.beta_df.iloc[t+lookback-1,:] = cur_lr.coef_
            
            # Save R^2 for current day
            self.r_df.iloc[t+lookback] = cur_lr.score(regression_window.iloc[:,1:], regression_window.iloc[:,0])
            
            # Save MSE for current day
            self.mse_df.iloc[t+lookback] = np.square(cur_lr.predict(regression_window.iloc[:,1:]) - regression_window.iloc[:,0]).mean()
            
            # Save prediction for current day
            self.pred_ts.iloc[t+lookback] = cur_lr.predict(np.array(full_df.iloc[t+lookback,1:]).reshape(1,-1))
        
        self.fitted = True
        
        
        
    def coefficients(self):
        
        ## Return series of beta coefficients
        
        if self.fitted == True:
            return self.beta_df
        else:
            print('No regression fitted')
            
    def pred_series(self):
        
        ## Return prediction time series
        
        if self.fitted == True:
            return self.pred_ts
        else:
            print('No regression fitted')
    
    def beta_plot(self):
        
        ## Plot time series of beta coefficients
        
        if self.fitted == True:
            # Plot beta time series
            plt.figure(figsize=(20,10))
            for col in self.beta_df.columns:    
                plt.plot(self.beta_df[col].iloc[self.lookback:], lw=1, label = col)
            plt.xlabel('Index')
            plt.ylabel('Value of Coefficicent in Linear Regression')
            plt.title('Estimated Beta Coefficients in Rolling Linear Regression')
            plt.legend(loc=3)
            plt.show()
        else:
            print('Please fit a regression first!')
          
            
    def R_plot(self):
        
        ## Plot series of cofficient of determination of the fitted model
        
        if self.fitted == True:
            
            # Plot coefficient of determination time series
            plt.figure(figsize=(20,10))
            plt.plot(self.r_df[self.lookback:], lw=1, label = 'R Squared')
            # plt.plot(mse_df[lookback:], lw=1, label = 'MSE')
            plt.xlabel('Year')
            plt.ylabel('Coefficient of Determination')
            plt.title('Plot of R^2 Over Time in Rolling Linear Regression')
            plt.legend(loc=3)
            plt.show()
        
        else:
            print('Please fit a regression first!')
        
        
    def MSE_plot (self):
    
        ## Plot series of cofficient of determination of the fitted model
        if self.fitted == True:

            # Plot coefficient of determination time series
            plt.figure(figsize=(20,10))
            
            # MSE Plot            
            plt.plot(self.mse_df[self.lookback:], lw=1, label = 'MSE')
            # Bias^2 PLot
            plt.plot(self.output-self.pred_ts)
            # Var PLot
            
            
            plt.xlabel('Index')
            plt.ylabel('Mean Squared Error')
            plt.title('Plot of MSE Over Time in Rolling Linear Regression')
            plt.legend(loc=3)
            plt.show()
        
        else:
            print('Please fit a regression first!')
    
         
    def pred_plot(self):
        
        ## Plot the fit of prediction time series
        
        if self.fitted == True:
            
            # Plot fitted time series against observed time series
            plt.figure(figsize=(20,10))
            plt.scatter(self.pred_ts[self.lookback:], self.outcome[self.lookback:], lw=1, label = 'Prediction')
            # plt.plot(outcome[lookback:], lw=1, label = 'True Outcome')
            plt.xlabel('Predicted Value')
            plt.ylabel('Observed Value')
            plt.title('Plot of Prediction Compared to True Outcome')
            plt.legend(loc=3)
            plt.show()
        
        else:
            print('Please fit a regression first!')

    def residual_plot(self):
        
        ## Plot time series of residuals
        
        if self.fitted == True:
            
            # Plot residual plot
            plt.figure(figsize=(20,10))
            # plt.scatter(x = outcome[lookback:], y = outcome[lookback:] - pred_ts[lookback:], lw=1, label = 'Prediction')
            plt.scatter(x = self.outcome[self.lookback:]-self.pred_ts[self.lookback:], y = self.outcome[self.lookback:]-self.pred_ts[self.lookback:], lw=1, label = 'Prediction')
            plt.ylabel('Residual Value')
            plt.xlabel('Observed Value')
            plt.title('Plot of Residuals Against True Outcome')
            plt.legend(loc=3)
            plt.show()
            
        else:
            print('Please fit a regression first!')
        
#%%
class Rolling_LR_OneD():
    
    # This class uses the particular result that in one dimension beta hat is 
    # the ratio of covariance of covariate and outcome variable with variance 
    # of covariate to perform a rolling linear regression on a dataframe of any size.

    # In streamlining the code for this function, detailed analytics have been
    # removed. These can still be computed using the regular rolling LR function.
    
    
    def __init__(self):
        
        ## Initialise check that regression has not been fitted yet
        
        self.fitted = False
        self.outcome = []
     
        
    def fit(self, outcome, predictor, lookback, intercept = False, true_betas = []):
        
        ## Regress outcome series on predictors on a rolling window that is 'lookback' long.
        ## Intercept is not fitted by default.
         
        # # Save inputs of regression for later analytics
        self.outcome = outcome[1:] # Remove potential initial condition for a series of returns
        self.predictor = predictor[1:] # Remove potential initial condition for a series of returns
        self.lookback = lookback
        self.true_betas = true_betas

        # Initialise empty array for beta coefficients
        self.beta_df = pd.DataFrame([[np.nan]]*self.predictor.shape[0], 
                                    columns = ['Beta'], 
                                    index = self.predictor.index)
    
        # Initialise empty array for prediction
        self.pred_ts = pd.DataFrame([[np.nan]]*self.predictor.shape[0], 
                                    index = self.predictor.index, 
                                    columns = ['Prediction'])

        # Merge outcome and predictor series into a single dataframe
        full_df = pd.DataFrame(self.predictor.copy())
        full_df = full_df.join(self.outcome)
        # return full_df
        # Foward fill all na entries in full_df
        # full_df = full_df.fillna(method='ffill')

        
        # Compute beta hat estimate using one-D result
        cov_mats = full_df.rolling(window=lookback).cov()
        cov_mats.reset_index(inplace=True)
        cov_mats = cov_mats[cov_mats.columns[-2:]]
        cov_mats = cov_mats[cov_mats.index % 2 == 0]
        cov_mats.reset_index(inplace=True, drop=True)
        
        beta_series = cov_mats[full_df.columns[1]] / cov_mats[full_df.columns[0]]
        # print(full_df.columns[1], full_df.columns[0])
        self.beta_df['Beta'] = np.array(beta_series)

        # Fill in prediction series
        self.pred_ts['Prediction'] = self.beta_df['Beta'] * self.predictor


        self.fitted = True
        
        
    def coefficients(self):
        
        ## Return series of beta coefficients
        
        if self.fitted == True:
            return self.beta_df
        else:
            print('No regression fitted')
            
    def pred_series(self):
        
        ## Return prediction time series
        
        if self.fitted == True:
            return self.pred_ts
        else:
            print('No regression fitted')
    
    def beta_plot(self):
        
        ## Plot time series of beta coefficients
        
        if self.fitted == True:
            # Plot beta time series
            plt.figure(figsize=(20,10))
            for col in self.beta_df.columns:    
                plt.plot(self.beta_df[col].iloc[self.lookback:], lw=1, label = col)
            plt.xlabel('Index')
            plt.ylabel('Value of Coefficicent in Linear Regression')
            plt.title('Estimated Beta Coefficients in Rolling Linear Regression')
            plt.legend(loc=3)
            plt.show()
        else:
            print('Please fit a regression first!')    
         
    def pred_plot(self):
        
        ## Plot the fit of prediction time series
        
        if self.fitted == True:
            
            # Plot fitted time series against observed time series
            plt.figure(figsize=(20,10))
            plt.scatter(self.pred_ts[self.lookback:], self.outcome[self.lookback:], lw=1, label = 'Prediction')
            # plt.plot(outcome[lookback:], lw=1, label = 'True Outcome')
            plt.xlabel('Predicted Value')
            plt.ylabel('Observed Value')
            plt.title('Plot of Prediction Compared to True Outcome')
            plt.legend(loc=3)
            plt.show()
        
        else:
            print('Please fit a regression first!')

    def residual_plot(self):
        
        ## Plot time series of residuals
        
        if self.fitted == True:
            
            # Plot residual plot
            plt.figure(figsize=(20,10))
            # plt.scatter(x = outcome[lookback:], y = outcome[lookback:] - pred_ts[lookback:], lw=1, label = 'Prediction')
            plt.scatter(x = self.outcome[self.lookback:], y = self.pred_ts[self.lookback:], lw=1, label = 'Prediction')
            plt.ylabel('Residual Value')
            plt.xlabel('Observed Value')
            plt.title('Plot of Residuals Against True Outcome')
            plt.legend(loc=3)
            plt.show()
            
        else:
            print('Please fit a regression first!')



#%%


class LSTM_predictor():

    
    ## This class takes in a one-dimensional covariate series and an observed series,
    ## splits the data temporally into training and testing sets, and trains a single LSTM
    ## regressor. The fitted values can then be returned to be leveraged into residuals for 
    ## signal generation.
    
    def __init__(self):
        
        ## Initialise check that regression has not been fitted yet
        self.fitted = False
        import tensorflow as t
        
        # Ensure we are working with GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
        

    def train(self, outcome, predictor, lookback):
        
        # Save lookabck variable
        self.lookback = lookback
        
        # Initialise LSTM Model
        input_layer = Input(shape=(lookback+1,1), dtype='float32')
        
        # lstm_layer = LSTM(10, input_shape=(lookback+1,1),
        #                   return_sequences=True)(input_layer)
        lstm_layer = LSTM(10, input_shape=(lookback+1,1),
                          return_sequences=True)(input_layer)
        output_layer = Dense(1, activation='linear')(lstm_layer)
        
        # Create datasets for training and testing
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.makeXy(pd.DataFrame(outcome), pd.DataFrame(predictor), self.lookback)
        # print('Shape of train arrays:', self.X_train.shape, self.y_train.shape)
        
        # Compile Model and begin training
        self.ts_model = Model(inputs=input_layer,
                     outputs=output_layer)
        self.ts_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                     optimizer=tf.keras.optimizers.Adam())
    
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        self.ts_model.fit(x=self.X_train, y=self.y_train, 
                 batch_size=32, epochs=10,
                 verbose=False, callbacks=[callback], validation_data=(self.X_val, self.y_val),
                 shuffle=False)
        
        # Confirm that fitting is complete
        self.fitted = True

    def test(self):
        
        self.test_predictions = np.array([pred[-1] for pred in np.squeeze(self.ts_model.predict(self.X_test))])
        
        return self. test_predictions
    
    def makeXy(self, comm_df, cur_df, nb_timesteps):
        """
        Input: 
                ts: original time series
                nb_timesteps: number of time steps in the regressors
        Output: 
                X: 2-D array of regressors
                y: 1-D array of target 
        """
        
        ## This function contains the logic for the creation of datasets
        ## of the correct dimensions to be used in training and testing.
        
        # Split data into train/val/test sets
        n = len(comm_df)
        
        # Grab column names
        comm_col, cur_col = comm_df.columns[0], cur_df.columns[0]
        
        # Split full data into train, validation, and test sets
        comm_train_unscaled, cur_train_unscaled = pd.DataFrame(comm_df[0:int(n*0.4)]).reset_index(drop=True), pd.DataFrame(cur_df[0:int(n*0.4)]).reset_index(drop=True)
        comm_val_unscaled, cur_val_unscaled = pd.DataFrame(comm_df[int(n*0.4):int(n*0.5)]).reset_index(drop=True), pd.DataFrame(cur_df[int(n*0.4):int(n*0.5)]).reset_index(drop=True)
        comm_test_unscaled, cur_test_unscaled = pd.DataFrame(comm_df[int(n*0.5):]).reset_index(drop=True), pd.DataFrame(cur_df[int(n*0.5):]).reset_index(drop=True)
        
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
    
        

        
           
            





























































#%%
# from sklearn.linear_model import Lasso

# def rolling_lasso(outcome, predictors, lookback, intercept, alph):
    
#     # Initialise empty array for beta coefficients
#     beta_df = pd.DataFrame([[np.nan]*predictors.shape[1]]*predictors.shape[0], columns = predictors.columns, index = predictors.index)
    
#     # Initialise empty array for R^2
#     r_df = pd.DataFrame([[np.nan]*1]*predictors.shape[0], index = predictors.index)
    
#     # Initialise empty array for MSE
#     mse_df = pd.DataFrame([[np.nan]*1]*predictors.shape[0], index = predictors.index)
    
#     # Initialise empty array for prediction
#     pred_ts = pd.DataFrame([[np.nan]*1]*predictors.shape[0], index = predictors.index)
    
#     # Merge input data
#     full_df = predictors.join(outcome, on = outcome.index)

#     # Roll through each day
#     for t in range(full_df.shape[0]-lookback):
        
#         # Splice data frame to the last lookback-1 days
#         regression_window = full_df.iloc[t:t+lookback-1,:].dropna()
        
#         # Perform linear regression
#         cur_lr = Lasso(alpha = alph, fit_intercept=intercept)
#         cur_lr.fit(regression_window.iloc[:,1:], regression_window.iloc[:,0])
        
#         # Save beta values for current day
#         beta_df.iloc[t+lookback-1,:] = cur_lr.coef_
        
#         # Save R^2 for current day
#         r_df.iloc[t+lookback] = cur_lr.score(regression_window.iloc[:,1:], regression_window.iloc[:,0])
        
#         # Save MSE for current day
#         mse_df.iloc[t+lookback] = np.square(cur_lr.predict(regression_window.iloc[:,1:]) - regression_window.iloc[:,0]).mean()
        
#         # Save prediction for current day
#         pred_ts.iloc[t+lookback] = cur_lr.predict(regression_window.iloc[:,1:])[-1]
    
#     # Plot beta time series
#     plt.figure(figsize=(20,10))
#     for col in beta_df.columns:    
#         plt.plot(beta_df[col].iloc[lookback:], lw=1, label = col)
#     plt.xlabel('Year')
#     plt.ylabel('Value of Coefficicent in Lasso Regression')
#     plt.title('Beta Coefficients in Rolling Lasso Regression')
#     plt.legend(loc=3)
#     plt.show()
    
#     # Plot coefficient of determination time series
#     plt.figure(figsize=(20,10))
#     plt.plot(r_df[lookback:], lw=1, label = 'R Squared')
#     # plt.plot(mse_df[lookback:], lw=1, label = 'MSE')
#     plt.xlabel('Year')
#     plt.ylabel('Coefficient of Determination')
#     plt.title('Plot of R^2 Over Time in Rolling Lasso Regression')
#     plt.legend(loc=3)
#     plt.show()
    
    
#     # Plot fitted time series against observed time series
#     # plt.figure(figsize=(20,10))
#     # plt.plot(pred_ts[lookback:] - outcome[lookback:], lw=1, label = 'Prediction')
#     # # plt.plot(outcome[lookback:], lw=1, label = 'True Outcome')
#     # plt.xlabel('Year')
#     # plt.ylabel('Coefficient of Determination')
#     # plt.title('Plot of Prediction Compared to True Outcome')
#     # plt.legend(loc=3)
#     # plt.show()
    
#     return beta_df

# # rolling_reg_coeffs = rolling_lr(PC_proj_df['1st PC Projection'], PC_proj_df.iloc[:,1:], 500, False)



# #%%

# from sklearn.linear_model import Ridge

# def rolling_ridge(outcome, predictors, lookback, intercept, alph):
    
#     # Initialise empty array for beta coefficients
#     beta_df = pd.DataFrame([[np.nan]*predictors.shape[1]]*predictors.shape[0], columns = predictors.columns, index = predictors.index)
    
#     # Initialise empty array for R^2
#     r_df = pd.DataFrame([[np.nan]*1]*predictors.shape[0], index = predictors.index)
    
#     # Initialise empty array for MSE
#     mse_df = pd.DataFrame([[np.nan]*1]*predictors.shape[0], index = predictors.index)
    
#     # Initialise empty array for prediction
#     pred_ts = pd.DataFrame([[np.nan]*1]*predictors.shape[0], index = predictors.index)
    
#     # Merge input data
#     full_df = predictors.join(outcome, on = outcome.index)

#     # Roll through each day
#     for t in range(full_df.shape[0]-lookback):
        
#         # Splice data frame to the last lookback-1 days
#         regression_window = full_df.iloc[t:t+lookback-1,:].dropna()
        
#         # Perform linear regression
#         cur_lr = Ridge(alpha = alph, fit_intercept=intercept)
#         cur_lr.fit(regression_window.iloc[:,1:], regression_window.iloc[:,0])
        
#         # Save beta values for current day
#         beta_df.iloc[t+lookback-1,:] = cur_lr.coef_
        
#         # Save R^2 for current day
#         r_df.iloc[t+lookback] = cur_lr.score(regression_window.iloc[:,1:], regression_window.iloc[:,0])
        
#         # Save MSE for current day
#         mse_df.iloc[t+lookback] = np.square(cur_lr.predict(regression_window.iloc[:,1:]) - regression_window.iloc[:,0]).mean()
        
#         # Save prediction for current day
#         pred_ts.iloc[t+lookback] = cur_lr.predict(regression_window.iloc[:,1:])[-1]
    
#     # Plot beta time series
#     plt.figure(figsize=(20,10))
#     for col in beta_df.columns:    
#         plt.plot(beta_df[col].iloc[lookback:], lw=1, label = col)
#     plt.xlabel('Year')
#     plt.ylabel('Value of Coefficicent in Lasso Regression')
#     plt.title('Beta Coefficients in Rolling Lasso Regression')
#     plt.legend(loc=3)
#     plt.show()
    
#     # Plot coefficient of determination time series
#     plt.figure(figsize=(20,10))
#     plt.plot(r_df[lookback:], lw=1, label = 'R Squared')
#     # plt.plot(mse_df[lookback:], lw=1, label = 'MSE')
#     plt.xlabel('Year')
#     plt.ylabel('Coefficient of Determination')
#     plt.title('Plot of R^2 Over Time in Rolling Lasso Regression')
#     plt.legend(loc=3)
#     plt.show()
    
#     # Plot fitted time series against observed time series
#     # plt.figure(figsize=(20,10))
#     # plt.plot(pred_ts[lookback:] - outcome[lookback:], lw=1, label = 'Prediction')
#     # # plt.plot(outcome[lookback:], lw=1, label = 'True Outcome')
#     # plt.xlabel('Year')
#     # plt.ylabel('Coefficient of Determination')
#     # plt.title('Plot of Prediction Compared to True Outcome')
#     # plt.legend(loc=3)
#     # plt.show()
    
#     return beta_df

# #%% Rolling PCA Projection

# def PC_proj_ts(returns_data, lookback, title):

#     # Initialising empty projection array
#     proj_col = [np.nan]*returns_data.shape[0]

#     # Rolling through each day
#     for t in range(returns_data.shape[0]-lookback):
        
#         # Splice data frame to last ookback-1 days
#         returns_window = returns_data.iloc[t:t+lookback-1,:]
        
#         # Perform PCA on all non-na rows
#         cur_pca = PCA(n_components = returns_data.shape[1], svd_solver='full')
#         cur_pca.fit(returns_window.dropna())
        
#         # Check for NaN in day t+1
#         if returns_data.iloc[t+lookback,:].dropna().shape[0] < returns_data.shape[1]:
#             # If PCA cant be computed, set projection to np.nan
#             proj_col[t+lookback] = np.nan
#         else:
#             # Project day t's returns onto PC space and take first component
#             day_t_proj = cur_pca.transform(np.array(returns_data.iloc[t+lookback,:]).reshape(1,-1))
#             proj_col[t+lookback] = day_t_proj[0][0]

            
#     # Save results into a dataframe
#     proj_df = pd.DataFrame([], index = returns_data.index)
#     proj_df['1st PC Projection'] = proj_col
    
#     # Plot resulting time series
#     plt.figure(figsize=(20,10))    
#     plt.plot(proj_df['1st PC Projection'].iloc[lookback:], lw=1)
#     plt.xlabel('Year')
#     plt.ylabel('Projection onto 1st PC')
#     plt.title('Projection of {} Returns onto 1st PC'.format(title))
#     # plt.legend()
#     plt.show()
    
#     return proj_df