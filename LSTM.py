import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('-f', '--filename', type = str)
# args = parser.parse_args()

# filepath = args.filename

# from __future__ import print_function
import os
import sys
import pandas as pd
import numpy as np
import datetime


# First read in the data    
df = pd.read_csv("weather_data.csv")

df['datetime'] = df[['year', 'month', 'day', 'hour']].apply(lambda row: datetime.datetime(year=row['year'], month=row['month'], day=row['day'],
                                                                                          hour=row['hour']), axis=1)
df.sort_values('datetime', ascending=True, inplace=True)



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_PRES'] = scaler.fit_transform(np.array(df['PRES']).reshape(-1, 1))

split_date = datetime.datetime(year=2014, month=1, day=1, hour=0)
df_train = df.loc[df['datetime']<split_date]
df_val = df.loc[df['datetime']>=split_date]
print('Shape of train:', df_train.shape)
print('Shape of test:', df_val.shape)


df_val.reset_index(drop=True, inplace=True)
# Now we need to generate regressors (X) and target variable (y) for train and validation. A 2-D array of regressor and 1-D array of target is created from the original 1-D array of columm log_PRES in the DataFrames. For the time series forecasting model, the past seven days of observations are used to predict for the next day. This is equivalent to a AR(7) model. We define a function which takes the original time series and the number of timesteps in regressors as input to generate the arrays of X and y.

def makeXy(ts, nb_timesteps):
    """
    Input: 
           ts: original time series
           nb_timesteps: number of time steps in the regressors
    Output: 
           X: 2-D array of regressors
           y: 1-D array of target 
    """
    X = []
    y = []
    for i in range(nb_timesteps, ts.shape[0]):
        X.append(list(ts.loc[i-nb_timesteps:i-1]))
        y.append(ts.loc[i])
    X, y = np.array(X), np.array(y)
    return X, y


X_train, y_train = makeXy(df_train['scaled_PRES'], 7)
print('Shape of train arrays:', X_train.shape, y_train.shape)



X_val, y_val = makeXy(df_val['scaled_PRES'], 7)
print('Shape of validation arrays:', X_val.shape, y_val.shape)


# The input to RNN layers must be of shape (number of samples, number of timesteps, number of features per timestep). In this case we are modeling only air pressure, hence number of features per timestep is one. The number of timesteps is seven and number of samples is same as the number of samples in X_train and X_val, which are reshaped to 3D arrays.
X_train, X_val = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)),                 X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
print('Shape of 3D arrays:', X_train.shape, X_val.shape)


# Now we define the neural network topology using the Keras Functional API. In this approach a layer can be declared as the input of the following layer at the time of defining the next layer. 
import keras
from keras.layers import Dense, Input, Dropout, Masking
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional, RNN, LSTMCell

# Define input layer which has shape (None, 7) and of type float32. None indicates the number of instances in the mini-batch used for training.
input_layer = Input(shape=(7,1), dtype='float32')

# The LSTM layer is defined for seven timesteps
lstm_layer = LSTM(10, input_shape=(7,1), return_sequences=False)(input_layer)
# lstm_layer = Bidirectional(RNN(LSTMCell(10, input_shape=(7,1), dtype="float32")))(input_layer)

# Use Dropout regularization - the parameter chosen here can in principle be optimized.
dropout_layer = Dropout(0.2)(lstm_layer)


# Finally, the output layer gives prediction for the next day's air pressure. This should be a real number value, and so no activation function is used (i.e. linear activation).
output_layer = Dense(1, activation='linear')(lstm_layer)




ts_model = Model(inputs=input_layer, outputs=output_layer)
ts_model.compile(loss='mae', optimizer='adam')
ts_model.summary()

save_weights_at = os.path.join('keras_models', 'PRSA_data_Air_Pressure_LSTM_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min',
                            period=1)
ts_model.fit(x=X_train, y=y_train, batch_size=16, epochs=2,
             verbose=1, callbacks=[save_best], validation_data=(X_val, y_val),
             shuffle=True)


# Prediction are made for the air pressure from the best saved model. The model's predictions, which are on the scaled  air-pressure, are inverse transformed to get predictions on original air pressure. The goodness-of-fit, R-squared is also calculated for the predictions on the original variable.
# In[ ]:


best_model = load_model(os.path.join('keras_models', 'PRSA_data_Air_Pressure_LSTM_weights.02-0.0128.hdf5'))
preds = best_model.predict(X_val)
pred_PRES = scaler.inverse_transform(preds)
pred_PRES = np.squeeze(pred_PRES)


# In[ ]:


from sklearn.metrics import r2_score
r2 = r2_score(df_val['PRES'].loc[7:], pred_PRES)
print('R-squared on validation set of the original air pressure:', r2)


# Let's plot the first 50 actual and predicted values of air pressure.

# In[ ]:
import matplotlib.pyplot as plt    

k=8000
plt.figure(figsize=(10, 10))
plt.plot(range(k), df_val['PRES'].loc[7:(6+k)], marker='*', color='r',lw=0.1)
plt.plot(range(k), pred_PRES[:k], marker='.', color='b', lw=0.1)
plt.legend(['Actual','Predicted'], loc=2)
plt.title('Actual vs Predicted Air Pressure')
plt.ylabel('Air Pressure')
plt.xlabel('Index')