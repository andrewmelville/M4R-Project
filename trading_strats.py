from rolling_functions import Rolling_LR, Rolling_LR_OneD
from plotting_functions import series_plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MeanReversion():
    
    
    # This class holds different functions for trading strategies and the analysis
    # thereof. At the moment, we have a simple Mean-Reversion strategy which creates signals
    # determined by the residuals of an appropriate rolling linear regression
    
    
    def __init__(self):
        
        self.residuals_df = []
        self.lookback = []
        self.chunk_size = []
        self.signals_df = []
    
    
    def back_test(self, currency_returns, true_commods_returns, noise, chunk_size, lookback, noise_props=[0,0.25,0.5,0.75,1]):  
        
        self.chunk_size = chunk_size
        self.lookback = lookback
        self.noise_props = noise_props
        self.noisy_commods_returns = true_commods_returns + noise # These are the returns that all strategies/models will trade on
        self.currency_returns = currency_returns
        
        # Create empty residual dataframe
        self.residuals_df = pd.DataFrame([0]).reindex_like(self.noisy_commods_returns)
        self.beta_df = pd.DataFrame([]).reindex_like(self.noisy_commods_returns)
        self.pred_series = pd.DataFrame([]).reindex_like(self.noisy_commods_returns)
        
        # Get daily (simple) commodities prices with same contracts as signal_df
        self.daily_prices = pd.DataFrame([])
        
        for commod in self.noisy_commods_returns:   
            self.daily_prices[commod] = np.exp([i for i in self.noisy_commods_returns[commod]]).cumprod()
        
        # Create empty df to hold PL curves
        PL_curve_df = pd.DataFrame([], index = currency_returns.index, columns=["{:.1f}%".format(noise_level*100) for noise_level in noise_props])
        
        # Perform trade passes at each level of noise in noise_props for comparison of performance
        for noise_level in noise_props:
            
            PL_curve_df["{:.1f}%".format(noise_level*100)] = self.noisy_trade(true_commods_returns + (noise_level * noise), noise_level)
            print("Trade on {:.1f}% Noise Level Complete".format(noise_level*100))
        
        # Plot PL Curves
        series_plot(PL_curve_df, 'Strategy Performance Compared to Optimal Performance', legend=True)
        
        return PL_curve_df
    
    
    def noisy_trade(self, added_noise_commods_returns, noise_level):
        
        ## This function estimates the residuals using some specified model (currently LR only)
        ## and calls the class function 'trade' to compute a PL curve for the generated residuals.
        
        # Create df of chunk signals using Signals function
        self.Residuals(added_noise_commods_returns)
        
        # Perform trade algorithm
        return self.trade(noise_level)
    
    
    def trade(self, noise_level):
        
        ## This function creates a signal from the class variable dataframe of residuals
        ## which are clauclated before this function is called. It then performs
        ## the trading over the testing period by multiplying this signal df
        ## against the prices for each day to generate a PL curve.
        
        # Create df of chunk signals using Signals function
        self.Signals()
        print('{:.1f}% Noise Level signals generated'.format(noise_level*100))
                                                
        # Multiply signals df by simple daily prices df to get daily P/L for each contract
        commod_PL = self.signal_df * self.daily_prices.diff()
        
        # Sum across columns for daily P/L, cumsum daily P/L for P/L curve
        PL_curve = commod_PL.sum(axis=1).cumsum()

        return PL_curve
    
        
    def Residuals(self, commods_returns):
        
        ## This function creates a dataframe of commoditiy residuals from a 
        ## rolling linear regression onto a currency averages returns. These residuals
        ## will then be used to create several dataframes of trading signals.
        
        # Loop through each commodity
        for i, commod in enumerate(commods_returns.columns):
            
            # Regress currency average onto commodities returns and take # prediction series            
            roll_reg = Rolling_LR_OneD()
            roll_reg.fit(commods_returns[commod], self.currency_returns, lookback = self.lookback)
            self.pred_series[commod] = roll_reg.pred_ts
            
            self.beta_df[commod] = roll_reg.beta_df
            self.residuals_df[commod] = self.noisy_commods_returns[commod] - self.pred_series[commod]
            
            # print('{} residuals completed {}/{}'.format(commod, i+1, commods_returns.shape[1]))
    
    
    def Signals(self):
        
        ## This function takes a df of residuals and splits it into chunks of a
        ## specified size. The residuals are then averaged over the chunk, ranked,
        ## and then a signal is applied to the contract for the month after, held
        ## in signals_df
        
        # Create empty signals df
        self.signal_df = pd.DataFrame([0]).reindex_like(self.residuals_df)
        
        # Split full returns data of commodities into chunks that are to be used as trading windows.
        chunk_list = np.array_split(self.residuals_df, np.floor(len(self.residuals_df) / self.chunk_size))
        
        # Loop through each trading chunk and make a df of that chunks signals
        for i, chunk in enumerate(chunk_list[:-1]):
    
            # Average residuals over current chunk
            current_chunk_list = chunk.mean().sort_values(axis=0,ascending = False)
            
            # Get top three positive residual contracts
            pos_mask = current_chunk_list > 0
            sell_list = current_chunk_list[pos_mask]
            
            # Mask to select month ahead
            signal_mask = chunk_list[i+1].index

            # Assign positive (sell) value to contracts for month ahead
            self.signal_df.loc[signal_mask, sell_list.index[:3]] = 1

            # Get bottom three negative residual contracts
            neg_mask = current_chunk_list < 0
            buy_list = current_chunk_list[neg_mask]
            
            # Assign negative (buy) value to contracts for month ahead
            self.signal_df.loc[signal_mask, buy_list.index[-3:]] = -1
    
    
    def beta_plot(self):
     
        ## Plot time series of estimated beta coefficients
        
        # Plot beta time series
        plt.figure(figsize=(20,10))
        for col in self.beta_df.columns:    
           plt.plot(self.beta_df[col], lw=1, label=col)
       
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Estimated Beta Coefficients that generated residual signals')
        plt.legend(loc=3)
        plt.show()