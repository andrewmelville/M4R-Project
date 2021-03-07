# This file holds different functions to be used for standardised plotting of 
# time series and their analytics

# 15/10/20 Andrew Melville


import matplotlib.pyplot as plt
import imageio
import numpy as np
import pandas as pd

def series_plot(data, title, linesize = [], xlab = 'Index', ylab = 'Value', legend = False):
    
    ## This function defines a uniform look for the plots of the data in this project
    
    
    # If linewidths of each series are not specified, assign each plot the same linewidth
    if len(linesize) == 0:
        linesize = [1 for i in data]
    
    # Create the figure
    plt.figure(figsize=(20,10))
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    
    # Turn data inputs into single pandas dataframe
    if type(data) == list:
        data_concat = pd.concat(data, axis=1)
    else:
        data_concat = data.copy()
    
    # Plot each series onto the axes
    for i, series in enumerate(data_concat):
        plt.plot(data_concat[series], label=series, lw=linesize[i])
    
    # If legend is asked for, print it
    if legend == True:
        plt.legend()

def rolling_beta_plot(covariates, true_coefficients, est_coefficients, output, lookback, gifname, freq = 50):
    
    images = []
    
    for i in range(len(true_coefficients)-lookback):
        
        if (i+lookback) % freq == 0:
    
            x_true = np.linspace(-100, 100, num=100)
            y_true =  float(true_coefficients.loc[i+lookback]) * x_true
            
            x_est = np.linspace(-100, 100, num=100)
            y_est =  float(est_coefficients.loc[i+lookback]) * x_est
    
            fig, ax = plt.subplots(figsize=(10,10))
            ax.scatter(covariates[i:i+lookback-1], output[i:i+lookback-1])
            # ax.scatter(covariates.loc[i+lookback], output.loc[i+lookback], 'r')
            ax.plot(x_true, y_true, 'r', label = 'True')
            ax.plot(x_est, y_est, 'g', label = 'Estimated')
            
            ax.grid()
            ax.set(xlabel = 'Simulated Currency Returns', ylabel = 'Simulated Model Output',
                   title = 'True Beta through time')
            ax.legend()
            
            # IMPORTANT ANIMATION CODE HERE
            # Used to keep the limits constant
            ax.set_ylim(min(output.values), max(output.values))
            ax.set_xlim(min(covariates.values), max(covariates.values))
        
            # Used to return the plot as an image rray
            fig.canvas.draw()       # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            images.append(image) 
            
            print('{}/{} Completed'.format(int((i+lookback)/freq), np.floor(len(covariates)/freq)))
    imageio.mimsave('./Plots/' + gifname + '.gif', images, fps=20)
    

def signal_plot(returns, signals):
    
    signals = np.array(signals)
    
    plt.figure(figsize=(20,10))
    plt.title('Predicted Buy and Sell Signals for Next Day Returns')
    plt.xlabel('Index')
    plt.ylabel('Commodity Value')
    
    # Plot price series
    plt.plot(returns.cumsum(), lw=0.5)
    
    # Create buy/sell masks
    buy_mask = signals > 0
    sell_mask = signals <= 0
    
    # Plot buy/sell signals atop price series
    plt.plot(returns.cumsum()[buy_mask], 'g^', lw=0.01)
    plt.plot(returns.cumsum()[sell_mask], 'rv', lw=0.01)
    
    plt.show()
