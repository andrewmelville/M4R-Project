# This file holds different functions to be used for standardised plotting of 
# time series and their analytics

# 15/10/20 Andrew Melville


import matplotlib.pyplot as plt
import imageio
import numpy as np

def series_plot(data, title, xlab = 'Index', ylab = 'Value', legend = False):
    plt.figure(figsize=(20,10))
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    
    for series in data:
        plt.plot(data[series], label = series)
    
    if legend == True:
        plt.legend()

def rolling_beta_plot(covariates, coefficients, output, lookback, gifname, frames = 0.25, freq = 100):
    
    images = []
    
    for i in range(len(coefficients)-lookback):
        
        if (i+lookback) % freq == 0:
    
            x = np.linspace(-100, 100, num=100)
            y =  float(coefficients.loc[i+lookback]) * x
    
            fig, ax = plt.subplots(figsize=(10,10))
            ax.scatter(covariates[i:i+20], output[i:i+lookback])
            ax.plot(x, y, 'r')
            
            ax.grid()
            ax.set(xlabel = 'Simulated Currency Returns', ylabel = 'Simulated Model Output',
                   title = 'True Beta through time')
    
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
    imageio.mimsave('./Plots/' + gifname + '.gif', images, fps=frames)