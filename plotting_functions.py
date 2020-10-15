# This file holds different functions to be sed for standardised plotting of 
# time series and their analytics

# 15/10/20 Andrew Melville


import matplotlib.pyplot as plt

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
