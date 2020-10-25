This is the README file for my M4R Project.

Project started 15/10/20 with implementation of brownian motion generator and 
simple model creator function"

The model functions are defined in models.py, which currently only holds the 
rolling linear model we discussed in our last meeting. This will be expanded
to include ARMA, ARIMA models etc as the random walk is generated seperately in 
brownian_motion.py.

This allows for different types of data to be fed into different models. A similar 
modularisation of the process is found in the beta_functions.py, kept seperate 
so that different relationships between covariates and outcome can be modelled 
easily.

Andrew, 15/10/20


Added Rolling Linear Regression functions in rolling_functions.py, and a gif making
fucntion in plotting_functions.py that saves a gif explaining what data the rolling fit
used to estimate the rolling betas.