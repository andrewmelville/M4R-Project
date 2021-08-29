# This is where I will bring together all the modules of my code project for integrated testing

# 15/10/20 Andrew Melville
# from brownian_motion import walk_generator
from plotting_functions import series_plot, pred_truth_vis
from beta_functions import beta_generator
from models import model_generator
from rolling_functions import Rolling_LR_OneD

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
from brownian_motion import geo_bm, bm_std
# brown = geo_bm(n=10000, d=1, drift = 0.000001, sigma=0.005)
sig_val = 0.005
mu_val = sig_val**2/2

brown2 = geo_bm(d=5, n=10000, mu=mu_val, sigma=sig_val, initial_range=[0,1])

series_plot(brown2, 'Sample Paths of Simulated Currency Series', fontsize=32)

#%%
from models import model_generator
test_model = model_generator()

model = test_model.linear_model(num_obs=1000,
                                num_covariates=1, 
                                beta_type='bm_std',
                                beta_sigma=0.005,
                                noise=0.0005)


plt.figure(figsize=(20,10))
plt.title('Signal and Noise in Commodities Prices', fontsize=30)
plt.plot(np.exp([i for i in test_model.covariates()['True'][1][:]]).cumprod(), lw=2, label='Implied Signal')
plt.plot(np.exp([i for i in test_model.covariates()['Noisy'][1][:]]).cumprod(), lw=2, label='Mean-Reverting Price')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.xlabel('Date Index', fontsize=30)
plt.ylabel('Price', fontsize=30)
# plt.plot(test_model.covariates()['Noisy'][1:])
plt.legend(fontsize=25)


#%%
from ARMA import ARMAmodel
from statsmodels.tsa.stattools import acf
arma = ARMAmodel()
l=1
x=arma(n=1000,
      phi=[.98],
      theta=[0]+[(i+1)**(-2) for i in range(100)],
      sigma=0.00001,
      burnin=10000)

plt.figure(figsize=(15,10))
# plt.xlim((0,1000))
# plt.title('Sample Auto Correlation Function of ARMA Process', fontsize=32)

# plt.xlabel('Lag', fontsize=22)
# plt.ylabel('Correlation', fontsize=22)

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.plot(x)
# plt.plot(np.log(x/x.shift(1)))
# plt.plot(acf(x[0],fft=True, nlags=1000))
# plt.psd(x[0].values)

# series_plot(x,'Sample Path of ARMA Process', fontsize=32)
#%%

from rolling_functions import Rolling_LR_OneD
import time


from models import model_generator
test_model = model_generator()

model = test_model.linear_model(num_obs=10000,
                                num_covariates=1, 
                                beta_type='bm_std',
                                beta_sigma=0.5,
                                noise=0.0001)
    
reg = Rolling_LR()
reg_oneD = Rolling_LR_OneD()


    
# one_D_times = []
# regular_times = []

# for i in range(100,1010,10):
    
#     test_model = model_generator()
#     model = test_model.linear_model(num_obs=i,
#                                     num_covariates=1, 
#                                     beta_type='bm_std',
#                                     beta_sigma=0.0035,
#                                     noise=0.0001)
#     oneD_obs = []
#     for k in range(3):
        
#         start_time = time.time()
#         reg_oneD.fit(covs['True'][:],model, 120)
#         end_time = time.time()
#         oneD_obs.append(end_time - start_time)
        
#     one_D_times.append(np.mean(oneD_obs))
    
#     reg_obs = []
#     for k in range(3):
#         start_time = time.time()
#         reg.fit(covs['True'][:],model, 120)
#         end_time = time.time()
#         reg_obs.append(end_time - start_time)
        
#     regular_times.append(np.mean(reg_obs))

#%%

plt.figure(figsize=(20,10))
plt.title('Reducing Fit Time of Multiple Linear Regresions', fontsize=27)

plt.xlabel('Number of Regressions fit', fontsize=22)
plt.ylabel('Run Time (seconds)', fontsize=22)

plt.xticks(np.arange(0,100,10),[str(k) for k in range(0,1000,100)], fontsize=22)
plt.yticks(fontsize=22)
plt.plot(one_D_times, label='One-Dimensional LR')

plt.plot(regular_times, label='SK-Learn LR')

plt.legend(fontsize=22)

#%%

from models import model_generator
test_model = model_generator()

model = test_model.linear_model(num_obs=500,
                                num_covariates=1, 
                                beta_type='bm_std',
                                beta_sigma=0.003,
                                noise=0.0001)
# betas = test_model.params
covs = test_model.covariates()

plt.figure(figsize=(20,10))
plt.title('Components in Simulation of Commodities', fontsize=32)
plt.xticks(fontsize=27)
plt.yticks(fontsize=27)

plt.xlabel('Index', fontsize=27)
plt.ylabel('Simulated Value', fontsize=27)

plt.plot(np.exp([i for i in test_model.covariates()['True'][1]]).cumprod(), label='Price Series Implied by the Currency Log-Returns')
plt.plot(np.exp([i for i in test_model.covariates()['Noisy'][1]]).cumprod(), label='Final Price Series (Implied log-Returns + ARMA Noise')
plt.legend(fontsize=27)
# plt.plot(test_model.arma_noise)

# test_model.beta_plot()
# test_model.model_plot()
# #test_model.noisy_covariates_plot()
# test_model.true_covariates_plot()

# plt.figure(figsize=(20,10))
# plt.plot(np.exp([i for i in covs['Noisy'][1]]).cumprod()), plt.plot(np.exp([i for i in covs['True'][1]]).cumprod())
# plt.scatter(covs['Noisy'][1][1:], covs['True'][1][1:])
#%%
from rolling_functions import Rolling_LR_OneD


lb_mse_list = []
lookbacks = [i for i in range(10,500)]
model = test_model.linear_model(num_obs=2000,
                            num_covariates=20, 
                            beta_type='cor_bb',
                            beta_sigma=0.03,
                            noise=0.001,
                            t=0)

# reg = Rolling_LR_OneD()

# reg.fit(test_model.covariates()['Noisy'][1], model, 50)
# series_plot(reg.beta_df,'')
# series_plot()
#%%
for k in range(20):

    reg_oneD = Rolling_LR_OneD()
    lookback_df = pd.DataFrame([], index=test_model.params.index, columns=lookbacks)
    lookback_df['True Beta Series'] = test_model.params[k+1]
    
    for lb in lookbacks:
    
        reg_oneD.fit(test_model.covariates()['Noisy'][k+1], model, lb)
        lookback_df[lb] = reg_oneD.beta_df
        
    lookback_sq_er = pd.DataFrame([]).reindex_like(lookback_df.iloc[:,:-1])
    
    # Compute Mean Squared Error of Estimate
    for win_len in lookback_df.iloc[:,:-1]:

        lookback_sq_er[win_len] = (lookback_df[win_len] - lookback_df['True Beta Series'])**2

    lookback_mses = lookback_sq_er.mean()

    lb_mse_list.append(lookback_mses)
    print("{}/k".format(k+1))


plt.figure(figsize=(20,10))
plt.title('Average Squared Error Across Each Estimated Beta Series with Different Window Lengths', fontsize=27)
plt.xlabel('Window Length', fontsize=27)
plt.ylabel('Avg Squared Error', fontsize=27)

plt.xticks(fontsize=27)
plt.yticks(fontsize=27)

for lb in lb_mse_list:
    
    plt.plot(lb)

# %%

series_plot(lookback_df[[10,50,120,'True Beta Series']],'Estimated Beta Series for Different Lookback Window Sizes', legend=True, fontsize=32, linesize=[1,3,3,3])



# plt.scatter(covs['Noisy'][1][600:], covs['Noisy'][1][600:] - reg_oneD.pred_ts['Prediction'].dropna())
# pred_truth_vis(covs['Noisy'][1][200:], [i for i in (covs['Noisy'][1][200:]-reg_oneD.pred_ts['Prediction'].dropna())])

# preds = reg_oneD.pred_series()
# pred_truth_vis(covs['Noisy'][1], preds)

 #%%

# from rolling_functions import Rolling_LR
# t = 0

# reg = Rolling_LR()
# reg.fit(covs['True'][:], model, 120)
# # reg_oneD.fit((1-t)*covs['True'][:] + t*covs['Noisy'][:], model, 2)
# reg.beta_plot()
# # series_plot(test_model.params.join(reg_oneD.beta_df), '', legend=True)
# # plt.scatter(covs['Noisy'][1][600:], covs['Noisy'][1][600:] - reg_oneD.pred_ts['Prediction'].dropna())
# # pred_truth_vis(covs['Noisy'][1][200:], [i for i in (covs['Noisy'][1][200:]-reg_oneD.pred_ts['Prediction'].dropna())])

# # preds = reg_oneD.pred_series()
# # pred_truth_vis(covs['Noisy'][1], preds)
#%%
from trading_strats import MeanReversion
from models import model_generator

high_freq_model = model_generator()
cur_ret = high_freq_model.linear_model(num_obs=10000, 
                                       num_covariates=27, 
                                       beta_type='cor_bb', 
                                       beta_sigma=0.06, 
                                       noise=0.01,
                                       t=0.55,
                                       mean_reverting_amount=0.6)
betas = high_freq_model.params
noisy_covs = high_freq_model.covariates()['Noisy']
true_covs = high_freq_model.covariates()['True']
arma = high_freq_model.arma_noise
# high_freq_model.beta_plot()
#%%
from trading_strats import MeanReversion
mean_rev = MeanReversion()
test = mean_rev.back_test(cur_ret,
                          true_covs,
                          noisy_covs,
                          arma,
                          chunk_size=30,
                          lookback=50,
                          noise_props=[1],
                          plot=True, 
                          pos_ratio=1/3)

print(np.mean(mean_rev.rank_cor_list))# mean_rev.beta_plot()
# high_freq_model.beta_plot()
#%% CHANGING Beta Sigma BEHAVIOUR


variable_change_PL_list = []
#%% ARMA Noise influence on price data
from trading_strats import MeanReversion
# [0,0.25,0.5,0.75,1,5]
for variable in [0]:

    cur_ret = high_freq_model.linear_model(num_obs=10000, 
                                           num_covariates=27, 
                                           beta_type='cor_bb', 
                                           beta_sigma=0.06, 
                                           noise=0.01,
                                           t=0.55)
    
    noisy_covs = high_freq_model.covariates()['Noisy']
    true_covs = high_freq_model.covariates()['True']
    arma = high_freq_model.arma_noise
    
    mean_rev = MeanReversion()
    test = mean_rev.back_test(cur_ret,
                              true_covs,
                              noisy_covs,
                              arma,
                              chunk_size=30,
                              lookback=50,
                              noise_props=[variable],
                              plot=False, 
                              pos_ratio=1/3)
    
    variable_change_PL_list.append(mean_rev.LR_PL_curve_df.iloc[:,0])
    # plt.plot(df[5000:], label=labels[i], linewidth=3)
    # variable_change_cor_list.append(np.mean(mean_rev.rank_cor_list))
    print(variable)
#%%
# variable_change_PL_list[-1] = mean_rev.LR_PL_curve_df.iloc[:,0]
# variable_change_PL_list.append(mean_rev.LSTM_PL_curve_df.iloc[:,0])
#%%
plt.figure(figsize=(20,10))
plt.title('RLR on Different Proportions of ARMA Noise in Simulated Log-Returns', fontsize=32)
plt.xlabel('Date Index', fontsize=30)
plt.ylabel('Beta Value', fontsize=30)

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

labels = ['Signals on 0% of Noise',
          'Signals on 50% of Noise',
          'Signals on 100% of Noise',
          'Signals on 2000% of Noise',
          'Random Signals']
# colors = []

for i, df in enumerate(variable_change_PL_list):

    plt.plot(df[5000:], label=labels[i], linewidth=3)
    
plt.legend(fontsize=22)













#%%
# variable_change_cor_list = []

from trading_strats import MeanReversion
from models import model_generator


plt.figure(figsize=(20,10))
 # [0.06, 0.12, 0.18, 0.24, 0.3, 0.36]
for variable in [0.06]:
    
    high_freq_model = model_generator()
    cur_ret = high_freq_model.linear_model(num_obs=10000, 
                                           num_covariates=27, 
                                           beta_type='cor_bb', 
                                           beta_sigma=variable, 
                                           noise=0.01,
                                           t=0.55)
    
    noisy_covs = high_freq_model.covariates()['Noisy']
    true_covs = high_freq_model.covariates()['True']
    arma = high_freq_model.arma_noise
    
    mean_rev = MeanReversion()
    test = mean_rev.back_test(cur_ret,
                              true_covs,
                              noisy_covs,
                              arma,
                              chunk_size=30,
                              lookback=50,
                              noise_props=[1],
                              plot=False, 
                              pos_ratio=1/3)
    
    # variable_change_PL_list.append(mean_rev.LSTM_PL_curve_df.iloc[:,0])
    # variable_change_cor_list.append(np.mean(mean_rev.rank_cor_list))
    print(variable)
#%%

plt.figure(figsize=(20,10))
plt.title('Simulated Beta Series with Different of Variance Levels', fontsize=32)
plt.xlabel('Date Index', fontsize=30)
plt.ylabel('Beta Value', fontsize=30)

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

labels = ['Variance: {}'.format(k) for k in [0.06, 0.12, 0.18, 0.36]]
# colors = []

for i, df in enumerate(variable_change_PL_list):

    plt.plot(df[5000:], label=labels[i], linewidth=3)
    
plt.legend(fontsize=22)


    #%% CHANGING Beta Sigma BEHAVIOUR beta plots


variable_change_PL_list = []
variable_change_cor_list = []

plt.figure(figsize=(20,10))
 # [0.06, 0.12, 0.18, 0.24, 0.3, 0.36]
for variable in [0.06, 0.12, 0.18, 0.36]:

    cur_ret = high_freq_model.linear_model(num_obs=10000, 
                                           num_covariates=1, 
                                           beta_type='cor_bb', 
                                           beta_sigma=variable, 
                                           noise=0.01,
                                           t=0.55)
    
    
    variable_change_PL_list.append(high_freq_model.params.iloc[:,0])
    # variable_change_cor_list.append(np.mean(mean_rev.rank_cor_list))
    print(variable)



#%%


plt.figure(figsize=(20,10))
plt.title('Simulated Beta Series with Different of Variance Levels', fontsize=32)
plt.xlabel('Date Index', fontsize=30)
plt.ylabel('Beta Value', fontsize=30)

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

labels = ['Variance: {}'.format(k) for k in [0.06, 0.12, 0.18, 0.36]]
# colors = []

for i, df in enumerate(variable_change_PL_list):

    plt.plot(df[5000:], label=labels[i], linewidth=3)
    
plt.legend(fontsize=22)

    #%% CHANGING Beta Sigma BEHAVIOUR beta plots


variable_change_PL_list = []
variable_change_cor_list = []

plt.figure(figsize=(20,10))
 # [0.06, 0.12, 0.18, 0.24, 0.3, 0.36]
for variable in [0.06, 0.12, 0.18, 0.36]:

    cur_ret = high_freq_model.linear_model(num_obs=10000, 
                                           num_covariates=1, 
                                           beta_type='cor_bb', 
                                           beta_sigma=variable, 
                                           noise=0.01,
                                           t=0.55)
    
    
    variable_change_PL_list.append(hgih_freq_model.beta_df.iloc[:,0])
    # variable_change_cor_list.append(np.mean(mean_rev.rank_cor_list))
    print(variable)
#%%
    
# plt.figure(figsize=(20,10))
# plt.title('RLR Performance for Different Noise Ratios', fontsize=32)
# plt.xlabel('Beta', fontsize=30)
# plt.ylabel('Avg Correlation Coefficient', fontsize=30)

# plt.xticks(fontsize=22)
# plt.yticks(fontsize=22)

# plt.plot(variable_change_cor_list, linewidth=1)


plt.figure(figsize=(20,10))
plt.title('RLR Performance for Different Noise Ratios', fontsize=32)
plt.xlabel('Date Index', fontsize=30)
plt.ylabel('Profit/Loss Value', fontsize=30)

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

labels=['Signals on 0% of Noise',
        'Signals on 50% of Noise',
        'Signals on 100% of Noise',
        'Signals on 2000% of Noise',
        'Random Signals']

for i, df in enumerate(beta_var_ls[:4] + [beta_var_ls[-1]]):
    
    plt.plot(df.iloc[5000:], label=labels[i])
    
plt.legend(fontsize=22)












































#%%
# plt.figure(figsize=(20,10))

# for commod in high_freq_model.covariates()['Noisy']:
#     plt.plot(np.exp([i for i in high_freq_model.covariates()['True'][commod]]).cumprod())

# plt.plot()
#%%
# from models import model_generator
# from trading_strats import MeanReversion
# profit_loss_vec =[]

# def avg_back_test(n):
    
#     for i in range(n):
#         high_freq_model = model_generator()
#         cur_ret = high_freq_model.linear_model(num_obs=10000, num_covariates=30, beta_type='cor_bb', beta_sigma=0.0035, noise=0.0001, t=0.6)
#         betas = high_freq_model.params
#         noisy_covs = high_freq_model.covariates()['Noisy']
#         true_covs = high_freq_model.covariates()['True']
#         noise = high_freq_model.noise
         
#         mean_rev = MeanReversion()
#         test = mean_rev.back_test(cur_ret, true_covs, noisy_covs, chunk_size = 25, lookback = 120, noise_props=[1])
#         profit_loss_vec.append(mean_rev.PL_curve_df["100.0%"].iloc[-1])
    
#         print("Test {}/{}".format(i+1, n))
        
#     print("Average Profit: {:.2f}".format(np.mean(profit_loss_vec)))
#     print("Standard Deviation of Profit: {:.2f}".format(np.std(profit_loss_vec)))
    
# avg_back_test(2)
# # high_freq_model.beta_plot()
# # mean_rev.Residuals()
#%%
# from plotting_functions import pred_truth_vis
# print(len(high_freq_model.covariates()['Noisy'][1][600:]),len([i for i in mean_rev.residuals_df[2].dropna()]))
pred_truth_vis(high_freq_model.covariates()['Noisy'][2][600:-30], [i for i in mean_rev.residuals_df[2].dropna().iloc[30:]])
plt.plot(np.exp([i for i in high_freq_model.covariates()['Noisy'][1][1:]]).cumprod()),plt.plot(np.exp([i for i in mean_rev.pred_series.iloc[121:,1]]).cumprod())
#%%
# # plt.plot(high_freq_model.params[4]), plt.plot(mean_rev.beta_df[4])
# series_plot(high_freq_model.params,'True Beta Values')
# series_plot(mean_rev.beta_df,'Estimated Beta Values')
#%%
# from models import model_generator

# test_model = model_generator()

# model = test_model.linear_model(num_obs=10000,
#                                 num_covariates=1,
#                                 beta_type='bm_std',
#                                 noise=0.0002)
# betas = test_model.params
# covs = test_model.covariates()
# series_plot([covs['Noisy'].cumsum() - test_model.noise, covs['Noisy'].cumsum()],'', legend=True)
# # series_plot(covs['Noisy'].cumsum(),'')
# # test_model.model_plot()
# # test_model.noisy_covariates_plot()
# # test_model.true_covariates_plot()

#%% Simulating Data
# from models import model_generator

# # Define a generative model to simulate 10000 days of data for 1 currency basket and 30 commodities
# model = model_generator()
# sim_currency = model.linear_model(num_obs=10000, num_covariates=1, beta_type='bm_std', noise=0)
# sim_betas = model.params
# sim_commods = model.covariates()['Noisy']
# # model.model_plot()

# next_day_returns = pd.DataFrame(sim_commods[1])
# # pd.plotting.autocorrelation_plot(np.exp(next_day_returns[1].astype(float)).cumprod())


# plt.plot(np.exp(next_day_returns[1].astype(float)).cumprod())
# # series_plot(next_day_returns.cumsum(),'')
# # CBOT_OATS_df = pd.read_csv('Data/Continuous Futures Series/CBOT Rough Rice.csv',
# #                            index_col = 0, skiprows = 0, skipfooter = 1, header = 1, engine = 'python')
# # next_day_returns = pd.DataFrame(CBOT_OATS_df['Close'].diff(1))


#%%

# from models import model_generator

# # Define a generative model to simulate 10000 days of data for 1 currency basket and 30 commodities
# model = model_generator()
# sim_currency = model.linear_model(num_obs=10000, 
#                                        num_covariates=2, 
#                                        beta_type='cor_bb', 
#                                        beta_sigma=0.0035, 
#                                        noise=0.0001,
#                                        t=0.6)
# sim_betas = model.params
# sim_commods = model.covariates()['Noisy']
# # model.model_plot()

# next_day_returns = pd.DataFrame(sim_commods[1])
# #%%
# from rolling_functions import LSTM_predictor

# lstm_class = LSTM_predictor()
# lstm_class.train(next_day_returns[1:], sim_currency[1:], 120)
# lstm_class.test()


#%%
# from beta_functions import beta_generator

# beta_plotter = beta_generator('cor_bb', 10001, 27, t=0.55)
# beta_series = beta_plotter()

# series_plot(beta_series, '27 Sample Paths of Correlated Brownian Bridge Beta Series, q=0.55', fontsize=27, lw=0.1)
