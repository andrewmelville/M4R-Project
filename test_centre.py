# This is where I will bring together all the modules of my code project for integrated testing

# 15/10/20 Andrew Melville
from brownian_motion import walk_generator
from plotting_functions import series_plot
from beta_functions import beta_generator
from models import model_generator
from plotting_functions import series_plot
from rolling_functions import Rolling_LR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
from brownian_motion import walk_generator
brown = walk_generator(n=10000, d=100)


series_plot(brown, 'Random Walk Plot', xlab = 'Value', ylab= 'T')

#%%
from beta_functions import beta_generator
test =  beta_generator()
#%%
from models import linear_model_generator
test = linear_model_generator()
#%%
import pandas as pd
import numpy as np
test = pd.DataFrame(np.ones((2,2)) + [[1,2],[3,4]]) * [[1,1],[1,1]]
print(test.values.sum(axis = 1))


#%%
from models import model_generator
test_model = model_generator()

model = test_model.linear_model(num_obs = 10000, num_covariates = 1, beta_type = 'sin_correlated')
betas = test_model.params
covs = test_model.covariates

#%%

from rolling_functions import Rolling_LR

reg = Rolling_LR()

reg.fit(model, covs, 20)

#%%
from plotting_functions import rolling_beta_plot

rolling_beta_plot(covs, betas, reg.coefficients(), model, 20, 'True_Est_Betas')

#%%or
from models import model_generator
high_freq_model = model_generator()

model = high_freq_model.linear_model(num_obs = 10000, num_covariates = 1, beta_type = 'high_freq')
betas = high_freq_model.params
covs = high_freq_model.covariates

#%%
from rolling_functions import Rolling_LR

reg = Rolling_LR()

reg.fit(model, covs, 20)
