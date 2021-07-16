"""
@author: Alfredo Cinelli

RESAMPLING PORTFOLIO OPTIMIZATION

"""

#%% Import packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import statsmodels.api as sm
from arch import arch_model
from arch.unitroot import PhillipsPerron

#%% Define some functions

' Define some Functions '
' Functions are written for pd.Series and extended to pd.DataFrames '

# Average Annualized Return

def annualize_rets_Series(s, periods_per_year):
    '''
    Computes the return per year, or, annualized return.
    The variable periods_per_year can be, e.g., 12, 52, 252, in 
    case of monthly, weekly, and daily data.
    '''
    s = s.dropna()
    growth = (1 + s).prod()
    n_period_growth = s.shape[0]
    return growth**(periods_per_year/n_period_growth) - 1

def annualize_rets(s, periods_per_year):
    '''
    Computes the return per year, or, annualized return.
    The variable periods_per_year can be, e.g., 12, 52, 252, in 
    case of monthly, weekly, and daily data.
    The method takes in input either a DataFrame or a Series, 
    in the former case, it computes the annualized return for every column (Series) by using pd.aggregate
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate(annualize_rets_Series, periods_per_year=periods_per_year)
    elif isinstance(s, pd.Series):
        return annualize_rets_Series(s, periods_per_year)

# Annualized Volatility

def annualize_vol_Series(s, periods_per_year):
    '''
    Computes the volatility per year, or, annualized volatility.
    The variable periods_per_year can be, e.g., 12, 52, 252, in 
    case of monthly, weekly, and daily data.
    '''
    s = s.dropna()   
    return s.std() * (periods_per_year)**(0.5)

def annualize_vol(s, periods_per_year):
    '''
    Computes the volatility per year, or, annualized volatility.
    The variable periods_per_year can be, e.g., 12, 52, 252, in 
    case of monthly, weekly, and daily data.
    The method takes in input either a DataFrame, or a Series
    In the former case, it computes the annualized volatility of every column 
    (Series) by using pd.aggregate. 
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate(annualize_vol_Series, periods_per_year=periods_per_year)
    elif isinstance(s, pd.Series):
        return annualize_vol_Series(s, periods_per_year)
    
# Expected Portfolio Returns 

def port_ret(x, mu, annual):
    'x: weights of the portfolio'
    'mu: returns of the assets '
    return (x.T@mu)*annual


# Portfolio Variance 

def port_variance(x, r, annual):
    'x: weights of the portfolio'
    'r: returns of the assets '
    S = r.cov()
    return (x.T@S@x)*annual

# Portfolio Volatility 

def port_vola(x, r, annual):
    'x: weights of the portfolio'
    'r: returns of the assets '  
    return np.sqrt(port_variance(x, r, annual))

#%% Load and Prepare the data
Data=pd.read_excel('Data_QAA.xlsx',sheet_name='Data')
Data.set_index(Data.loc[:,'Date'],inplace=True)
Data.drop(columns=['Date'],inplace=True)
ast=list(Data.columns) # asset products (representing classes)

#%% Investigate the time series
Data.plot(grid=True,ylabel='$R_t$',title='ASSETS DAILIY RETURNS') # plot the assets daily returns
plt.legend(bbox_to_anchor=(1.1, -0.15), fancybox=True, ncol=2)
for ticker in ast:
    pp=PhillipsPerron(Data[ticker]) # run the PP test for unit-root (H0:Unit root)
    print('Test on '+str(ticker))
    print(pp.summary()) # all the assets returns are stationary
    #sm.qqplot(Data[ticker],fit=True,line='45')
#from Granger import granger
#granger(Data,ast)  # Granger Causality test p-values (H0: No Granger Causality)  
# at 10% all the assets are influencing each other --> using a VAR could be better 
#sb.heatmap(Data,annot=True,cmap='Blues')
#Cr=Data.corr() # assets returns correlation matrix
#%% Fitting the VAR
nobs_in=1000
nobs_out=250
obs_in=nobs_in+nobs_out # last observation to take
obs_out=Data.shape[0]-nobs_out
Data_fit=Data.iloc[-obs_in:obs_out,:] # data on which fit the models 
Data_test=Data.iloc[obs_out:,:] # data on which test the portfolios
mdl=sm.tsa.VAR(Data_fit) # set the VAR model
lags=np.arange(0,16,1) # stream of lags
ic=pd.DataFrame(index=lags,columns=['AIC','BIC'])
for p in lags:
    res=mdl.fit(p)
    print('Lag order: ',p)
    print('AIC: ',res.aic)
    print('BIC: ',res.bic)
    ic.iloc[p,0],ic.iloc[p,1]=res.aic,res.bic
    
ic.plot(grid=True,xlabel='Lag',title='INFORMATION CRITERIA VALUES') 
# Using a VAR(2 or 1) if BIC or VAR(6) is AIC
#%% Fit the optimal VAR and take the forecasts
mdl=sm.tsa.VAR(Data_fit)
l=1 # VAR lag order
res=mdl.fit(l)
print(res.summary())
nobs=1 # number of the ahead forecast (one year)
forc_var=res.forecast(Data.values[-l:],steps=nobs)
E=pd.Series(forc_var.reshape(forc_var.shape[1]),index=ast,name='Daily forc')# forecasted returns
#%% Fit the DCC-GARCH and date the forecasts
import mgarch # proprietary library for DCC
mdl=mgarch.mgarch() # specify the DCC with Gaussian distribution
mdl.fit(Data_fit.iloc[-500:,:]) # fit on half of the observations of the VAR
V=mdl.predict(nobs)['cov']
V=pd.DataFrame(V,index=ast,columns=ast) # Forecasted daily Variance-Covariance matrix 
#%% Computation of the Resampled Porfolios
nsim=1000 # number of simulations
sz=30 # size of each simulation (the lower the lower the confidence in forecasts) --> Mid Confidence
annual=252 # depending on the frequency of the data
var_tg=np.array((0.02,0.05,0.10,0.15,0.20,0.25)) # target annual volatilities
nport=len(var_tg) # number of portfolios

from res_fun_vt import mc_opt # import the function from the library (home made)
W=mc_opt(ast,E,V,sz,annual,nsim,var_tg) # run the Resampling
#%% Plot the results
path='/Users/alfredo/Desktop/CC_Materiale/Pomante/Exam_QAA/'
ax=W.transpose().plot(kind='area',title='COMPOSITION OF RESAMPLED PORTFOLIOS',figsize=(10,6))
plt.legend(bbox_to_anchor=(1, -0.1), fancybox=True,ncol=3)
#plt.savefig(path+'Resampled_Portfolios1'+'.pdf',bbox_inches='tight')
#W.to_excel('Res.xlsx')
#%% Portfolios performance Out-of-Sample in Turbulent period
R=Data_test@W # porfolios returns
print('Annualized Percentage Returns: '+str(np.around(annualize_rets(R,annual)*100,2)))
print('Annualized Percentage Volatility: '+str(np.around(annualize_vol(R,annual)*100,2)))
f=R.plot(title='PORTFOLIOS RETURNS',ylabel='$R_t$',grid=True)
plt.legend(bbox_to_anchor=(1, -0.2), fancybox=True,ncol=5)
#plt.savefig(path+'Resampled_Portfolios_R1'+'.pdf',bbox_inches='tight')
plt.show()
plt.close()
f=(1+R).cumprod().plot(title='PORTFOLIOS CUMULATIVE RETURNS',ylabel='$R_t^C$',grid=True)
plt.legend(bbox_to_anchor=(1, -0.2), fancybox=True,ncol=5)
#plt.savefig(path+'Resampled_Portfolios_RC1'+'.pdf',bbox_inches='tight')
plt.show()
plt.close()
S=np.around(annualize_vol(R,252)*100,2) # porfolios percentage volatility
R_m=np.around(annualize_rets(R,annual)*100,2)
ax=R_m.plot(kind='bar',title='PORTFOLIOS ANNUALIZED % RETURN',grid=True)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
#plt.savefig(path+'Resampled_Portfolios_HM1'+'.pdf',bbox_inches='tight')
plt.show()
plt.close()
ax=S.plot(kind='bar',title='PORTFOLIOS ANNUALIZED % VOLATILITY',grid=True)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
#plt.savefig(path+'Resampled_Portfolios_HV1'+'.pdf',bbox_inches='tight')
plt.show()
plt.close()

(W*100).transpose().plot(kind='bar',ylabel='$\% W_i$',title='PERCENTAGE WEIGHTS FOR PROFILES',xlabel='Profile',grid=True)
plt.legend(bbox_to_anchor=(1.1, -0.2), fancybox=True,ncol=2)
#plt.savefig(path+'Resampled_Portfolios_Bar'+'.pdf',bbox_inches='tight')
plt.show()
plt.close()




''' ------------------------------ COMMENT -------------------------------------
 The Monte Carlo Resampling as it can be seen from the area plot is able to 
 give back smoother transitions when going through different target returns
 and moreover is able to generate way more diversified portfolios compared
 to plain Markowitz model (leading to less overall risk).
 The MC resampling moreover is able to strongly reduce, by nature, the flaw of
 the Markowitz model of being prone to error in the sense of being error maxi-
 mizer. Therefore the resampling increases the robustness of the optimization
 model.
'''

#%% #%% Herfindhal Index

def HHI(s):
    '''
    Computes the Herndahl Index over a series. 
    The variable s represent the dataset use to calcualte the Herndahl Index 
    '''    
    s = s**2
    return np.sum(s)

HI=pd.DataFrame(columns=['Herfindhal Index'],index=W.columns)
for i in range(len(var_tg)):
    p=W.values[:,i]
    HI.iloc[i]=HHI(p)
    
HI.plot(kind='bar',title='HHI FOR THE VARIOUS PROFILES',ylabel='$HHI_i$',grid=True)
plt.savefig(path+'HHI'+'.pdf',bbox_inches='tight')
plt.show()
plt.close()  