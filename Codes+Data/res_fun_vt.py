"""
@author: Alfredo Cinelli
FUNCTION FOR MEAN-VARIANCE PORTFOLIO OPTIMIZATION VIA MC SIMULATION
Objective: Mean Return Maximization
Target: Portfolio (annual) Volatility
"""
# Import relevant packages

import pandas as pd
import numpy as np
import scipy as sp

# Define functions

def port_return(x,r,annual):
    'x: weights of the portfolio'
    'r: returns of the assets'
    'annual: time rescaling parameter (e.g. 12 if daily data, ecc.)'
    
    mu=r.mean(axis=0) # mean over columns
    return -(x.T@mu)*annual # negate for optimization purposes

def mc_opt(ast,E,V,sz,annual,nsim,var_tg):
    'ast: list of asset tickers'
    'E: vector of mean expected returns'
    'V: variance-covariance matrix of returns'
    'sz: size of the simulate paths'
    'annual: time unit in year basis (e.g. 252 if daily, 12 if monthly, etc.)'
    'nsim: number of simulation for each target return'
    'var_tg: portfolios target volatility'
    
    x0=np.ones(len(E))*(1/len(E)) # Equally weighted starting 
    nsim=np.arange(1,nsim+1,1) # number of simulations stream
    lb,ub=0,None # lower and upper bounds (no short sale allowed)
    bnd=[(lb,ub) for i in range(len(E))] # list of tuple of bounds
    var_tg=np.sort(var_tg) # sort volatiltiy in ascending order
    W=pd.DataFrame(index=ast,columns=var_tg) # DataFrame to store the portfolio weights
    W_tmp=pd.DataFrame(index=ast,columns=nsim)

    for va_0 in var_tg:
        for i in nsim:
            print('Simulation '+str(i)+' for target volatility '+str(np.around(va_0*100,2))+'%')
            r_s=pd.DataFrame(np.random.multivariate_normal(E,V,sz)) # simulated Multivariate Gaussian returns
            mu=r_s.mean(axis=0) # mean over columns
            S=r_s.cov() # VC matrix
            cons_MV=({'type':'eq','fun':lambda x: sum(x)-1},{'type':'eq','fun':lambda x: ((x.T@S@x)*annual)**0.5-va_0}) # constraints
            res=sp.optimize.minimize(port_return,x0,method='SLSQP',args=(r_s,annual),bounds=bnd,constraints=cons_MV,options={'disp': False}) # optimization routine 
            W_tmp.iloc[:,i-1]=res.x 
        W.loc[:,va_0]=W_tmp.mean(axis=1) # portfolio weights for each target volatility
        
    idx=['p_'+str(i+1) for i in range(len(var_tg))] # list of indexes
    W.columns=idx # change the columns of the portfolios
    return W