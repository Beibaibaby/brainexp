import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import statsmodels.api as sm

import csv

def get_r2_statsmodels(x, y, k=1):
    xpoly = np.column_stack([x**i for i in range(k+1)])
    return sm.OLS(y, xpoly).fit().rsquared

i=5
abss = np.load('/Users/dragon/Desktop/brainexp/pca/series/beforestress/pca.npy')
abset= np.load('/Users/dragon/Desktop/brainexp/pca/series/afterstress/pca.npy')
for i in range(5):
    abb = abss[i]
    ab1 = abset[i]
    a, b = np.polyfit(abb, ab1, 1)
    r2=get_r2_statsmodels(abb,ab1)
    # add points to plot
    plt.scatter(abb, ab1)

    # add line of best fit to plot
    plt.plot(abb, a * abb + b)
    # naming the x axis
    plt.xlabel('before stress pc' + str(i+1)+'r2='+ str(r2))
    # naming the y axis
    plt.ylabel('after stress pc' + str(i+1)+'a='+str(a))
    plt.savefig('./com12/fig' + str(i+1))
    plt.show()