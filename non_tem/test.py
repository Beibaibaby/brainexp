
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy import array, dot, mean, std, empty, argsort
from numpy.linalg import eigh
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA





def get_r2_statsmodels(x, y, k=1):
    xpoly = np.column_stack([x**i for i in range(k+1)])
    return sm.OLS(y, xpoly).fit().rsquared





def smoothing(data,w):
    dt=1
    length = len(data)
    smoothed_data = np.zeros(length)
    width = int(w / dt)
    for i in range(length):
        if length - (i + 1) < width:
            smoothed_data[i] = np.average(data[i:])
        else:
            smoothed_data[i] = np.average(data[i:i + width])
    return smoothed_data

def dig(num):
   return (float(int(num * 10000) / 10000))




myarray=[]
with open('/Users/dragon/Desktop/brainexp/behav/w1.csv') as f:
    lines=f.readlines()
    for line in lines:
        myarray.append(line)

myarray = np.asarray(myarray,dtype=np.float32)



pca = PCA(n_components=4)




ts_pca = np.load('./beforestress/pca.npy')
pc1=ts_pca[0]
print(pc1.shape)
pc1=pc1.reshape([321,492])
print(pc1.shape)
plt.rcParams["figure.figsize"] = (5, 8)
pc1=np.flip(pc1.T,axis=0)
plt.imshow(pc1)
plt.colorbar()

plt.show()