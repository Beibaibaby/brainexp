

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from numpy import array, dot, mean, std, empty, argsort
from numpy.linalg import eigh
import statsmodels.api as sm
import statsmodels.formula.api as smf
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
import numpy as np
import matplotlib.pyplot as plt
from numpy import array, dot, mean, std, empty, argsort
from numpy.linalg import eigh
import statsmodels.api as sm
import statsmodels.formula.api as smf

#Function to get R2
def get_r2_statsmodels(x, y, k=1):
    xpoly = np.column_stack([x**i for i in range(k+1)])
    return sm.OLS(y, xpoly).fit().rsquared
dt=1
#Function for sliding
def slided(data):
    timestep = 5 * 10 ** -3
    slided_data = []
    for (index, value) in enumerate(data):
        if index % int(timestep / dt) == 0:
            slided_data.append(value)
    return slided_data

#Function for smoothing with weight
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

#keep number.xxxx
def dig(num):
   return (float(int(num * 10000) / 10000))



hf = h5py.File('/Users/dragon/Downloads/Research/CogT/gradients/hdf5_files/201217_01_ica_filtered.hdf5', 'r')
print(hf.keys())
filtered = hf.get('filtered').value
print(filtered.shape)
ts=np.asarray(filtered)
ts=ts[:,:321,:481]
ts = ts.reshape(*ts.shape[:-2], -1)
ts = np.delete(ts, (0), axis=0)

hf1 = h5py.File('/Users/dragon/Downloads/Research/CogT/gradients/hdf5_files/201224_01_ica_filtered.hdf5', 'r')
print(hf1.keys())
filtered = hf1.get('filtered').value
print(filtered.shape)
ts1=np.asarray(filtered)
ts1=ts1[:,:321,:481]
ts1 = ts1.reshape(*ts1.shape[:-2], -1)
ts1 = np.delete(ts1, (0), axis=0)


hf2 = h5py.File('/Users/dragon/Downloads/Research/CogT/gradients/hdf5_files/201224_02_ica_filtered.hdf5', 'r')
print(hf2.keys())
filtered = hf2.get('filtered').value
print(filtered.shape)
ts2=np.asarray(filtered)
ts2=ts2[:,:321,:481]
ts2 = ts2.reshape(*ts2.shape[:-2], -1)
ts2 = np.delete(ts2, (0), axis=0)

hf3 = h5py.File('/Users/dragon/Downloads/Research/CogT/gradients/hdf5_files/201225_01_ica_filtered.hdf5', 'r')
print(hf3.keys())
filtered = hf3.get('filtered').value
print(filtered.shape)
ts3=np.asarray(filtered)
ts3=ts3[:,:321,:481]
ts3 = ts3.reshape(*ts3.shape[:-2], -1)
ts3 = np.delete(ts3, (0), axis=0)




ts=np.concatenate((ts, ts1,ts2,ts3), axis=0)



#neg=[]
#pos=[]

print('Shape of the TS')
print(ts.shape)


myarray=[]
with open('/Users/dragon/Desktop/brainexp/behav/w1.csv') as f:
    lines=f.readlines()
    for line in lines:
        myarray.append(line)


with open('/Users/dragon/Desktop/brainexp/behav/w2.csv') as f:
    lines=f.readlines()
    for line in lines:
        myarray.append(line)

with open('/Users/dragon/Desktop/brainexp/behav/w3.csv') as f:
    lines=f.readlines()
    for line in lines:
        myarray.append(line)

with open('/Users/dragon/Desktop/brainexp/behav/w4.csv') as f:
    lines=f.readlines()
    for line in lines:
        myarray.append(line)

wishking = np.asarray(myarray,dtype=np.float32)


from sklearn.linear_model import Lasso
print('start')
reg = Lasso(alpha=0.0001)
wishking = wishking[:35989]
print(ts.shape)
print(wishking.shape)


reg.fit(ts, wishking)
print('R squared training set', round(reg.score(ts, wishking)*100, 2))
print('R squared test set', round(reg.score(ts, wishking)*100, 2))
np.save('reg.coef_',reg.coef_)