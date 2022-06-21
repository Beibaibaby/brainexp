

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

w = np.load('./reg.coef_.npy')

for i in range(10000):
    print(w[i])
print(w.shape)
w=w.reshape([321,481])
print(w.shape)
plt.rcParams["figure.figsize"] = (5, 8)
w=np.flip(w.T,axis=0)
plt.imshow(w)
plt.colorbar()

plt.show()