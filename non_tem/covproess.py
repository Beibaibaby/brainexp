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


hf = h5py.File('/Users/dragon/Downloads/Research/CogT/gradients/hdf5_files/201217_01_ica_filtered.hdf5', 'r')
print(hf.keys())
filtered = hf.get('filtered').value
print(filtered.shape)
ts=np.asarray(filtered)
ts=ts[:,:321,:]
ts = ts.reshape(*ts.shape[:-2], -1)
print(ts.shape)



print('Shape of the TS')
print(ts.shape)
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca.fit(ts)
print(pca.components_[0])
print(pca.explained_variance_ratio_[0])

pos = [i for i, x in enumerate(pca.components_[0]) if x >= 0]
#pos=np.argmax(pca.components_[0])
neg = [i for i, x in enumerate(pca.components_[0]) if x < 0]
#neg=np.argmax(-pca.components_[0])
pos_ts=ts[:,pos]
pos_ts=np.average(pos_ts, axis=1)
neg_ts=ts[:,neg]
neg_ts=np.average(neg_ts, axis=1)
print(pos)
print(neg)





def get_r2_statsmodels(x, y, k=1):
    xpoly = np.column_stack([x**i for i in range(k+1)])
    return sm.OLS(y, xpoly).fit().rsquared


def slided(data):
    timestep = 5 * 10 ** -3
    slided_data = []
    for (index, value) in enumerate(data):
        if index % int(timestep / dt) == 0:
            slided_data.append(value)
    return slided_data


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
pca = PCA(n_components=5)
ts_transformed = pca.fit_transform(ts)
#np.save('./beforestress/pca', pca.components_)

ccc = ['blue', 'red', 'green', 'brown', 'purple']
dt = 1
starttime = 0
endtime = 9000
steps = int(abs(starttime - endtime) / dt)
time = np.linspace(starttime, endtime, steps)

ccc = ['blue', 'red', 'green', 'brown', 'purple']
for idx in [1,2,3,4,5]:

    r2 = get_r2_statsmodels(ts_transformed.T[idx - 1], myarray[:time.shape[0]])
    fig, ax = plt.subplots()
    a = [22, 266, 512, 1561, 2237, 2959, 3393, 3580, 3760, 5133, 5565]
    b = [117, 355, 602, 1628, 2389, 3010, 3475, 3633, 3827, 5284, 5692]
    for i in range(len(a)):
        ax.axvspan(a[i], b[i], alpha=0.3, color='green')


    # plt.plot(time,smoothing(ts_transformed.T[0],50),'blue',label='pc1-'+str(pca.explained_variance_ratio_[0]),markersize=3)
    ax.plot(time, smoothing(ts_transformed.T[idx - 1], 50), ccc[idx - 1],
            label='ER' + str(float(int(pca.explained_variance_ratio_[idx - 1] * 1000) / 1000)) + ' SD-' + str(
                dig(np.std(ts_transformed.T[idx - 1]))), markersize=3)

    # plt.plot(time,smoothing(pca.components_[1],100),'red',label='pc2-'+str(pca.explained_variance_ratio_[1]),markersize=3)
    # plt.plot(time,smoothing(pca.components_[2],100),'green',label='pc3-'+str(pca.explained_variance_ratio_[2]),markersize=3)
    # plt.plot(time,smoothing(pca.components_[3]),'orange',label='pc2-'+str(pca.explained_variance_ratio_[3])',markersize=3)

    # plt.title('Before stress: PC (smoothing =50)')
    ax.set_xlabel('times', fontsize=14)
    ax.set_ylabel('pc', fontsize=14)

    ax2 = ax.twinx()
    ax2.plot(time, smoothing(myarray[:time.shape[0]], 50), color="black", label='whisking', alpha=0.5, markersize=3)
    ax2.set_ylabel("magnitude", fontsize=14)
    fig.legend()
    plt.title('PC' + str(idx) + '-Before Stress R2=' + str(float(int(r2 * 1000) / 1000)))

    plt.savefig('./beforestress/pc' + str(idx) + '.png')
    plt.show()
