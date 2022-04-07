import numpy as np
import matplotlib.pyplot as plt
from numpy import array, dot, mean, std, empty, argsort
from numpy.linalg import eigh
import statsmodels.api as sm
import statsmodels.formula.api as smf

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



ts = np.load('/Users/dragon/Desktop/brainexp/pro_data/ts_201224_02.npy')
ts = np.delete(ts, (0), axis=0)
print('Shape of the TS')
print(ts.shape)
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca.fit(ts)
print(pca.components_)
print(pca.explained_variance_ratio_[0])


myarray=[]
with open('/Users/dragon/Desktop/brainexp/behav/w3.csv') as f:
    lines=f.readlines()
    for line in lines:
        myarray.append(line)

myarray = np.asarray(myarray,dtype=np.float32)

pca = PCA(n_components=5)
ts_transformed = pca.fit_transform(ts)
np.save('./series/injection/pca', pca.components_)
# print('transform shape')
# print(ts_transformed.shape)
ccc = ['blue', 'red', 'green', 'brown', 'purple']
dt = 1
starttime = 0
endtime = 8995
steps = int(abs(starttime - endtime) / dt)
time = np.linspace(starttime, endtime, steps)
print(time.shape)
ccc = ['blue', 'red', 'green', 'brown', 'purple']
a = [0,480,848,1596,2394,2912,3842,3955,4581,5923,6071,6616,6667,7126,7214,7879,8055,8577]
b = [126,549,893,1679,2432,3061,3855,4014,4609,6015,6084,6656,6680,7167,7260,8014,8075,8617]

for idx in [1,2,3,4,5]:
    r2 = get_r2_statsmodels(ts_transformed.T[idx - 1], myarray[:time.shape[0]])
    fig, ax = plt.subplots()

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
    plt.title('PC' + str(idx) + '-After Injection R2=' + str(float(int(r2 * 1000) / 1000)))

    plt.savefig('./series/injection/pc' + str(idx) + '.png')
    plt.show()

