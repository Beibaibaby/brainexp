import numpy as np
import matplotlib.pyplot as plt
from numpy import array, dot, mean, std, empty, argsort
from numpy.linalg import eigh

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



ts = np.load('./ts_201224_01.npy')
ts = np.delete(ts, (0), axis=0)
print('Shape of the TS')
print(ts.shape)
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca.fit(ts)
print(pca.components_)
print(pca.explained_variance_ratio_[0])



fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.scatter(pca.components_[0],  pca.components_[1], c = 'r')

axes.set_xlabel('x1')
axes.set_ylabel('x2')
axes.set_xlim((-1, 1))
axes.set_xticks(np.arange(-1, 1, step=0.2))
axes.set_ylim((-1, 1))
axes.set_yticks(np.arange(-1, 1, step=0.2))
plt.show()
fig.savefig('pca.png')

idx=1
pca = PCA(n_components=5)
ts_transformed = pca.fit_transform(ts)
np.save('./201224_01/pca', pca.components_)
print('transform shape')
print(ts_transformed.shape)
ccc=['blue','red','green','brown','purple']
dt=1
starttime = 0
endtime = 8995
steps = int(abs(starttime - endtime) / dt)
time = np.linspace(starttime, endtime, steps)
print(time.shape)

fig, ax = plt.subplots()
a=[22,266,512,1561,2237,2959,3393,3580,3760,5133,5565]
b=[117,355,602,1628,2389,3010,3475,3633,3827,5284,5692]
for i in range(len(a)):
 ax.axvspan(a[i], b[i], alpha=0.3, color='green')

#plt.plot(time,smoothing(ts_transformed.T[0],50),'blue',label='pc1-'+str(pca.explained_variance_ratio_[0]),markersize=3)
plt.plot(time,smoothing(ts_transformed.T[idx-1],50),ccc[idx-1],label='ER-'+str(dig(pca.explained_variance_ratio_[idx-1]))+',std'+str(dig(np.std(ts_transformed.T[idx-1]))),markersize=3)
#plt.plot(time,smoothing(pca.components_[1],100),'red',label='pc2-'+str(pca.explained_variance_ratio_[1]),markersize=3)
#plt.plot(time,smoothing(pca.components_[2],100),'green',label='pc3-'+str(pca.explained_variance_ratio_[2]),markersize=3)
#plt.plot(time,smoothing(pca.components_[3]),'orange',label='pc2-'+str(pca.explained_variance_ratio_[3])',markersize=3)


plt.title('After 7 days of stress: PC'+str(idx))
plt.xlabel('times')
plt.ylabel('pc')
plt.legend()


plt.savefig('./201224_01/pc'+str(idx)+'.png')
plt.show()