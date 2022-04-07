

###########
import numpy as np
import matplotlib.pyplot as plt
from numpy import array, dot, mean, std, empty, argsort
from numpy.linalg import eigh

myarray=[]
with open('./201224_02/w2.csv') as f:
    lines=f.readlines()
    for line in lines:
        myarray.append(line)

myarray = np.asarray(myarray,dtype=np.float32)
print(myarray)


import statsmodels.api as sm
import statsmodels.formula.api as smf

# Construct the columns for the different powers of x
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

ts = np.load('./ts_201224_02.npy')

ts = np.delete(ts, (0), axis=0)
print('Shape of the TS')
print(ts.shape)

from sklearn.decomposition import PCA

pca = PCA(n_components=5)
pca.fit(ts)
print(pca.components_)
print(pca.explained_variance_ratio_[0])
np.save('./201224_02/pca', pca.components_)

def dig(num):
   return (float(int(num * 10000) / 10000))




idx = 5


pca = PCA(n_components=5)
ts_transformed = pca.fit_transform(ts)
print('transform shape')
print(ts_transformed.shape)

dt=1
starttime = 0
endtime = 8995
steps = int(abs(starttime - endtime) / dt)
time = np.linspace(starttime, endtime, steps)
print(time.shape)



r2=get_r2_statsmodels(ts_transformed.T[idx-1],myarray[:time.shape[0]])
fig, ax = plt.subplots()

ccc=['blue','red','green','brown','purple']
#plt.plot(time,smoothing(ts_transformed.T[0],50),'blue',label='pc1-'+str(pca.explained_variance_ratio_[0]),markersize=3)
ax.plot(time,smoothing(ts_transformed.T[idx-1],50),ccc[idx-1],label='ER'+str(float (int (pca.explained_variance_ratio_[idx-1] * 1000) / 1000))+' SD-'+str(dig(np.std(ts_transformed.T[idx-1]))),markersize=3)

#plt.plot(time,smoothing(pca.components_[1],100),'red',label='pc2-'+str(pca.explained_variance_ratio_[1]),markersize=3)
#plt.plot(time,smoothing(pca.components_[2],100),'green',label='pc3-'+str(pca.explained_variance_ratio_[2]),markersize=3)
#plt.plot(time,smoothing(pca.components_[3]),'orange',label='pc2-'+str(pca.explained_variance_ratio_[3])',markersize=3)


#plt.title('Before stress: PC (smoothing =50)')
ax.set_xlabel('times',fontsize=14)
ax.set_ylabel('pc',fontsize=14)

ax2=ax.twinx()
ax2.plot(time, smoothing(myarray[:time.shape[0]],50),color="black",label='whisking',alpha=0.5,markersize=3)
ax2.set_ylabel("magnitude",fontsize=14)
fig.legend()
plt.title('PC'+str(idx)+'-'' R2='+str(float (int (r2 * 10000000) / 10000000))+' Imm After Injection')
plt.show()


fig.savefig('./201224_02/pc'+str(idx)+'.png')
plt.show()