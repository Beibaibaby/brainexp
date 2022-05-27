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

myarray=[]

with open('/Users/dragon/Desktop/brainexp/behav/w1.csv') as f:
    lines=f.readlines()
    for line in lines:
        myarray.append(line)


myarray = np.asarray(myarray,dtype=np.float32)


ccc = ['blue', 'red', 'green', 'brown', 'purple']
dt = 1
starttime = 0
endtime =9000
steps = int(abs(starttime - endtime) / dt)
time = np.linspace(starttime, endtime, steps)
ccc = ['blue', 'red', 'green', 'brown', 'purple']
a = np.asarray([22, 266, 512, 1561, 2237, 2959, 3393, 3580, 3760, 5133, 5565])
b = np.asarray([117, 355, 602, 1628, 2389, 3010, 3475, 3633, 3827, 5284, 5692])
pos_ts_gradient=np.gradient(smoothing(pos_ts,100))
pos_ts_gradient=pos_ts_gradient*15

neg_ts_gradient=np.gradient(smoothing(neg_ts,100))
neg_ts_gradient=neg_ts_gradient*15
sum_der_pos = 0
sum_der_pos_pre =0
sum_der_pos_after=0
nor=0
for i in range(len(a)):
    for j in range(b[i]-a[i]):

        sum_der_pos+=pos_ts_gradient[a[i]+j]
        if a[i]-j>=0:
          sum_der_pos_pre +=pos_ts_gradient[a[i]-j]
        sum_der_pos_after+=pos_ts_gradient[b[i]+j]
        nor+=1

avg_der_pos=15*sum_der_pos/nor
avg_der_pos_pre=15*sum_der_pos_pre/nor
avg_der_pos_after=15*sum_der_pos_after/nor

sum_der_neg=0
sum_der_neg_pre =0
sum_der_neg_after=0

for i in range(len(a)):
    for j in range(b[i]-a[i]):
        sum_der_neg+=neg_ts_gradient[a[i]+j]
        if a[i]-j>=0:
          sum_der_neg_pre +=neg_ts_gradient[a[i]-j]
        sum_der_neg_after+=neg_ts_gradient[b[i]+j]
        nor+=1

avg_der_neg=15*sum_der_neg/nor
avg_der_neg_pre=15*sum_der_neg_pre/nor
avg_der_neg_after=15*sum_der_neg_after/nor
print(avg_der_pos)
print('pre',avg_der_pos_pre)

print(avg_der_neg)


fig, ax = plt.subplots()
fig.set_size_inches(15.5, 10.5)
x = ['Pos_Pre_Run','Neg_Pre_Run','Pos_Run', 'Neg_Run','Pos_After_Run','Neg_After_Run', 'Pos_all', 'Neg_all']
y = [avg_der_pos_pre,avg_der_neg_pre,avg_der_pos,avg_der_neg,avg_der_pos_after,avg_der_neg_after,15*np.average(pos_ts_gradient),15*np.average(neg_ts_gradient)]
ax.bar(x,y,color=['b','r','b','r','b','r','b','r'])
plt.savefig('der_compare')
plt.show()



print(15*np.average(pos_ts_gradient))
print(15*np.average(neg_ts_gradient))

np.save('pos_ts_gradient.npy',pos_ts_gradient)
np.save('neg_ts_gradient.npy',neg_ts_gradient)

for idx in [1]:

    fig, ax = plt.subplots()
    fig.set_size_inches(38.5, 10.5)
    r2_p = get_r2_statsmodels(pos_ts, myarray[:time.shape[0]])
    r2_n= get_r2_statsmodels(neg_ts, myarray[:time.shape[0]])

    for i in range(len(a)):
        ax.axvspan(a[i], b[i], alpha=0.3, color='green')


    # plt.plot(time,smoothing(ts_transformed.T[0],50),'blue',label='pc1-'+str(pca.explained_variance_ratio_[0]),markersize=3)
    ax.plot(time, smoothing(pos_ts,100), ccc[idx - 1],
            label='pos', markersize=3)
    ax.plot(time, smoothing(neg_ts,100), ccc[idx],
            label='neg', markersize=3)
    #ax.plot(time, smoothing(pos_ts_gradient[:time.shape[0]], 1), color="brown", label='grad', alpha=0.5,         markersize=3)
    # plt.plot(time,smoothing(pca.components_[1],100),'red',label='pc2-'+str(pca.explained_variance_ratio_[1]),markersize=3)
    # plt.plot(time,smoothing(pca.components_[2],100),'green',label='pc3-'+str(pca.explained_variance_ratio_[2]),markersize=3)
    # plt.plot(time,smoothing(pca.components_[3]),'orange',label='pc2-'+str(pca.explained_variance_ratio_[3])',markersize=3)

    # plt.title('Before stress: PC (smoothing =50)')
    ax.set_xlabel('times', fontsize=14)
    ax.set_ylabel('activity', fontsize=14)

    ax2 = ax.twinx()
    ax2.plot(time, smoothing(pos_ts_gradient[:time.shape[0]], 10), color="black", label='grad', alpha=0.9,
             markersize=3)
    ax2.plot(time, np.zeros(time.shape), color="yellow", alpha=0.5,
             markersize=3)
    ax2.set_ylabel("magnitude", fontsize=14)

    fig.legend()
    plt.title('Activity (Pick based on pc1)'+' pr2='+str(float(int(r2_p * 1000) / 1000)) +' nr2='+str(float(int(r2_n * 1000) / 1000)) )

    plt.savefig('./pn_der_average' + str(idx) + '.png')
    plt.show()

