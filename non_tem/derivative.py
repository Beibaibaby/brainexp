


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
neg=[]
pos=[]

print('Shape of the TS')
print(ts.shape)
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca.fit(ts)
print(pca.components_)
print(pca.explained_variance_ratio_[0])
#ts_transformed = pca.fit_transform(ts)
print('start')
#np.save('pcseries', ts_transformed.T[0])

pos = [i for i, x in enumerate(pca.components_[0]) if x >= 0]
#pos=np.argmax(pca.components_[0])
neg = [i for i, x in enumerate(pca.components_[0]) if x < 0]
#neg=np.argmax(-pca.components_[0])
pos_ts=ts[:,pos]
pos_ts=np.average(pos_ts, axis=1)
neg_ts=ts[:,neg]
neg_ts=np.average(neg_ts, axis=1)

np.save('pos_ts.npy',pos_ts)
np.save('neg_ts.npy',neg_ts)



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

myarray = np.asarray(myarray,dtype=np.float32)


ccc = ['blue', 'red', 'green', 'brown', 'purple']
dt = 1
starttime = 0
endtime = 35989
steps = int(abs(starttime - endtime) / dt)
time = np.linspace(starttime, endtime, steps)
#time = time/15
print(time.shape)
ccc = ['blue', 'red', 'green', 'brown', 'purple']
a = np.asarray([22, 266, 512, 1561, 2237, 2959, 3393, 3580, 3760, 5133, 5565])
b = np.asarray([117, 355, 602, 1628, 2389, 3010, 3475, 3633, 3827, 5284, 5692])

a1 = np.asarray([10, 193, 371, 1261, 1488, 1722, 2933, 3097, 3257, 3364, 3745, 4034, 4366, 5279, 5656, 5781, 6066, 6349, 6469,
      7111, 7375, 7711, 7912, 8026, 8194, 8379, 8744])+9000
b1 = np.asarray([51, 207, 442, 1342, 1542, 1764, 2965, 3135, 3281, 3412, 3867, 4098, 4458, 5360, 5757, 5800, 6081, 6406, 6507,
      7200, 7401, 7744, 7959, 8069, 8252, 8400, 8843])+9000

a2 = 9000+8996+np.asarray([0,480,848,1596,2394,2912,3842,3955,4581,5923,6071,6616,6667,7126,7214,7879,8055,8577])
b2 = 9000+8996+np.asarray( [126,549,893,1679,2432,3061,3855,4014,4609,6015,6084,6656,6680,7167,7260,8014,8075,8617])

a3= 9000+8996+8998+np.asarray([181,968,1060,1130,3464,5042,6260,6503,6744,6939,7325,7393,7608,7869,8164])
b3= 9000+8996+8998+np.asarray([230,1051,1076,1161,3593,5059,6338,6566,6784,7005,7378,7462,7663,7915,8437])

pos_ts_gradient=np.gradient(smoothing(pos_ts,100))
pos_ts_gradient=pos_ts_gradient*15
a=np.concatenate((a, a1,a2,a3), axis=0)
#a=a/15
b=np.concatenate((b, b1,b2,b3), axis=0)
#b=b/15
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

print('pre',avg_der_pos_pre)




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
    #r2_p = get_r2_statsmodels(pos_ts, myarray[:time.shape[0]])
    #r2_n= get_r2_statsmodels(neg_ts, myarray[:time.shape[0]])
    print(pos_ts.shape)
    print(neg_ts.shape)
    print(time.shape[0])
    r2_p=1
    r2_n=1
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


