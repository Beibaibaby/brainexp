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



ts = np.load('/Users/dragon/Desktop/brainexp/pro_data/ts_201217_01.npy')
ts = np.delete(ts, (0), axis=0)
ts1 = np.load('/Users/dragon/Desktop/brainexp/pro_data/ts_201224_01.npy')
ts1 = np.delete(ts1, (0), axis=0)
ts2 = np.load('/Users/dragon/Desktop/brainexp/pro_data/ts_201224_02.npy')
ts2 = np.delete(ts2, (0), axis=0)
ts3 = np.load('/Users/dragon/Desktop/brainexp/pro_data/ts_201225_01.npy')
ts3 = np.delete(ts3, (0), axis=0)
ts=np.concatenate((ts, ts1,ts2,ts3), axis=0)

print('Shape of the TS')
print(ts.shape)
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca.fit(ts)
print(pca.components_)
print(pca.explained_variance_ratio_[0])


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

pca = PCA(n_components=5)
ts_transformed = pca.fit_transform(ts)
np.save('./series/overall/pca', pca.components_)
np.save('./series/overall/ts_transformed', ts_transformed)
ccc = ['blue', 'red', 'green', 'brown', 'purple']
dt = 1
starttime = 0
endtime = 4*8995
steps = int(abs(starttime - endtime) / dt)
time = np.linspace(starttime, endtime, steps)
print(time.shape)
ccc = ['blue', 'red', 'green', 'brown', 'purple']
a = np.asarray([22, 266, 512, 1561, 2237, 2959, 3393, 3580, 3760, 5133, 5565])
b = np.asarray([117, 355, 602, 1628, 2389, 3010, 3475, 3633, 3827, 5284, 5692])

a1 = np.asarray([10, 193, 371, 1261, 1488, 1722, 2933, 3097, 3257, 3364, 3745, 4034, 4366, 5279, 5656, 5781, 6066, 6349, 6469,
      7111, 7375, 7711, 7912, 8026, 8194, 8379, 8744])+8995
b1 = np.asarray([51, 207, 442, 1342, 1542, 1764, 2965, 3135, 3281, 3412, 3867, 4098, 4458, 5360, 5757, 5800, 6081, 6406, 6507,
      7200, 7401, 7744, 7959, 8069, 8252, 8400, 8843])+8995

a2 = 8995*2+np.asarray([0,480,848,1596,2394,2912,3842,3955,4581,5923,6071,6616,6667,7126,7214,7879,8055,8577])
b2 = 8995*2+np.asarray( [126,549,893,1679,2432,3061,3855,4014,4609,6015,6084,6656,6680,7167,7260,8014,8075,8617])

a3= 8995*3+np.asarray([181,968,1060,1130,3464,5042,6260,6503,6744,6939,7325,7393,7608,7869,8164])
b3= 8995*3+np.asarray([230,1051,1076,1161,3593,5059,6338,6566,6784,7005,7378,7462,7663,7915,8437])


a=np.concatenate((a, a1,a2,a3), axis=0)
b=np.concatenate((b, b1,b2,b3), axis=0)
for idx in [1,2,3,4,5]:

    r2 = get_r2_statsmodels(ts_transformed.T[idx - 1], myarray[:time.shape[0]])
    fig, ax = plt.subplots()
    fig.set_size_inches(23.5, 10.5)




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
    plt.title('PC' + str(idx) + '-Overall R2=' + str(float(int(r2 * 1000) / 1000)))

    plt.savefig('./series/overall/pc' + str(idx) + '.png')
    plt.show()


