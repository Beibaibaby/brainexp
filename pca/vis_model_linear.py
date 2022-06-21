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

wishking = np.asarray(myarray,dtype=np.float32)


from sklearn.linear_model import Lasso
print('start')
reg1 = Lasso(alpha=0.0001)
wishking = wishking[:35980]
print(ts.shape)
print(wishking.shape)


reg1.fit(ts, wishking)
print('R squared training set', round(reg1.score(ts, wishking)*100, 2))
print('R squared test set', round(reg1.score(ts, wishking)*100, 2))
np.save('reg.coef_',reg1.coef_)
print(reg1.coef_)


print('start2')




a = [22, 266, 512, 1561, 2237, 2959, 3393, 3580, 3760, 5133, 5565]
b = [117, 355, 602, 1628, 2389, 3010, 3475, 3633, 3827, 5284, 5692]

a1 = np.asarray([10, 193, 371, 1261, 1488, 1722, 2933, 3097, 3257, 3364, 3745, 4034, 4366, 5279, 5656, 5781, 6066, 6349, 6469,
      7111, 7375, 7711, 7912, 8026, 8194, 8379, 8744])+8995
b1 = np.asarray([51, 207, 442, 1342, 1542, 1764, 2965, 3135, 3281, 3412, 3867, 4098, 4458, 5360, 5757, 5800, 6081, 6406, 6507,
      7200, 7401, 7744, 7959, 8069, 8252, 8400, 8843])+8995

a2 = 8995*2+np.asarray([0,480,848,1596,2394,2912,3842,3955,4581,5923,6071,6616,6667,7126,7214,7879,8055,8577])
b2 = 8995*2+np.asarray( [126,549,893,1679,2432,3061,3855,4014,4609,6015,6084,6656,6680,7167,7260,8014,8075,8617])

a3= 8995*3+np.asarray([181,968,1060,1130,3464,5042,6260,6503,6744,6939,7325,7393,7608,7869,8164])
b3= 8995*3+np.asarray([230,1051,1076,1161,3593,5059,6338,6566,6784,7005,7378,7462,7663,7915,8437])





run = np.zeros(ts.shape[0])
for i in range(len(a)):
    run[a[i]:b[i]]=1

for i in range(len(a1)):
    run[a1[i]:b1[i]]=1

for i in range(len(a2)):
    run[a2[i]:b2[i]]=1

for i in range(len(a3)):
    run[a3[i]:b3[i]]=1



reg = Lasso(alpha=0.0001)
print(ts.shape)
print(run.shape)


reg.fit(ts, run)
print('RUN:R squared training set', round(reg.score(ts, run)*100, 2))
print('RUN:R squared test set', round(reg.score(ts, run)*100, 2))
np.save('RUN_reg.coef_',reg.coef_)
print(reg.coef_)





mask_ss = np.load('./mask_ss.npy')
akk = mask_ss[:480,:]*200
mcopy = np.full_like(akk, 0)
def searchregion_1(matrix,ini_i,ini_j,value):


    i_index = ini_i
    j_index = ini_j

    while True: #bring back to the center
            i_curr = i_index
            j_curr = j_index
            i_curr_1=i_curr
            j_curr_1=j_curr

            while True:
                if matrix[i_curr, j_curr] > 100:
                    break
                else:
                    mcopy[i_curr, j_curr] = value
                    i_curr = i_curr
                    j_curr = j_curr + 1

            i_index = i_index - 1

            if matrix[i_index, j_index] > 100:

                if matrix[i_index, j_index + 1]<100:
                    j_index = j_index + 1
                elif matrix[i_index, j_index - 1]<100:
                    j_index = j_index - 1

                else:
            #print(i_index, j_index)
                    break
    i_index = ini_i
    j_index = ini_j
    while True: #bring back to the center
            i_curr = i_index
            j_curr = j_index
            i_curr_1=i_curr
            j_curr_1=j_curr

            while True:
                if matrix[i_curr, j_curr] > 100:
                    break
                else:
                    mcopy[i_curr, j_curr] = value
                    i_curr = i_curr
                    j_curr = j_curr + 1

            i_index = i_index + 1

            if matrix[i_index, j_index] > 100:
                if matrix[i_index, j_index + 1]<100:

                    j_index = j_index + 1

                elif matrix[i_index, j_index - 1]<100:

                    j_index = j_index - 1

                else:
            #print(i_index, j_index)
                    break


    i_index = ini_i
    j_index = ini_j

    while True: #bring back to the center
            i_curr = i_index
            j_curr = j_index

            while True:
                if matrix[i_curr, j_curr] > 100:
                    break
                else:
                    mcopy[i_curr, j_curr] = value
                    i_curr = i_curr
                    j_curr = j_curr - 1

            i_index = i_index + 1

            if matrix[i_index, j_index] > 100:
                if matrix[i_index, j_index + 1]<100:

                    j_index = j_index + 1

                elif matrix[i_index, j_index - 1]<100:

                    j_index = j_index - 1

                else:
            #print(i_index, j_index)
                    break
    i_index = ini_i+1
    j_index = ini_j

    while True: #bring back to the center
            i_curr = i_index
            j_curr = j_index

            while True:
                if matrix[i_curr, j_curr] > 100:
                    break
                else:
                    mcopy[i_curr, j_curr] = value
                    i_curr = i_curr
                    j_curr = j_curr - 1

            i_index = i_index - 1

            if matrix[i_index, j_index] > 100:
                if matrix[i_index, j_index + 1]<100:

                    j_index = j_index + 1

                elif matrix[i_index, j_index - 1]<100:

                    j_index = j_index - 1

                else:
            #print(i_index, j_index)
                    break

    return matrix







abss = reg.coef_
for i in range(1):
    abb = abss

    print(abb)

    r1 = searchregion_1(akk, 50, 255, abb[0])

    r2 = searchregion_1(r1, 110, 170, abb[1])
    r2 = searchregion_1(r2, 210, 250, abb[1])

    r3 = searchregion_1(r2, 150, 100, abb[2])
    r4 = searchregion_1(r3, 200, 170, abb[3])

    r5 = searchregion_1(r4, 250, 200, abb[4])
    r6 = searchregion_1(r5, 300, 300, abb[5])
    r7 = searchregion_1(r6, 250, 75, abb[6])
    r8 = searchregion_1(r7, 210, 100, abb[7])

    r9 = searchregion_1(r8, 210, 132, abb[8])
    r9 = searchregion_1(r9, 260, 170, abb[8])
    r10 = searchregion_1(r9, 300, 150, abb[9])
    r10 = searchregion_1(r10, 320, 105, abb[9])

    r11 = searchregion_1(r10, 280, 200, abb[10])
    r11 = searchregion_1(r11, 250, 235, abb[10])
    r11 = searchregion_1(r11, 260, 230, abb[10])
    r12 = searchregion_1(r11, 310, 175, abb[11])

    r13 = searchregion_1(r12, 350, 225, abb[12])

    r14 = searchregion_1(r13, 360, 125, abb[13])
    r14 = searchregion_1(r14, 315, 150, abb[13])

    r15 = searchregion_1(r14, 350, 75, abb[14])

    r16 = searchregion_1(r15, 370, 100, abb[15])

    r17 = searchregion_1(r16, 400, 200, abb[16])

    r18 = searchregion_1(r17, 380, 225, abb[17])
    r18 = searchregion_1(r18, 350, 200, abb[17])

    r19 = searchregion_1(r18, 410, 65, abb[18])
    r19 = searchregion_1(r19, 390, 80, abb[18])

    r20 = searchregion_1(r19, 415, 85, abb[19])

    r21 = searchregion_1(r20, 430, 110, abb[20])

    r22 = searchregion_1(r21, 470, 120, abb[21])

    r23 = searchregion_1(r22, 440, 80, abb[22])

    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = (5,8)
    plt.imshow(mcopy)
    plt.colorbar()
    plt.title('Lasso' + str(i+1) )

    plt.savefig('./lasso' + str(i))
    plt.show()


print(ts.shape)
print(np.nonzero(reg1.coef_)[0])
ts=ts[:,np.nonzero(reg1.coef_)[0]]

from sklearn.linear_model import LinearRegression
status=[]
lenth=ts.shape[0]
print(ts.shape)
for i in range(lenth):
    if i < int(lenth/4):
        status.append(0)
    elif i>=int(lenth/4) and i < int(lenth/2):
        status.append(6)
    elif i >= int(lenth / 2) and i < int(3*lenth / 4):
        status.append(3)
    else:
        status.append(2)

status=np.asarray(status)
lin = LinearRegression().fit(ts, status)
print(lin.score(ts, status))