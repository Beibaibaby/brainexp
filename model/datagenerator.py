
import numpy as np
import matplotlib.pyplot as plt
from numpy import array, dot, mean, std, empty, argsort
from numpy.linalg import eigh
import statsmodels.api as sm
import statsmodels.formula.api as smf

ts_transformed=np.load('/Users/dragon/Desktop/brainexp/pca/series/overall/ts_transformed.npy')
pc1 = ts_transformed.T[0]



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





run = np.zeros((np.size(pc1)))
for i in range(len(a)):
    run[a[i]:b[i]]=1

for i in range(len(a1)):
    run[a1[i]:b1[i]]=1

for i in range(len(a2)):
    run[a2[i]:b2[i]]=1

for i in range(len(a3)):
    run[a3[i]:b3[i]]=1


wishking = []
with open('/Users/dragon/Desktop/brainexp/behav/w1.csv') as f:
        lines = f.readlines()
        for line in lines:
            wishking.append(line)

with open('/Users/dragon/Desktop/brainexp/behav/w2.csv') as f:
    lines = f.readlines()
    for line in lines:
        wishking.append(line)

with open('/Users/dragon/Desktop/brainexp/behav/w3.csv') as f:
        lines = f.readlines()
        for line in lines:
            wishking.append(line)

with open('/Users/dragon/Desktop/brainexp/behav/w4.csv') as f:
    lines = f.readlines()
    for line in lines:
        wishking.append(line)

wishking = np.asarray(wishking, dtype=np.float32)
wishking = wishking[:run.size]

time = []

for i in range(pc1.size):
    time.append(i%8995)

status=[]

for i in range(pc1.size):
    if i < int(pc1.size/4):
        status.append(1)
    elif i>=int(pc1.size/4) and i < int(pc1.size/2):
        status.append(4)
    elif i >= int(pc1.size / 2) and i < int(3*pc1.size / 4):
        status.append(2)
    else:
        status.append(3)

status=np.asarray(status)
time = np.asarray(time)

print(time)


group = np.zeros((np.size(pc1)))
stand=True
pos_ts_gradient=np.load('/Users/dragon/Desktop/brainexp/pca/pos_ts_gradient.npy')
neg_ts_gradient=np.load('/Users/dragon/Desktop/brainexp/pca/neg_ts_gradient.npy')
if stand == True:
    status = np.interp(status, (status.min(), status.max()), (0, pc1.max()))
    time = np.interp(time, (time.min(), time.max()), (0, pc1.max()))
    # wishking = np.interp(wishking, (wishking.min(), wishking.max()), (0, +1))
    # pc1 = np.interp(pc1, (pc1.min(), pc1.max()), (-1, +1))
    run = np.interp(run, (run.min(), run.max()), (pc1.min(), pc1.max()))
    # wishking = np.interp(wishking, (wishking.min(), wishking.max()),(pc1.min(), pc1.max()))
    pos_ts_gradient=np.interp(pos_ts_gradient, (pos_ts_gradient.min(), pos_ts_gradient.max()), (pc1.min(), pc1.max()))
    neg_ts_gradient = np.interp(neg_ts_gradient, (neg_ts_gradient.min(), neg_ts_gradient.max()), (pc1.min(), pc1.max()))

model = sm.MixedLM(run.T, np.asarray([pc1,time,wishking,status,pos_ts_gradient,neg_ts_gradient]).T,group)
result = model.fit()
print(result.summary())
data=np.asarray([run,pc1,time,wishking,status]).T
np.savetxt("data.csv", data, delimiter=",")

model = sm.MixedLM(wishking.T, np.asarray([pc1, time, run, status,pos_ts_gradient,neg_ts_gradient]).T, group)
result = model.fit()
print(result.summary())


