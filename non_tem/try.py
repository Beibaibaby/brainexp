import numpy as np
coef=np.load('reg.coef_.npy')
print(coef.shape)
for i in range(10000):
    print(coef[i])