import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import csv
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



abss = np.load('/Users/dragon/Desktop/brainexp/pca/series/beforestress/pca.npy')
for i in range(5):
    abb = abss[i]

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
    plt.title('PC' + str(i+1) )

    plt.savefig('./beforestress/fig' + str(i))
    plt.show()