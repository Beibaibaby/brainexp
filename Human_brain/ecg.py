import numpy as np
import heartpy as hp
import math
import pandas as pd
from scipy import stats as st
import bioread

shapeletA = [-1.5116833, -0.5682773, 1.46413934, -0.7661631, 0.6143466, 0.12867262, 1.04853951, -0.4095744]
shapeletB = [-2.0847277, -0.2029212, 1.22671969, -0.3026837, 0.37005532, 0.72853974, 0.5283917, -0.2633737]

fs = 2000  # sampling rate of Biosignalplux
lowFreq = .1
highFreq = 100

shapelet_size = len(shapeletA)



def getFilterData(rawData):
    data = np.asarray(rawData)
    data_mV = ((((data / np.power(2, 16)) - 0.5) * 3) / 1019) * 1000  # ecg manual pdf page 5
    data_t = hp.filter_signal(data, [lowFreq, highFreq], fs, order=4, filtertype='bandpass', return_top=False)

    return data_t


def getHF(rawData):
    working_data, measures = hp.process_segmentwise(rawData, fs, segment_width=15, calc_freq=True, segment_overlap=0.5)
    lnHF_HRV = np.log(measures['hf'])

    return lnHF_HRV


def compareShapelets(hf_Data):
    results_shapeletA = []
    results_shapeletB = []
    for i in range(0, math.floor(len(hf_Data) / shapelet_size)):
        a = i * shapelet_size
        b = a + shapelet_size
        temp = hf_Data[a:b]
        results_shapeletA.append(st.pearsonr(temp, shapeletA))
        results_shapeletB.append(st.pearsonr(temp, shapeletB))

    shapA_results = pd.DataFrame(results_shapeletA, columns=['r', 'p value'])
    shapB_results = pd.DataFrame(results_shapeletB, columns=['r', 'p value'])


    return shapA_results, shapB_results


data0 = bioread.read_file('/Users/dragon/Desktop/fMRI_ECG/s161.acq')
print(data0.channels)
print(data0)
print(len(data0.channels))
data1=data0.channels[0].data
import matplotlib.pylab as plt

print(len(data1))
plt.show()

data_cleaned=getFilterData(data1)
#plt.plot(data_cleaned)
#plt.show()


print(data_cleaned)
HF_feature=getHF(data_cleaned)
print(HF_feature)


import pydicom as dicom
import matplotlib.pylab as plt
import cv2

# specify your image path
image_path = '/Users/dragon/Desktop/brainexp/101_T1.20200915.105517.14.rest_ep2d_bold_MB8_2mm.Echo_1.0001.dcm'
ds = dicom.dcmread(image_path)

pixel_array_numpy = ds.pixel_array

image_format = '.jpg' # or '.png'
image_path = image_path.replace('.dcm', image_format)

cv2.imwrite(image_path, pixel_array_numpy)
