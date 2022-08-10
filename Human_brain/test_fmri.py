import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib # common way of importing nibabel

import numpy as np
import heartpy as hp
import math
import pandas as pd
from scipy import stats as st
import bioread
from scipy import ndimage

shapeletA = [-1.5116833, -0.5682773, 1.46413934, -0.7661631, 0.6143466, 0.12867262, 1.04853951, -0.4095744]
shapeletB = [-2.0847277, -0.2029212, 1.22671969, -0.3026837, 0.37005532, 0.72853974, 0.5283917, -0.2633737]

fs = 2000  # sampling rate of Biosignalplux
lowFreq = .1
highFreq = 100

shapelet_size = len(shapeletA)



def getFilterData(rawData):
    data = np.asarray(rawData)
    data = np.interp(data, (data.min(), data.max()), (0, +1024))
    #data = ((((data / np.power(2, 16)) - 0.5) * 3) / 1019) * 2000  # ecg manual pdf page 5
    #data = hp.filter_signal(data, [lowFreq, highFreq], fs, order=4, filtertype='bandpass', return_top=False)

    return data


def getHF(rawData):
    working_data, measures = hp.process_segmentwise(rawData, fs, segment_width=30, calc_freq=True, segment_overlap=0)
    lnHF_HRV = np.log(measures['hf'])
    rmssd=np.log(measures['rmssd'])
    return lnHF_HRV, rmssd


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


data0 = bioread.read_file('/Users/dragon/Downloads/BREATHE/ecg/BREATHE_s106_T1 (Rest).acq')


import matplotlib.pylab as plt
rawdata=data0.channels[0].data
print(rawdata.size)
fs = 2000.0

working_data, measures = hp.process(rawdata, fs, report_time=True)
hf,rmssd=getHF(rawdata)
print(hf)
print(rmssd)

#plt.plot(hf)
plt.plot(rmssd)
plt.show()










###################################
mri_file = '/Users/dragon/Downloads/BREATHE/fmri/s106/ouput.feat/filtered_func_data.nii.gz'
img = nib.load(mri_file)
print(img)
print(img.shape)
img_data = img.get_fdata()
#print(type(img_data))  # it's a numpy array!
#print(img_data)
mid_slice_x = img_data[60, :, :,150]
print(mid_slice_x.shape)
plt.imshow(mid_slice_x.T, cmap='gray', origin='lower')
plt.xlabel('First axis')
plt.ylabel('Second axis')
plt.colorbar(label='Signal intensity')
plt.show()

#new_arr = img_data.reshape(-1, img_data.shape[-1])
#print(new_arr.shape)


def extract_average_series(ts,sliding_window_size):
    #ts is a numpy array
    seriesofts= np.array_split(ts, int(ts.shape[-1]/sliding_window_size),axis=-1)
    seriesofts= np.asarray(seriesofts[:-1])
    last_element=np.asarray(seriesofts[-1])
    last_mean=np.mean(last_element,-1)
    mean_series = np.mean(seriesofts,-1)

    return np.concatenate((mean_series, np.expand_dims(last_mean, axis=0)), axis=0)


mri_file_2 = '/Users/dragon/Downloads/BREATHE/fmri/s106/ouput.feat/filtered_func_data.nii.gz'
img_2 = nib.load(mri_file)
img_data_2 = img_2.get_fdata()


fmri_feature=extract_average_series(img_data,30)
ecg_feature=rmssd[:fmri_feature.shape[0]]



img = ndimage.zoom(fmri_feature[0], 0.5)


