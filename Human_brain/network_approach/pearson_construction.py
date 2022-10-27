#import test_fmri
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
from os import path

Filepath = '/Users/dragon/Downloads/BREATHE'
fmripath = os.path.join(Filepath, 'fmri')
ecgpath = os.path.join(Filepath, 'ecg')

fs = 2000  # sampling rate of Biosignalplux
lowFreq = .1
highFreq = 100

def getFilterData(rawData):
    data = np.asarray(rawData)
    data = np.interp(data, (data.min(), data.max()), (0, +1024))
    #data = ((((data / np.power(2, 16)) - 0.5) * 3) / 1019) * 2000  # ecg manual pdf page 5
    #data = hp.filter_signal(data, [lowFreq, highFreq], fs, order=4, filtertype='bandpass', return_top=False)

    return data


def getHF(rawData):
    working_data, measures = hp.process_segmentwise(rawData, fs, segment_width=45, calc_freq=True, segment_overlap=0)
    lnHF_HRV = np.log(measures['hf'])
    rmssd=np.log(measures['rmssd'])
    return lnHF_HRV, rmssd



#print(ecgpath)
def listdirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def extract_average_series(ts,sliding_window_size):
    #ts is a numpy array
    seriesofts= np.array_split(ts, int(ts.shape[-1]/sliding_window_size),axis=-1)
    seriesofts= np.asarray(seriesofts[:-1])
    last_element=np.asarray(seriesofts[-1])
    last_mean=np.mean(last_element,-1)
    mean_series = np.mean(seriesofts,-1)

    return np.concatenate((mean_series, np.expand_dims(last_mean, axis=0)), axis=0)




def builddatset(path):
    d_list = listdirs(path)
    all_subject_fmri_features=[]
    all_subject_ecg_features=[]
    for d in d_list:
        tem_path = os.path.join(path, d)
        if os.path.exists(os.path.join(tem_path,'final.feat','filtered_func_data.nii.gz')):
            mri_file = os.path.join(tem_path,'final.feat','filtered_func_data.nii.gz')
            img = nib.load(mri_file)
            img_data = img.get_fdata()
            print(img_data.shape)
            fmri_feature = extract_average_series(img_data, 45)
            print(d+' fmri data extracted successfully')
            print('check the size')
            print(fmri_feature.shape)
            ecg_path_tem=os.path.join(tem_path, 'ecg')
            if os.path.exists(ecg_path_tem):
                for file in os.listdir(ecg_path_tem):
                    if file.endswith(".acq"):
                        print(os.path.join(ecg_path_tem, file))
                        data0 = bioread.read_file(os.path.join(ecg_path_tem, file))
                        rawdata = data0.channels[0].data
                        hf, rmssd = getHF(rawdata)
                        print(d + ' ecg feature extracted successfully')
                        print('check the data')
                        ecg_feature=rmssd[:fmri_feature.shape[0]]




            all_subject_fmri_features.append(fmri_feature)
            for i in range(ecg_feature.size):
                if np.isnan(ecg_feature[i]):
                    ecg_feature[i]=(ecg_feature[i-1]+ecg_feature[i+1])/2
            #print()
            print(ecg_feature)
            all_subject_ecg_features.append(ecg_feature)


        else:
            print('no_pre_fmri'+d)
    all_subject_fmri_features=np.asarray(all_subject_fmri_features)
    all_subject_ecg_features = np.asarray(all_subject_ecg_features)
    return all_subject_fmri_features, all_subject_ecg_features



def builddatset_nonav(path):
    d_list = listdirs(path)
    all_subject_fmri_features=[]
    for d in d_list:
        tem_path = os.path.join(path, d)
        if os.path.exists(os.path.join(tem_path,'final.feat','filtered_func_data.nii.gz')):
            mri_file = os.path.join(tem_path,'final.feat','filtered_func_data.nii.gz')
            img = nib.load(mri_file)
            img_data = img.get_fdata()
            print(img_data.shape)
            fmri_feature = img_data
            print(d+' fmri data extracted successfully')
            print('check the size')
            print(fmri_feature.shape)


            all_subject_fmri_features.append(fmri_feature)



        else:
            print('no_pre_fmri'+d)
    all_subject_fmri_features=np.asarray(all_subject_fmri_features)
    return all_subject_fmri_features

if __name__ == "__main__":
    #nonav_fmri_features=builddatset_nonav(fmripath)
    #print(nonav_fmri_features)
    #np.save('nonav_fmri_features', nonav_fmri_features)
    all_subject_fmri_features, all_subject_ecg_features =builddatset(fmripath)
    print(all_subject_fmri_features.shape)
    print(all_subject_ecg_features.shape)
    np.save('all_subject_fmri_features', all_subject_fmri_features)
    np.save('all_subject_ecg_features', all_subject_ecg_features)

