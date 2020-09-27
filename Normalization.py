from __future__ import division 
import numpy as np
import scipy.io as sio 
def Normalization(raw_data):
    '''
    normalize the whole data to [-0.5,0.5]
    '''
    MAX = np.max(raw_data.ravel()).astype('float32')
    MIN = np.min(raw_data.ravel())
    new_data = (raw_data - MIN)/(MAX - MIN)- 0.5
    return new_data

def NormalizationEachBand(raw_data, unit=True):
    '''
    normalize the whole data to [0,1]
    '''
    new_data=np.zeros([raw_data.shape[0],raw_data.shape[1],raw_data.shape[2]])
    for i in range(raw_data.shape[2]):
        temp = raw_data[:,:,i]
        MAX = np.max(temp.ravel()).astype('float32')
        MIN = np.min(temp.ravel())
        new_data[:,:,i] = (temp - MIN)/(MAX - MIN)
    if unit:
        new_data = new_data.reshape(np.prod(new_data.shape[:2]),np.prod(new_data.shape[2:]))
        new_data=zscores(new_data)
        new_data=new_data.reshape(raw_data.shape[0],raw_data.shape[1],raw_data.shape[2])        
    return new_data

def zscores(data):
    '''
    For matrix data, z-scores are computed using the mean and standard deviation along each row of data.
    returns a centered, scaled version of each sample, (X-MEAN(X)) ./ STD(X)
    input: data with the shape of [samples_number,feature]
    This function performs well in ELM algorithm, but not well in 1D-CNN
    '''
    new_data=np.zeros([data.shape[0],data.shape[1]])
    for j in range(data.shape[0]):
        new_data[j,:] = (data[j,:]-np.mean(data[j,:]))/np.std(data[j,:],ddof=1)
    return new_data

#train_dataf = '/home/amax/xibobo/function_library/P.mat'
#
#train_x = sio.loadmat(train_dataf)['P'] 
#   
#test_dataf = '/home/amax/xibobo/function_library/PA.mat'
#
#test_x = sio.loadmat(test_dataf)['PA']   
#train_x=train_x.T   
#test_AP = zscores(train_x)
#error = test_AP-test_x.T
#uPavia = sio.loadmat('/home/amax/xibobo/SSRN-master/datasets/UP/PaviaU.mat')
#data_IN = uPavia['paviaU']
#P = NormalizationEachBand(data_IN, unit=True)