# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 12:57:23 2020

@author: Jasen
"""
import numpy as np

def godinfilt(xin, skip_ind = 0) :
    """ function that returns 'godin' running averages of 24, 25, 25 hrs applied successively
    Note: time series must be hourly and only a vector or matrix of column vectors (dim 0 is time)
    skip_ind = 1 does not apply filter and returns xin unaltered
    Adopted from DAS matlab code"""
    
    if skip_ind != 1 :
        x_shape = xin.shape
        xnew = np.zeros(x_shape)
        
        #build 24 hour filter
        filter24 = np.ones([24,1])
        filter24 = np.ndarray.flatten(filter24/np.sum(filter24))
        
        #build 25 hour filter
        filter25 = np.ones([25,1])
        filter25 = np.ndarray.flatten(filter25/np.sum(filter25))
        
        #convolove filters together
        temp_filt = np.convolve(filter24, filter24)
        filt = np.convolve(temp_filt, filter25)
        
        #apply filter to time series
        a = np.int(np.round(filt.size/2))
                
        for i, col in enumerate(xin.T):
            temp = np.convolve(np.ndarray.flatten(col), filt)
            xnew[:,i] = temp[a:a+x_shape[0]]
            
        xfilt = xnew
            
    elif skip_ind == 1 :
            xfilt = xin
    
    return xfilt
