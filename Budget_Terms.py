# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:27:30 2020

Compute each term in variance budget

@author: Jasen
"""
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset as nc4
import Differential as df

def Prime(var_4dim) :
    """
    Compute anomaly of variable for each time step
    var_4dim = var_4dim[time, depth, lat, lon]
    """
    var_prime = np.empty(var_4dim.shape)
    var_prime.fill(np.nan)
    for n in range(var_4dim.shape[0]) :
        var_prime[n, :, :, :] = var_4dim[n,:,:,:] - np.mean(var_4dim[n, :, :, :])
    
    return var_prime

def TermOne(RomsFile, Mask, Variance) :
    """
    Compute the change of variance within a control volume
    """
    #ocean time
    RomsNC = nc4(RomsFile, 'r')
    time = RomsNC.variables['ocean_time']
    
    #differential volume
    dv = df.dV(RomsFile)
    dv = ma.array(dv, mask = Mask)
    
    #multiply to get integrad
    S_m = Variance*dv
    
    #integrate volume at each time step
    S_int = np.empty(S_m.shape[0])
    S_int.fill(np.nan)
    for n in range(S_int.shape[0]) :
        S_int[n] = np.sum(S_m[n, :, :, :])
        
    #time derivative
    dVar_dt = np.diff(S_int)/np.diff(time)
    
    return dVar_dt

