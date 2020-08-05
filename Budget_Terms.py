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
import ROMS_Tools_Mask as rt


def Prime(var_4dim) :
    """
    Compute anomaly of variable for each time step
    var_4dim = var_4dim[time, depth, lat, lon]
    """
    var_prime = np.empty(var_4dim.shape)
    var_prime.fill(np.nan)
    for n in range(var_4dim.shape[0]) :
        var_prime[n, :, :, :] = var_4dim[n,:,:,:] - ma.mean(var_4dim[n, :, :, :])
    
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

def TermTwo(RomsFile, RomsGrd, varname, latbounds, lonbounds) :
    """
    Flux of variance across open boundaries
    """
    RomsNC = nc4(RomsFile, 'r')
    var = RomsNC.variables[varname][:]
    
    #shift variable from rho points to u and v points
    _var_u = 0.5*(var[:,:, :, 0:var.shape[3]-1] + var[:,:, :, 1:var.shape[3]])
    _var_v = 0.5*(var[:,:, 0:var.shape[2]-1, :] + var[:,:, 1:var.shape[2], :])
    
    #define masks for u and v points
    _, U_Mask, V_Mask = rt.RhoUV_Mask(RomsFile, latbounds, lonbounds)
    
    #compute variance 
    var_u = ma.array(_var_u, mask = U_Mask)
    var_v = ma.array(_var_v, mask = V_Mask)
    
    #variance squared
    prime2_u = Prime(var_u)**2
    
    prime2_v = Prime(var_v)**2
    
    #define face masks
    NorthFace, WestFace, SouthFace, EastFace = rt.FaceMask(RomsFile,\
                                                       latbounds, lonbounds)
    
    #apply face masks to variance squared
    North_var = ma.array(prime2_v, mask = NorthFace)
    South_var = ma.array(prime2_v, mask = SouthFace)
    West_var = ma.array(prime2_u, mask = WestFace)
    East_var = ma.array(prime2_u, mask = EastFace)
    
    #compute differential areas
    Ax_norm, Ay_norm = df.dA(RomsFile, RomsGrd)
    
    #subset areas
    North_Ay = ma.array(Ay_norm, mask = NorthFace)
    West_Ax = ma.array(Ax_norm, mask = WestFace)
    South_Ay = ma.array(Ay_norm, mask = SouthFace)
    East_Ax = ma.array(Ax_norm, mask = EastFace)
    
    #velocities
    North_V = ma.array(RomsNC.variables['v'][:], mask =  NorthFace)
    West_U = ma.array(RomsNC.variables['u'][:], mask = WestFace)
    South_V = ma.array(RomsNC.variables['v'][:], mask = SouthFace)
    East_U = ma.array(RomsNC.variables['u'][:], mask = EastFace)
    
    #multiply to get integrad
    North = North_V*North_Ay*North_var
    West = West_U*West_Ax*West_var
    South = South_V*South_Ay*South_var
    East = East_U*East_Ax*East_var
    
    #sum/integrate each time step
    Flux = np.empty(North.shape[0])
    Flux.fill(np.nan)
    for n in range(North.shape[0]) :
        Flux[n] = ma.sum(North[n,:,:,:]) + ma.sum(South[n,:,:,:]) + \
                  ma.sum(East[n,:,:,:]) + ma.sum(West[n,:,:,:])

    return Flux