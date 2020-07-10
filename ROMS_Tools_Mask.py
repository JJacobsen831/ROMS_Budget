# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:31:03 2020

@author: Jasen
"""
import numpy as np
from netCDF4 import Dataset as dt
import obs_depth_JJ as dep
from Filters import godinfilt 

def RhoPsiIndex_Mask(RomsFile, latbounds, lonbounds):
    """Locates indices of lat and lon within ROMS Output File with logical mask"""
    #load roms file
    RomsNC = dt(RomsFile, 'r')
    
    Rholats = (RomsNC.variables['lat_rho'][:] >= latbounds[0])*(RomsNC.variables['lat_rho'][:] <= latbounds[1])
    Rholons = (RomsNC.variables['lon_rho'][:] >= lonbounds[0])*(RomsNC.variables['lon_rho'][:] <= lonbounds[1])
    RhoMask = np.invert(Rholats*Rholons)
    
    Psilats = (RomsNC.variables['lat_psi'][:] >= latbounds[0])*(RomsNC.variables['lat_psi'][:] <= latbounds[1])
    Psilons = (RomsNC.variables['lon_psi'][:] >= lonbounds[0])*(RomsNC.variables['lon_psi'][:] <= lonbounds[1])
    PsiMask = np.invert(Psilats*Psilons)
    
    return RhoMask, PsiMask

def ROMS_CV(Var, RomsFile, Mask):
    #load
    RomsNC = dt(RomsFile, 'r')
    
    Var = np.ma.MaskedArray(RomsNC.variables[Var], Mask)
    
    return Var
    