# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:31:03 2020

@author: Jasen
"""
import numpy as np
from netCDF4 import Dataset as dt
import obs_depth_JJ as dep
from Filters import godinfilt 
import seawater as sw

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
    
def rho_dist_sw(RomsFile) :
    """
    Use seawater package to compute distance between rho points
    Returns distance in meters
    """
    RomsNC = dt(RomsFile, 'r')
    
    lat = RomsNC.variables['lat_rho'][:]
    lon = RomsNC.variables['lon_rho'][:]
    
    x_dist = np.empty((lon.shape[0], lon.shape[1]-1))
    x_dist.fill(np.nan)
    for i in range(lon.shape[0]) :
        for j in range(lon.shape[1]-1) :
            x_dist[i, j] = sw.dist([lat[i, j], lat[i, j+1]], [lon[i, j], lon[i, j +1]])[0]*1000
            
    y_dist = np.empty((lat.shape[0]-1, lat.shape[1]))
    y_dist.fill(np.nan)
    for i in range(y_dist.shape[0]):
        for j in range(y_dist.shape[1]) :
            y_dist[i, j] = sw.dist([lat[i,j], lat[i+1, j]], [lon[i,j], lon[i+1, j]])[0]*1000
            
    return x_dist, y_dist

def rho_dist_grd(RomsGrd) :
    """ 
    Compute distance between rho points using roms grid file
    """
    RomsNC = dt(RomsGrd, 'r')
    
    #distance from grid edge
    Xx = RomsNC.variables['x_rho'][:]
    Yy = RomsNC.variables['y_rho'][:]
    
    #difference between points (meters)
    dx = np.diff(Xx, n = 1, axis = 1)
    dy = np.diff(Yy, n = 1, axis = 0)
    
    return dx, dy
    






