# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:31:03 2020

@author: Jasen
"""
import numpy as np
from netCDF4 import Dataset as dt
import seawater as sw

def RhoPsiIndex_Mask(RomsFile, latbounds, lonbounds):
    """Locates indices of lat and lon within ROMS Output File with logical mask"""
    #load roms file
    RomsNC = dt(RomsFile, 'r')
    var = RomsNC.variables['salt'][:]
    
    #define masks
    Rholats = (RomsNC.variables['lat_rho'][:] >= latbounds[0])*(RomsNC.variables['lat_rho'][:] <= latbounds[1])
    Rholons = (RomsNC.variables['lon_rho'][:] >= lonbounds[0])*(RomsNC.variables['lon_rho'][:] <= lonbounds[1])
    RHOMASK = np.invert(Rholats*Rholons)
    
    Psilats = (RomsNC.variables['lat_psi'][:] >= latbounds[0])*(RomsNC.variables['lat_psi'][:] <= latbounds[1])
    Psilons = (RomsNC.variables['lon_psi'][:] >= lonbounds[0])*(RomsNC.variables['lon_psi'][:] <= lonbounds[1])
    PSIMASK = np.invert(Psilats*Psilons)
    
    #repeat masks over depth and time dimensions
    _RM = np.repeat(np.array(RHOMASK)[np.newaxis, :, :], var.shape[1], axis = 0)
    RhoMask = np.repeat(np.array(_RM)[np.newaxis, :, :, :], var.shape[0], axis = 0)
    
    _PM = np.repeat(np.array(PSIMASK)[np.newaxis, :, :], var.shape[1], axis = 0)
    PsiMask = np.repeat(np.array(_PM)[np.newaxis, :, :, :], var.shape[0], axis = 0)
    
    
    return RhoMask, PsiMask

def FaceMask(RomsFile, latbounds, lonbounds) :
    """
    Define masks along each face of control volume
    """
    RomsNC = dt(RomsFile, 'r')
    var = RomsNC.variables['salt']
    
    #north face
    Rholats = (RomsNC.variables['lat_rho'][:] == latbounds[0])*(RomsNC.variables['lat_rho'][:] <= latbounds[1])
    Rholons = (RomsNC.variables['lon_rho'][:] >= lonbounds[0])*(RomsNC.variables['lon_rho'][:] <= lonbounds[1])
    _NF = np.invert(Rholats*Rholons)
    _NFdep = np.repeat(np.array(_NF)[np.newaxis, :, :], var.shape[1], axis = 0)
    NorthFace = np.repeat(np.array(_NFdep)[np.newaxis, :, :, :], var.shape[0], axis = 0)
    
    #south face
    Rholats = (RomsNC.variables['lat_rho'][:] >= latbounds[0])*(RomsNC.variables['lat_rho'][:] == latbounds[1])
    Rholons = (RomsNC.variables['lon_rho'][:] >= lonbounds[0])*(RomsNC.variables['lon_rho'][:] <= lonbounds[1])
    _SF = np.invert(Rholats*Rholons)
    _SFdep = np.repeat(np.array(_SF)[np.newaxis, :, :], var.shape[1], axis = 0)
    SouthFace = np.repeat(np.array(_SFdep)[np.newaxis, :, :, :], var.shape[0], axis = 0)
    
    #east face
    Rholats = (RomsNC.variables['lat_rho'][:] >= latbounds[0])*(RomsNC.variables['lat_rho'][:] <= latbounds[1])
    Rholons = (RomsNC.variables['lon_rho'][:] == lonbounds[0])*(RomsNC.variables['lon_rho'][:] <= lonbounds[1])
    _EF = np.invert(Rholats*Rholons)
    _EFdep = np.repeat(np.array(_EF)[np.newaxis, :, :], var.shape[1], axis = 0)
    EastFace = np.repeat(np.array(_EFdep)[np.newaxis, :, :, :], var.shape[0], axis = 0)
    
    #west face
    Rholats = (RomsNC.variables['lat_rho'][:] >= latbounds[0])*(RomsNC.variables['lat_rho'][:] <= latbounds[1])
    Rholons = (RomsNC.variables['lon_rho'][:] >= lonbounds[0])*(RomsNC.variables['lon_rho'][:] == lonbounds[1])
    _WF = np.invert(Rholats*Rholons)
    _WFdep = np.repeat(np.array(_WF)[np.newaxis, :, :], var.shape[1], axis = 0)
    WestFace = np.repeat(np.array(_WFdep)[np.newaxis, :, :, :], var.shape[0], axis = 0)
    
    return NorthFace, WestFace, SouthFace, EastFace



def ROMS_CV(varname, RomsFile, Mask):
    #load
    RomsNC = dt(RomsFile, 'r')
    
    Var = np.ma.array(RomsNC.variables[varname][:], mask = Mask)
    
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
    






