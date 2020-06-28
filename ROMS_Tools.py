# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 12:29:58 2020

@author: Jasen
"""
# ROMS data extraction for CV budget
import numpy as np
from netCDF4 import Dataset as dt

def ROMS_Load(RomsFile, VarName, latbounds, lonbounds) :
    """Loads lat_rho/psi, lon_rho/psi, time, u, ubar, v, vbar, w, and a defined variable \
    at defined latitude and longitude. Stores in dictionary"""
    #load nc file   
    RomsNC = dt(RomsFile, 'r')
    
    #locate rho points within bounds 
    lat_li = np.argmin(np.array(np.abs(RomsNC.variables['lat_rho'][:, 0] - latbounds[0])))
    lat_ui = np.argmin(np.array(np.abs(RomsNC.variables['lat_rho'][:, 0] - latbounds[1])))
    lon_li = np.argmin(np.array(np.abs(RomsNC.variables['lon_rho'][0, :] - lonbounds[0])))
    lon_ui = np.argmin(np.array(np.abs(RomsNC.variables['lon_rho'][0, :] - latbounds[1])))
    
    #locate psi points within bounds
    latpsi_li = np.argmin(np.array(np.abs(RomsNC.variables['lat_psi'][:, 0] - latbounds[0])))
    latpsi_ui = np.argmin(np.array(np.abs(RomsNC.variables['lat_psi'][:, 0] - latbounds[1])))
    lonpsi_li = np.argmin(np.array(np.abs(RomsNC.variables['lon_psi'][0, :] - lonbounds[0])))
    lonpsi_ui = np.argmin(np.array(np.abs(RomsNC.variables['lon_psi'][0, :] - latbounds[1])))
    
    #subset variables and store in dictionary
    ROMS = {'lat_rho' : np.array(RomsNC.variables['lat_rho'][lat_li:lat_ui, lon_li:lon_ui], dtype = np.float64), \
            'lon_rho' : np.array(RomsNC.variables['lon_rho'][lat_li:lat_ui, lon_li:lon_ui], dtype = np.float64), \
            'time' : np.array(RomsNC.variables['ocean_time'][:], dtype = np.float64), \
            #'w' : np.array(RomsNC.variables['w'][:, :, lat_li:lat_ui, lon_li:lon_ui], dtype = np.float64), \
            VarName : np.array(RomsNC.variables[VarName][:, :, lat_li:lat_ui, lon_li:lon_ui], dtype = np.float64), \
            # psi variables (on edges)
            'lat_edge': np.array(RomsNC.variables['lat_psi'][latpsi_li:latpsi_ui, lonpsi_li:lonpsi_ui], dtype = np.float64), \
            'lon_edge': np.array(RomsNC.variables['lon_psi'][latpsi_li:latpsi_ui, lonpsi_li:lonpsi_ui], dtype = np.float64), \
            'u' : np.array(RomsNC.variables['u'][:, :, latpsi_li:latpsi_ui, lonpsi_li:lonpsi_ui], dtype = np.float64), \
            #'ubar' : np.array(RomsNC.variable['ubar'][:, latpsi_li:latpsi_ui, lonpsi_li:lonpsi_ui], dtype = np.float64), \
            'v' : np.array(RomsNC.variables['v'][:, :, latpsi_li:latpsi_ui, lonpsi_li:lonpsi_ui], dtype = np.float64), \
            #'vbar' : np.array(RomsNC.variables['vbar'][:, latpsi_li:latpsi_ui, lonpsi_li:lonpsi_ui], dtype = np.float64)
            }
        
    return ROMS

def Reynolds_Avg(Var):
    """takes temporal average at each grid cell and subtracts to get avg and prime terms """
    # temporal mean
    Var_Tmean = np.mean(Var, axis = 0)
    
    # prime value
    Var_prime = Var - Var_Tmean
    
    ReynAvg = {'bar' : Var_Tmean, 'prime' : Var_prime}
    
    return ReynAvg

