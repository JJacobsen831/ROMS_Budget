# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 12:29:58 2020

@author: Jasen
"""
# ROMS data extraction for CV budget
import numpy as np
from netCDF4 import Dataset as dt
import seawater as sw
import obs_depth_JJ as dep

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
    
    ReynAvg = {'avg' : Var_Tmean, 'prime' : Var_prime}
    
    return ReynAvg

def ModelDepth(RomsFile, point_type):
    """Computes ROMS depth using obs_depth, converted from set_depth.m"""
    #load nc file
    RomsNC = dt(RomsFile, 'r')
    
    #ROMS variables
    romsvars = {'Vtransform' : RomsNC.variables['Vtransform'], \
                'Vstretching' : RomsNC.variables['Vstretching'], \
                'N' : RomsNC.variables['s_rho'].size, \
                'theta_s' : RomsNC.variables['theta_s'], \
                'theta_b' : RomsNC.variables['theta_b'], \
                'hc' : RomsNC.variables['hc'], \
                'h' : RomsNC.variables['h'], \
                'zeta' : RomsNC.variables['zeta']}
    #compute depth
    depth = dep._set_depth(RomsFile, romsvars, point_type, romsvars['h'], romsvars['zeta'])
    
    return depth



def dV(ROMS_CV):
    """Compute differential volume of each cell within a control volume"""
    #convert horizontal location into distance b/t cells
    dist_lat = sw.dist(ROMS_CV['lat_edge'][:,0], ROMS_CV['lon_edge'][:,0], 'km').T
    dist_lon = sw.dist(ROMS_CV['lat_edge'][0,:], ROMS_CV['lon_edge'][0,:], 'km')
    
    #horizontal differential matrix
    d_lat = np.matmul(np.array([dist_lat[0]]).T, np.ones([1, dist_lon[0].size]))
    d_lon = np.matmul(np.array([dist_lon[0]]).T, np.ones([1, dist_lat[0].size])).T
    
    #convert vertical coordinate into difference