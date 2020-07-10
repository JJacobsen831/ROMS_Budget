# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 12:27:28 2020

@author: Jasen
Tools used to create differential volumes and areas of each cell in roms output
"""
import numpy as np
from netCDF4 import Dataset as dt
import obs_depth_JJ as dep

def dV(RomsFile) :
    """ Load full roms grid and compute differential volume of each cell"""
    RomsNC = dt(RomsFile, 'r')

    #compute z
    romsvars = {'h' : RomsNC.variables['h'], \
                'zeta' : RomsNC.variables['zeta']}
    
    #compute depth at w points
    depth_domain = dep._set_depth(RomsFile, None, 'w', romsvars['h'], romsvars['zeta'])
        
    dz = np.diff(depth_domain, n = 1, axis = 0)
            
    #compute lengths of horizontal cell directions
    dx = np.repeat(1/np.array(RomsNC.variables['pm'])[np.newaxis, :, :], dz.shape[0], axis = 0)
    dy = np.repeat(1/np.array(RomsNC.variables['pn'])[np.newaxis, :, :], dz.shape[0], axis = 0)
    
    #compute differential volume of each cell
    DV = dx*dy*dz
    
    return DV

def dAz(RomsFile) :
    """Compute differential area of each cell"""
    #load roms file
    RomsNC = dt(RomsFile, 'r')
    
    #compute z
    romsvars = {'h' : RomsNC.variables['h'], \
                'zeta' : RomsNC.variables['zeta']}
    
    #compute depth at w points
    depth_domain = dep._set_depth(RomsFile, None, 'w', romsvars['h'], romsvars['zeta'])
        
    dz = np.diff(depth_domain, n = 1, axis = 0)
            
    #compute lengths of horizontal cell directions repeat over depth axis
    dx = np.repeat(1/np.array(RomsNC.variables['pm'])[np.newaxis, :, :], dz.shape[0], axis = 0)
    dy = np.repeat(1/np.array(RomsNC.variables['pn'])[np.newaxis, :, :], dz.shape[0], axis = 0)
    
    #compute differential area assuming trapizoidal shape
    Axz = np.empty(dx.shape)
    Axz.fill(np.nan)
    Ayz = Axz
    
    for k in range(1, dx.shape[2]):
        Axz[:,:,k-1] = 0.5*(dz[:,:, k-1]+dz[:, :, k])*dx[:,:,k-1]
    
    for k in range(1, dy.shape[1]):
        Ayz[:, k, :] = 0.5*(dz[:, k-1, :]+ dz[:, k, :])*dy[:, k-1,:]
    
    #x area and y area
    
        
    return DA


    