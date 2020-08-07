# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:27:51 2020

@author: Jasen
Tools to compute gradients within a control volume defined in roms output
"""
import numpy as np
from netCDF4 import Dataset as nc4
import obs_depth_JJ as dep
import ROMS_Tools_Mask as rt

def x_grad(RomsFile, RomsGrd, varname) :
    """
    Compute x-gradient assuming rectangular grid cells
    """
    if type(varname) == str :
        #load roms file
        RomsNC = nc4(RomsFile, 'r')
        #load variable
        var = RomsNC.variables[varname][:]
        
    else:
        var = varname
    
    #gradient in x direction on u points
    dvar_u = np.diff(var, n = 1, axis = 3)
    
    #pad ends
    dvar_pad = np.concatenate((dvar_u[:,:,:,0:1], dvar_u, dvar_u[:,:,:,-2:-1 ]), axis = 3)
    
    #average to rho points
    dvar = 0.5*(dvar_pad[:,:,:,0:dvar_pad.shape[3]-1] + dvar_pad[:,:,:,1:dvar_pad.shape[3]])
    
    #lon positions [meters]
    x_dist = rt.rho_dist_grd(RomsGrd)[0]
    
    #repeat over depth and time space
    dx = rt.AddDepthTime(RomsFile, x_dist)
    
    #compute gradient
    dvar_dx = dvar/dx
    
    return dvar_dx

def y_grad(RomsFile, RomsGrd, varname):
    """
    Compute y-gradient assuming rectangular grid cells
    """
    if type(varname) == str :
        #load roms file
        RomsNC = nc4(RomsFile, 'r')
        #load variable
        var = RomsNC.variables[varname][:]
        
    else:
        var = varname
    
    #compute difference
    dvar_v = np.diff(var, n = 1, axis = 2)
    
    #pad ends
    dvar_pad = np.concatenate((dvar_v[:,:,0:1, :], dvar_v, dvar_v[:,:,-2:-1, :]), axis = 2)
    
    #average to rho points
    dvar = 0.5*(dvar_pad[:,:,0:dvar_pad.shape[2]-1, :] + dvar_pad[:,:,1:dvar_pad.shape[3], :])    
    
    #lon positions [meters]
    y_dist = rt.rho_dist_grd(RomsGrd)[1]
    
    #repeat over depth and time space
    dy = rt.AddDepthTime(RomsFile, y_dist)
    
    #compute gradient
    dvar_dy = dvar/dy
    
    return dvar_dy

def z_grad(RomsFile, varname) :
    """
    Compute z-gradient assuming rectangular grid cells
    """
    #load roms file
    RomsNC = nc4(RomsFile, 'r')
    
    if type(varname) == str :
        #load variable
        var = RomsNC.variables[varname][:]
        
    else:
        var = varname
    
    #vertical difference
    dvar_w = np.diff(var, n = 1, axis = 1)
    
    #pad ends
    dvar_pad = np.concatenate((dvar_w[:,0:1, :, :], dvar_w, dvar_w[:,-2:-1, :, :]), axis = 1)
    
    #average to rho points
    dvar = 0.5*(dvar_pad[:,0:dvar_pad.shape[1]-1,:, :] + dvar_pad[:,1:dvar_pad.shape[1], :, :])    
    
    #depth [meters]
    depth = dep._set_depth_T(RomsFile, None, 'w', \
                             RomsNC.variables['h'], RomsNC.variables['zeta'])
    d_dep = np.diff(depth, n = 1, axis = 1)
    
    #compute gradient
    dvar_dz = dvar/d_dep
    
    return dvar_dz

def Wpt_to_Upt(dvar_dz, direction) :
    """
    convert 'W-point' from vertical difference to 'U,V point'
    """
    #horzontal average to box points
    if direction in [0, 'U', 'u', 'lon'] :
        dvar_dz_box = 0.5*(dvar_dz[:,:,:, 0:dvar_dz.shape[3]-1] + \
                           dvar_dz[:,:,:, 1:dvar_dz.shape[3]])
        
    elif direction in [1, 'V', 'v', 'lat'] :
        dvar_dz_box = 0.5*(dvar_dz[:,:, 0:dvar_dz.shape[2]-1, :] + \
                           dvar_dz[:,:, 1:dvar_dz.shape[2], :])
    
    #pad ends on top and bottom of array with adjacent values
    dvar_dz_pad = np.concatenate((dvar_dz_box[:, 0:1, :, :], dvar_dz_box,\
                                  dvar_dz_box[:, -2:-1, :, :]), axis = 1)
    
    #average box points onto X points
    dvar_dz_U = 0.5*(dvar_dz_pad[:, 0:dvar_dz_pad.shape[1]-1, :, :] + \
                   dvar_dz_pad[:, 1:dvar_dz_pad.shape[1], :, :])
    
    return dvar_dz_U
    

def x_grad_GridCor(RomsFile, RomsGrd, varname) :
    """
    Compute Grid Correction for gradients in x direction
    
    """
    #load roms file
    RomsNC = nc4(RomsFile, 'r')
    
    #compute depth at rho points
    _depth = dep._set_depth_T(RomsFile, None, 'rho', \
                              RomsNC.variables['h'],RomsNC.variables['zeta'])
    
    #depth difference in vertical
    dz_z = np.diff(_depth, n = 1, axis = 1)
    
    #depth difference of adjacent grid points
    dz_x = np.diff(_depth, n = 1, axis = 3)
    
    #load variable 
    if type(varname) == str :
        _var = RomsNC.variables[varname][:]
    else :
        _var = varname
    
    # compute differential
    dvar_z = np.diff(_var, n = 1, axis = 1)
    
    #distance between rho points in x direction
    _x_dist = dist(RomsGrd)[0]
    
    #repeat over depth and time space
    _DX = np.repeat(np.array(_x_dist)[np.newaxis, :, :], _depth.shape[1], \
                    axis = 0)
    dx = np.repeat(np.array(_DX)[np.newaxis, :, :, :], _depth.shape[0], \
                   axis = 0)
    
    #vertical gradient on tri points
    dvar_dz_tri = dvar_z/dz_z
    
    #average to X points
    dvar_dz = Wpt_to_Upt(dvar_dz_tri, 'U')
    
    #correction for roms grid
    dv_dxCor = dvar_dz*(dz_x/dx)
    
    return dv_dxCor

def y_grad_GridCor(RomsFile, RomsGrd, varname) :
    """
    Compute Grid Correction for gradients in y direction
    
    """
    #load roms file
    RomsNC = nc4(RomsFile, 'r')
    
    #compute depth at rho points
    _depth = dep._set_depth_T(RomsFile, None, 'rho', \
                              RomsNC.variables['h'],RomsNC.variables['zeta'])
    
    #depth difference in vertical
    dz_z = np.diff(_depth, n = 1, axis = 1)
    
    #depth difference between adjacent rho points
    dz_y = np.diff(_depth, n = 1, axis = 2)
        
    #load variable 
    if type(varname) == str :
        _var = RomsNC.variables[varname][:]
    else :
        _var = varname
    
    #compute difference
    dvar_z = np.diff(_var, n = 1, axis = 1)
    
    #distance between rho points in x and y directions
    _y_dist = dist(RomsGrd)[1]
    
    #repeat over depth and time space
    _DY = np.repeat(np.array(_y_dist)[np.newaxis, :, :], _depth.shape[1], axis = 0)
    dy = np.repeat(np.array(_DY)[np.newaxis, :, :, :], _depth.shape[0], axis = 0)
    
    # vertical gradient on tri points
    #vertical gradient on tri points
    dvar_dz_tri = dvar_z/dz_z
    
    #average to X points
    dvar_dz = Wpt_to_Upt(dvar_dz_tri, 'V')
        
    #correction for roms grid
    dv_dyCor = dvar_dz*(dz_y/dy)
    
    return dv_dyCor
