# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:16:12 2020

@author: Jasen
"""
import os
os.chdir('/Users/Jasen/Documents/GitHub/ROMS_Budget/')

import numpy.ma as ma
import Budget_Terms as bud
import ROMS_Tools_Mask as rt

import Differential as df
import Gradients as gr
import Filters as flt
import numpy as np

# files
RomsFile = '/Users/Jasen/Documents/Data/ROMS_TestRuns/wc12_his_5day.nc'
RomsGrd = '/Users/Jasen/Documents/Data/ROMS_TestRuns/wc12_grd.nc.0'

# variable
varname = 'salt'

#bounds of control volume
latbounds = [35, 37]
lonbounds = [-126, -125]

# define mask
RhoMask, _ = rt.RhoPsiIndex_Mask(RomsFile, latbounds, lonbounds)

#load and subset variable
salt = rt.ROMS_CV(varname, RomsFile, RhoMask) 

#deviation from volume mean in each time step
_prime = bud.Prime(salt)
S_prime = ma.array(_prime, mask = RhoMask)

#variance
S_var = S_prime*S_prime   
       
#Term 1: change in variance within control volume
dVar_dt = bud.TermOne(RomsFile, RhoMask, S_var)

#Term 2: Flux of variance across boundary
#define masks
NorthFace, WestFace, SouthFace, EastFace = rt.FaceMask(RomsFile,\
                                                       latbounds, lonbounds)

#>>>>>>>>>masks on u, v points may not have valid entrys<<<<<<<<<<<<<<<<<<<<<<<

#subset variance
_NorthPrime = ma.array(_prime, mask = NorthFace['rho'])
_NorthVar = _NorthPrime**2
# shift rho to v points
NorthVar = 0.5*(_NorthVar[:,:, 0:_NorthVar.shape[2]-1, :] + \
                _NorthVar[:,:, 1:_NorthVar.shape[2], :])

#west
_WestPrime = ma.array(_prime, mask = WestFace['rho'])
_WestVar = _WestPrime**2
#shift rho to u points
WestVar = 0.5*(_WestVar[:,:, :, 0:_WestVar.shape[3]-1] + \
                _WestVar[:,:, :, 1:_WestVar.shape[3]])

#south
_SouthPrime = ma.array(_prime, mask = SouthFace['rho'])
_SouthVar = _SouthPrime**2
SouthVar = 0.5*(_SouthVar[:,:, 0:_SouthVar.shape[2]-1, :] + \
                _SouthVar[:,:, 1:_SouthVar.shape[2], :])

#east
_EastPrime = ma.array(_prime, mask = EastFace['rho'])
_EastVar = _EastPrime**2
EastVar = 0.5*(_EastVar[:,:, :, 0:_EastVar.shape[3]-1] + \
                _EastVar[:,:, :, 1:_EastVar.shape[3]])


#compute differential areas
Ax_norm, Ay_norm = df.dA(RomsFile, RomsGrd)

#subset areas
North_Ay = ma.array(Ay_norm, mask = NorthFace['v'])
West_Ax = ma.array(Ax_norm, mask = WestFace['u'])
South_Ay = ma.array(Ay_norm, mask = SouthFace['v'])
East_Ax = ma.array(Ax_norm, mask = EastFace['u'])

#velocities
North_V = ma.array(RomsNC.variables['v'][:], mask =  NorthFace['v'])
West_U = ma.array(RomsNC.variables['u'][:], mask = WestFace['u'])
South_V = ma.array(RomsNC.variables['v'][:], mask = SouthFace['v'])
East_U = ma.array(RomsNC.variables['u'][:], mask = EastFace['u'])

#multiply to get integrad
North = North_V*North_Ay*NorthVar
West = West_U*West_Ax*WestVar
South = South_V*South_Ay*SouthVar
East = East_U*East_Ax*EastVar

#sum/integrate each time step
Flux = np.empty(North.shape[0])
Flux.fill(np.nan)
for n in range(North.shape[0]) :
    Flux[n] = np.ma.sum(np.ma.sum(np.ma.sum(North[n,:,:,:], axis =0), axis = 0).shape, axis = 0) + np.sum(West[n,:,:,:]) + \
    np.sum(South[n,:,:,:]) + np.sum(East[n, :, :, :])

