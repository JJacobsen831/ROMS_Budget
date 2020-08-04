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

#subset variance
_NorthPrime = ma.array(_prime, mask = NorthFace)
NorthVar = _NorthPrime**2

_WestPrime = ma.array(_prime, mask = WestFace)
WestVar = _WestPrime**2

_SouthPrime = ma.array(_prime, mask = SouthFace)
SouthVar = _SouthPrime**2

_EastPrime = ma.array(_prime, mask = EastFace)
EastVar = _EastPrime**2

#subset differential areas
A_xz, A_yz = df.dA_int_w(RomsFile) #area code need to be repeated over depth and time




