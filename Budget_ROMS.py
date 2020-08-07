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
RhoMask, _, _ = rt.RhoUV_Mask(RomsFile, latbounds, lonbounds)

#load and subset variable
salt = rt.ROMS_CV(varname, RomsFile, RhoMask) 

#deviation from volume mean in each time step
_prime = bud.Prime(salt)
S_prime = ma.array(_prime, mask = RhoMask)

#variance squared
S_var2 = S_prime*S_prime   
       
#Term 1: change in variance within control volume
dVar_dt = bud.TermOne(RomsFile, RhoMask, S_var2)

#Term 2: Flux of variance across boundary
Flux = bud.TermTwo(RomsFile, RomsGrd, varname, latbounds, lonbounds)

#Term 4: Internal mixing - destruction of variance
Mixing = bud.TermFour(RomsFile, RomsGrd, S_prime)