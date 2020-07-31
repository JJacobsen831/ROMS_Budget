# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:16:12 2020

@author: Jasen
"""
import os
os.chdir('/Users/Jasen/Documents/GitHub/ROMS_Budget/')
import numpy as np
import Differential as df
import Gradients as gr
import ROMS_Tools_Mask as rt

# files
RomsFile = '/Users/Jasen/Documents/Data/ROMS_TestRuns/wc12_his_5day.nc'
RomsGrd = '/Users/Jasen/Documents/Data/ROMS_TestRuns/wc12_grd.nc.0'

# variable
varname = 'salt'

#bounds
latbounds = [35, 37]
lonbounds = [-126, -125]

# define mask
RhoMask, PsiMask = rt.RhoPsiIndex_Mask(RomsFile, latbounds, lonbounds)

#load and subset variable
Salt = rt.ROMS_CV(varname, RomsFile, RhoMask) #masked array too large to work with...

S_bar = np.mean(Salt, axis = 3)

salt = RomsNC.variables['salt'][:]

sbar = np.mean(salt[4,:,:,:])