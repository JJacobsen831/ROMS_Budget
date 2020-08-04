# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:12:20 2020

@author: Jasen
"""
import os
os.chdir('/Users/Jasen/Documents/GitHub/ROMS_Budget/')
import numpy as np
import Gradients as gr
import matplotlib.pyplot as plt
from netCDF4 import Dataset as nc4

#files
RomsFile = '/Users/Jasen/Documents/Data/ROMS_TestRuns/wc12_his_5day.nc'
RomsGrd = '/Users/Jasen/Documents/Data/ROMS_TestRuns/wc12_grd.nc.0'

#select variable
varname = 'salt'

#load bathymetry
RomsNC = nc4(RomsFile, 'r')
bathy = RomsNC.variables['h'][:]

#horizontal gradients
ds_dx = gr.x_grad(RomsFile, RomsGrd, varname)
ds_dy = gr.y_grad(RomsFile, RomsGrd, varname)

#grid corrections
xCor = gr.x_grad_GridCor(RomsFile, RomsGrd, varname)
yCor = gr.y_grad_GridCor(RomsFile, RomsGrd, varname)

#ratio
x_rat = np.array(np.abs(xCor)/np.abs(ds_dx))
y_rat = np.array(np.abs(yCor)/np.abs(ds_dy))

#median ratio
DM_ratx = np.median(x_rat[0,:,:,:], axis = 0)

#log median ration, skipping zeros
for n in range(DM_ratx.shape[0]) :
    ind = DM_ratx[n,:] != 0
    DM_ratx[n,ind] = np.log10(DM_ratx[n,ind])

DM_raty = np.median(y_rat[0,:,:,:], axis = 0)
for n in range(DM_raty.shape[0]) :
    ind = DM_raty[n,:] != 0
    DM_raty[n,ind] = np.log10(DM_raty[n, ind])

#plotting
fig1, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, constrained_layout=True)
#subplot 1
CS = ax1.contourf(DM_ratx, cmap = plt.cm.inferno)
CS1 = ax1.contour(bathy, cmap = plt.cm.gray)
ax1.set_title('GridCor:ds_dx')
#subplot 2
CS = ax2.contourf(DM_raty, cmap = plt.cm.inferno)
CS1 = ax2.contour(bathy, cmap = plt.cm.gray) 
ax2.set_title('GridCor:ds_dy')
fig1.colorbar(CS)
fig1.suptitle('Magnitude of Depth Median of Grid Corrention')

plt.figure(figsize=(20,10))
fig1.savefig('GridCor_Map.jpg')