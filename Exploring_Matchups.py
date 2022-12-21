#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 16:31:16 2022

@author: nicolausf
"""


import glob, os
import pandas as pd
import numpy as np
import gsw
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar
import urllib.request
from zipfile import ZipFile
from re import search
from scipy import ndimage
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.path as mpath
import re
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import scipy.io
import scipy.interpolate as interp

# This code currently uses the eddy_mathcup output from Chelton, and not from
#   Veronica's updated function 8/22


# Location where you want figures to save:
output_path = 'generated/'


# Read in pickle created in SOCCOM_eddy_matchups
u_stat_SOCCOM = pd.read_pickle('data/u_stat_SOCCOM.pkl')

# Cyclonic Vs. Anticyclonic Eddies
eddy_ID_cyclonic = u_stat_SOCCOM.loc[(u_stat_SOCCOM['eddy_type'] !=0) 
                                     & (u_stat_SOCCOM['eddy_type'] == -1)
                                     ]

eddy_ID_anticyclonic = u_stat_SOCCOM.loc[(u_stat_SOCCOM['eddy_type'] !=0) 
                                     & (u_stat_SOCCOM['eddy_type'] == 1)
                                     ]

non_eddy = u_stat_SOCCOM.loc[(u_stat_SOCCOM['eddy_type'] ==0)] 


# ---------  Creating a Map of all floats with eddies -----------

# Maps below are created to show cyclonic and anticyclonic eddies during winter,
#   spring, summer, and fall seasons to get an idea of how many eddies are in
#   a region during that season.  This will help us to make informed decisions
#   on whether or not there is enough data in a specific region at a certain time
#   to examine further (obtaining means, standard deviations and historgrams of 
#   profiles)


## Import Front Data

# Subtropical Front
stf=pd.read_csv('Front_Txt_Files/stf.txt', header=None,sep='\s+', na_values='%',
                names=['lon','lat'])
# Subantarctic Front
saf=pd.read_csv('Front_Txt_Files/saf.txt', header=None,sep='\s+', na_values='%',
                names=['lon','lat'])
# Polar Front
pf=pd.read_csv('Front_Txt_Files/pf.txt', header=None,sep='\s+', na_values='%',
               names=['lon','lat'])
# Southern Antarctic Cirumpolar Current Front
saccf=pd.read_csv('Front_Txt_Files/saccf.txt', header=None,sep='\s+', na_values='%',
                  names=['lon','lat'])
# Southern Boundary
sbdy=pd.read_csv('Front_Txt_Files/sbdy.txt', header=None,sep='\s+', na_values='%',
                 names=['lon','lat'])





# --------------- Anticyclonic Plots using Orsi Fronts -----------------#


### Create Figure of all Anticyclonic Eddies South PolarStereo (season ignored) ###

# Set Figure and Map Characteristics
plt.figure(figsize=(6, 6))
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([-180,180,-90,-30], ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
ax.gridlines()

# Create Circle Axes Coordinates
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

# Plot Fronts (uses circle axes from ARGO Float Track)
ax.set_boundary(circle, transform=ax.transAxes)
plt.plot(stf['lon'], stf['lat'], color='Red', transform=ccrs.PlateCarree(), 
         label='Subtropical Front')
plt.plot(saf['lon'], saf['lat'], color='Orange', transform=ccrs.PlateCarree(), 
         label='Subantarctic Front')
plt.plot(pf['lon'], pf['lat'], color='Yellow', transform=ccrs.PlateCarree(), 
         label='Polar Front')
plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree(), 
         label='Southern ACC Front')
plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree(), 
         label='Southern Boundary')
plt.legend()

# Plot ARGO Float Track (taken from ARGO Float Track)
plt.scatter(eddy_ID_anticyclonic['Lon [°E]'], eddy_ID_anticyclonic['Lat [°N]'], 
            color='Black', transform=ccrs.PlateCarree(), s = 5, zorder = 1000)
plt.title('Floats in Anticyclonic Eddies and Fronts in the Southern Ocean')
plt.savefig(output_path +'ACE_wFronts_SPS' + '.png')


### Create Figure of all Anticyclonic Eddies PlateCaree (season ignored) ###

# Set Figure and Map Characteristics
plt.figure(figsize=(9,3))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([0,140,-70,-30], ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
gl = ax.gridlines()
gl.ylabels_left = True
gl.xlabels_bottom = True

# Plot Fronts (uses circle axes from ARGO Float Track)
#ax.set_boundary(circle, transform=ax.transAxes)
plt.plot(stf['lon'], stf['lat'], color='Red', transform=ccrs.PlateCarree(), 
         label='Subtropical Front')
plt.plot(saf['lon'], saf['lat'], color='Orange', transform=ccrs.PlateCarree(), 
         label='Subantarctic Front')
plt.plot(pf['lon'], pf['lat'], color='Yellow', transform=ccrs.PlateCarree(), 
         label='Polar Front')
plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree(), 
         label='Southern ACC Front')
plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree(), 
         label='Southern Boundary')
#plt.legend()

# Plot ARGO Float Track (taken from ARGO Float Track)
plt.scatter(eddy_ID_anticyclonic['Lon [°E]'], eddy_ID_anticyclonic['Lat [°N]'], 
            color='Black', transform=ccrs.PlateCarree(), s = 5,  zorder = 1000) # change dot size
plt.title('Floats in Anticyclonic Eddies and Fronts in the Southern Ocean')
plt.savefig(output_path +'ACE_wFronts_PC' + '.png')




### Create Figure of all Anticyclonic Eddies in winter ###

# Create a subset with set Boundaries on Time to collect only winter months (July, August, September)
eddy_ID_anti_winter = eddy_ID_anticyclonic.loc[(eddy_ID_anticyclonic['mon/day/yr'] >= '07/01')
                                               & (eddy_ID_anticyclonic['mon/day/yr'] <= '09/31')]

# Set Figure and Map Characteristics
plt.figure(figsize=(9,3))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([0,140,-70,-30], ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
gl = ax.gridlines()
gl.ylabels_left = True
gl.xlabels_bottom = True

# Plot Fronts (uses circle axes from ARGO Float Track)
#ax.set_boundary(circle, transform=ax.transAxes)
plt.plot(stf['lon'], stf['lat'], color='Red', transform=ccrs.PlateCarree(), 
         label='Subtropical Front')
plt.plot(saf['lon'], saf['lat'], color='Orange', transform=ccrs.PlateCarree(), 
         label='Subantarctic Front')
plt.plot(pf['lon'], pf['lat'], color='Yellow', transform=ccrs.PlateCarree(), 
         label='Polar Front')
plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree(), 
         label='Southern ACC Front')
plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree(), 
         label='Southern Boundary')
#plt.legend()

# Plot ARGO Float Track (taken from ARGO Float Track)
plt.scatter(eddy_ID_anti_winter['Lon [°E]'], eddy_ID_anti_winter['Lat [°N]'], 
            color='Black', transform=ccrs.PlateCarree(), s = 5, zorder = 1000) # change dot size
plt.title('Floats in Anticyclonic Eddies and Fronts in the Southern Ocean (Winter)')
plt.savefig(output_path +'ACEwinter_wFronts_PC' + '.png')



### Create Figure of all Anticyclonic Eddies in Spring ###

# Create a subset with set Boundaries on Time to collect only winter months (October, November, December)
eddy_ID_anti_spring = eddy_ID_anticyclonic.loc[(eddy_ID_anticyclonic['mon/day/yr'] >= '10/01')
                                               & (eddy_ID_anticyclonic['mon/day/yr'] <= '12/31')]


# Set Figure and Map Characteristics
plt.figure(figsize=(9,3))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([0,140,-70,-30], ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
gl = ax.gridlines()
gl.ylabels_left = True
gl.xlabels_bottom = True

# Plot Fronts (uses circle axes from ARGO Float Track)
#ax.set_boundary(circle, transform=ax.transAxes)
plt.plot(stf['lon'], stf['lat'], color='Red', transform=ccrs.PlateCarree(), 
         label='Subtropical Front')
plt.plot(saf['lon'], saf['lat'], color='Orange', transform=ccrs.PlateCarree(), 
         label='Subantarctic Front')
plt.plot(pf['lon'], pf['lat'], color='Yellow', transform=ccrs.PlateCarree(), 
         label='Polar Front')
plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree(), 
         label='Southern ACC Front')
plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree(), 
         label='Southern Boundary')
#plt.legend()

# Plot ARGO Float Track (taken from ARGO Float Track)
plt.scatter(eddy_ID_anti_spring['Lon [°E]'], eddy_ID_anti_spring['Lat [°N]'], 
            color='Black', transform=ccrs.PlateCarree(), s = 5, zorder = 1000) # change dot size
plt.title('Floats in Anticyclonic Eddies and Fronts in the Southern Ocean (Spring)')
plt.savefig(output_path +'ACEspring_wFronts_PC' + '.png')



### Create Figure of all Anticyclonic Eddies in Summer ###

# Create a subset with set Boundaries on Time to collect only winter months (Jan, Feb, March)
eddy_ID_anti_summer = eddy_ID_anticyclonic.loc[(eddy_ID_anticyclonic['mon/day/yr'] >= '01/01')
                                               & (eddy_ID_anticyclonic['mon/day/yr'] <= '03/31')]


# Set Figure and Map Characteristics
plt.figure(figsize=(9,3))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([0,140,-70,-30], ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
gl = ax.gridlines()
gl.ylabels_left = True
gl.xlabels_bottom = True

# Plot Fronts (uses circle axes from ARGO Float Track)
#ax.set_boundary(circle, transform=ax.transAxes)
plt.plot(stf['lon'], stf['lat'], color='Red', transform=ccrs.PlateCarree(), 
         label='Subtropical Front')
plt.plot(saf['lon'], saf['lat'], color='Orange', transform=ccrs.PlateCarree(), 
         label='Subantarctic Front')
plt.plot(pf['lon'], pf['lat'], color='Yellow', transform=ccrs.PlateCarree(), 
         label='Polar Front')
plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree(), 
         label='Southern ACC Front')
plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree(), 
         label='Southern Boundary')
#plt.legend()

# Plot ARGO Float Track (taken from ARGO Float Track)
plt.scatter(eddy_ID_anti_summer['Lon [°E]'], eddy_ID_anti_summer['Lat [°N]'], 
            color='Black', transform=ccrs.PlateCarree(), s = 5, zorder = 1000) # change dot size
plt.title('Floats in Anticyclonic Eddies and Fronts in the Southern Ocean (Summer)')
plt.savefig(output_path +'ACEsummer_wFronts_PC' + '.png')



### Create Figure of all Anticyclonic Eddies in Fall ###

# Create a subset with set Boundaries on Time to collect only winter months (APril, May, June)
eddy_ID_anti_fall = eddy_ID_anticyclonic.loc[(eddy_ID_anticyclonic['mon/day/yr'] >= '04/01')
                                               & (eddy_ID_anticyclonic['mon/day/yr'] <= '06/30')]


# Set Figure and Map Characteristics
plt.figure(figsize=(9,3))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([0,140,-70,-30], ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
gl = ax.gridlines()
gl.ylabels_left = True
gl.xlabels_bottom = True

# Plot Fronts (uses circle axes from ARGO Float Track)
#ax.set_boundary(circle, transform=ax.transAxes)
plt.plot(stf['lon'], stf['lat'], color='Red', transform=ccrs.PlateCarree(), 
         label='Subtropical Front')
plt.plot(saf['lon'], saf['lat'], color='Orange', transform=ccrs.PlateCarree(), 
         label='Subantarctic Front')
plt.plot(pf['lon'], pf['lat'], color='Yellow', transform=ccrs.PlateCarree(), 
         label='Polar Front')
plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree(), 
         label='Southern ACC Front')
plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree(), 
         label='Southern Boundary')
#plt.legend()

# Plot ARGO Float Track (taken from ARGO Float Track)
plt.scatter(eddy_ID_anti_fall['Lon [°E]'], eddy_ID_anti_fall['Lat [°N]'], 
            color='Black', transform=ccrs.PlateCarree(), s = 5, zorder = 1000) # change dot size
plt.title('Floats in Anticyclonic Eddies and Fronts in the Southern Ocean (Fall)')
plt.savefig(output_path +'ACEfall_wFronts_PC' + '.png')








# --------------- Cyclonic Plots using Orsi Fronts -----------------#





### Create Figure of all Cyclonic Eddies South PolarStereo (season ignored) ###

# Set Figure and Map Characteristics
plt.figure(figsize=(6, 6))
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([-180,180,-90,-30], ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
ax.gridlines()

# Create Circle Axes Coordinates
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

# Plot Fronts (uses circle axes from ARGO Float Track)
ax.set_boundary(circle, transform=ax.transAxes)
plt.plot(stf['lon'], stf['lat'], color='Red', transform=ccrs.PlateCarree(), 
         label='Subtropical Front')
plt.plot(saf['lon'], saf['lat'], color='Orange', transform=ccrs.PlateCarree(), 
         label='Subantarctic Front')
plt.plot(pf['lon'], pf['lat'], color='Yellow', transform=ccrs.PlateCarree(), 
         label='Polar Front')
plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree(), 
         label='Southern ACC Front')
plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree(), 
         label='Southern Boundary')
plt.legend()

# Plot ARGO Float Track (taken from ARGO Float Track)
plt.scatter(eddy_ID_cyclonic['Lon [°E]'], eddy_ID_cyclonic['Lat [°N]'], 
            color='Black', transform=ccrs.PlateCarree(), s = 5, zorder = 1000)
plt.title('Floats in Cyclonic Eddies and Fronts in the Southern Ocean')
plt.savefig(output_path +'CE_wFronts_SPS' + '.png')




### Create Figure of all Cyclonic Eddies PlateCaree (season ignored) ###

# Set Figure and Map Characteristics
plt.figure(figsize=(9,3))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([0,140,-70,-30], ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
gl = ax.gridlines()
gl.ylabels_left = True
gl.xlabels_bottom = True

# Plot Fronts (uses circle axes from ARGO Float Track)
#ax.set_boundary(circle, transform=ax.transAxes)
plt.plot(stf['lon'], stf['lat'], color='Red', transform=ccrs.PlateCarree(), 
         label='Subtropical Front')
plt.plot(saf['lon'], saf['lat'], color='Orange', transform=ccrs.PlateCarree(), 
         label='Subantarctic Front')
plt.plot(pf['lon'], pf['lat'], color='Yellow', transform=ccrs.PlateCarree(), 
         label='Polar Front')
plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree(), 
         label='Southern ACC Front')
plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree(), 
         label='Southern Boundary')
#plt.legend()

# Plot ARGO Float Track (taken from ARGO Float Track)
plt.scatter(eddy_ID_cyclonic['Lon [°E]'], eddy_ID_cyclonic['Lat [°N]'], 
            color='Black', transform=ccrs.PlateCarree(), s = 5,  zorder = 1000) # change dot size
plt.title('Floats in Cyclonic Eddies and Fronts in the Southern Ocean')
plt.savefig(output_path +'CE_wFronts_PC' + '.png')




### Create Figure of all Cyclonic Eddies in Winter ###

# Create a subset with set Boundaries on Time to collect only winter months (July, August, September)
eddy_ID_c_winter = eddy_ID_cyclonic.loc[(eddy_ID_cyclonic['mon/day/yr'] >= '07/01')
                                               & (eddy_ID_cyclonic['mon/day/yr'] <= '09/31')]


# Set Figure and Map Characteristics
plt.figure(figsize=(9,3))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([0,140,-70,-30], ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
gl = ax.gridlines()
gl.ylabels_left = True
gl.xlabels_bottom = True

# Plot Fronts (uses circle axes from ARGO Float Track)
#ax.set_boundary(circle, transform=ax.transAxes)
plt.plot(stf['lon'], stf['lat'], color='Red', transform=ccrs.PlateCarree(), 
         label='Subtropical Front')
plt.plot(saf['lon'], saf['lat'], color='Orange', transform=ccrs.PlateCarree(), 
         label='Subantarctic Front')
plt.plot(pf['lon'], pf['lat'], color='Yellow', transform=ccrs.PlateCarree(), 
         label='Polar Front')
plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree(), 
         label='Southern ACC Front')
plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree(), 
         label='Southern Boundary')
#plt.legend()

# Plot ARGO Float Track (taken from ARGO Float Track)
plt.scatter(eddy_ID_c_winter['Lon [°E]'], eddy_ID_c_winter['Lat [°N]'], 
            color='Black', transform=ccrs.PlateCarree(), s = 5, zorder = 1000) # change dot size
plt.title('Floats in Cyclonic Eddies and Fronts in the Southern Ocean (Winter)')
plt.savefig(output_path +'CEwinter_wFronts_PC' + '.png')



### Create Figure of all Cyclonic Eddies in Spring ###

# Create a subset with set Boundaries on Time to collect only winter months (October, November, December)
eddy_ID_c_spring = eddy_ID_cyclonic.loc[(eddy_ID_cyclonic['mon/day/yr'] >= '10/01')
                                               & (eddy_ID_cyclonic['mon/day/yr'] <= '12/31')]


# Set Figure and Map Characteristics
plt.figure(figsize=(9,3))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([0,140,-70,-30], ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
gl = ax.gridlines()
gl.ylabels_left = True
gl.xlabels_bottom = True

# Plot Fronts (uses circle axes from ARGO Float Track)
#ax.set_boundary(circle, transform=ax.transAxes)
plt.plot(stf['lon'], stf['lat'], color='Red', transform=ccrs.PlateCarree(), 
         label='Subtropical Front')
plt.plot(saf['lon'], saf['lat'], color='Orange', transform=ccrs.PlateCarree(), 
         label='Subantarctic Front')
plt.plot(pf['lon'], pf['lat'], color='Yellow', transform=ccrs.PlateCarree(), 
         label='Polar Front')
plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree(), 
         label='Southern ACC Front')
plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree(), 
         label='Southern Boundary')
#plt.legend()

# Plot ARGO Float Track (taken from ARGO Float Track)
plt.scatter(eddy_ID_c_spring['Lon [°E]'], eddy_ID_c_spring['Lat [°N]'], 
            color='Black', transform=ccrs.PlateCarree(), s = 5, zorder = 1000) # change dot size
plt.title('Floats in Cyclonic Eddies and Fronts in the Southern Ocean (Spring)')
plt.savefig(output_path +'CEspring_wFronts_PC' + '.png')



### Create Figure of all Cyclonic Eddies in Summer ###

# Create a subset with set Boundaries on Time to collect only winter months (Jan, Feb, March)
eddy_ID_c_summer = eddy_ID_cyclonic.loc[(eddy_ID_cyclonic['mon/day/yr'] >= '01/01')
                                               & (eddy_ID_cyclonic['mon/day/yr'] <= '03/31')]

# Set Figure and Map Characteristics
plt.figure(figsize=(9,3))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([0,140,-70,-30], ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
gl = ax.gridlines()
gl.ylabels_left = True
gl.xlabels_bottom = True

# Plot Fronts (uses circle axes from ARGO Float Track)
#ax.set_boundary(circle, transform=ax.transAxes)
plt.plot(stf['lon'], stf['lat'], color='Red', transform=ccrs.PlateCarree(), 
         label='Subtropical Front')
plt.plot(saf['lon'], saf['lat'], color='Orange', transform=ccrs.PlateCarree(), 
         label='Subantarctic Front')
plt.plot(pf['lon'], pf['lat'], color='Yellow', transform=ccrs.PlateCarree(), 
         label='Polar Front')
plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree(), 
         label='Southern ACC Front')
plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree(), 
         label='Southern Boundary')
#plt.legend()

# Plot ARGO Float Track (taken from ARGO Float Track)
plt.scatter(eddy_ID_c_summer['Lon [°E]'], eddy_ID_c_summer['Lat [°N]'], 
            color='Black', transform=ccrs.PlateCarree(), s = 5, zorder = 1000) # change dot size
plt.title('Floats in Cyclonic Eddies and Fronts in the Southern Ocean (Summer)')
plt.savefig(output_path +'CEsummer_wFronts_PC' + '.png')



### Create Figure of all Cyclonic Eddies in Fall ###

# Create a subset with set Boundaries on Time to collect only winter months (APril, May, June)
eddy_ID_c_fall = eddy_ID_cyclonic.loc[(eddy_ID_cyclonic['mon/day/yr'] >= '04/01')
                                               & (eddy_ID_cyclonic['mon/day/yr'] <= '06/30')]


# Set Figure and Map Characteristics
plt.figure(figsize=(9,3))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([0,140,-70,-30], ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
gl = ax.gridlines()
gl.ylabels_left = True
gl.xlabels_bottom = True

# Plot Fronts (uses circle axes from ARGO Float Track)
#ax.set_boundary(circle, transform=ax.transAxes)
plt.plot(stf['lon'], stf['lat'], color='Red', transform=ccrs.PlateCarree(), 
         label='Subtropical Front')
plt.plot(saf['lon'], saf['lat'], color='Orange', transform=ccrs.PlateCarree(), 
         label='Subantarctic Front')
plt.plot(pf['lon'], pf['lat'], color='Yellow', transform=ccrs.PlateCarree(), 
         label='Polar Front')
plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree(), 
         label='Southern ACC Front')
plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree(), 
         label='Southern Boundary')
#plt.legend()

# Plot ARGO Float Track (taken from ARGO Float Track)
plt.scatter(eddy_ID_c_fall['Lon [°E]'], eddy_ID_c_fall['Lat [°N]'], 
            color='Black', transform=ccrs.PlateCarree(), s = 5, zorder = 1000) # change dot size
plt.title('Floats in Cyclonic Eddies and Fronts in the Southern Ocean (Fall)')
plt.savefig(output_path +'CEfall_wFronts_PC' + '.png')



# ---------- Non Eddy - Seaon Subsets ------------- #
non_eddy_winter = non_eddy.loc[(non_eddy['mon/day/yr'] >= '07/01')
                                               & (non_eddy['mon/day/yr'] <= '09/31')]
non_eddy_spring = non_eddy.loc[(non_eddy['mon/day/yr'] >= '10/01')
                                               & (non_eddy['mon/day/yr'] <= '12/31')]
non_eddy_summer = non_eddy.loc[(non_eddy['mon/day/yr'] >= '01/01')
                                               & (non_eddy['mon/day/yr'] <= '3/31')]
non_eddy_fall = non_eddy.loc[(non_eddy['mon/day/yr'] >= '4/01')
                                               & (non_eddy['mon/day/yr'] <= '6/30')]








# -------------- Load Kim and Orsi Front Data ---------------------- #



f = scipy.io.loadmat('/Users/nicolausf/Desktop/acc_fronts_Kim_Orsi.mat')
fronts_m = f['acc_fronts']

#Subantarctic Front
saf_m = fronts_m['saf'][0][0]
saf_lon = saf_m[:,0]
saf_lon[saf_lon<0] = saf_lon[saf_lon<0]+360
saf_lat = saf_m[:,1]

#Polar Front
pf_m = fronts_m['pf'][0][0]
pf_lon = pf_m[:,0]
pf_lon[pf_lon<0] = pf_lon[pf_lon<0]+360
pf_lat = pf_m[:,1]

# Southern ACC Front
saccf_m = fronts_m['saccf'][0][0]
saccf_lon = saccf_m[:,0]
saccf_lon[saccf_lon<0] = saccf_lon[saccf_lon<0]+360
saccf_lat = saccf_m[:,1]

#Southner Boundary Front
sbdy_m = fronts_m['sbdy'][0][0]
sbdy_lon = sbdy_m[:,0]
sbdy_lon[sbdy_lon<0] = sbdy_lon[sbdy_lon<0]+360
sbdy_lat = sbdy_m[:,1]


# ---------- Findnig where Anti-Winter Eddies are compared to ACC -----------
np_eddy_ID_anti_winter_lon = eddy_ID_anti_winter['Lon [°E]'].values

lat_range = 2
lat_str = str(lat_range)

front_group_anti_winter = np.empty(len(eddy_ID_anti_winter))
#loop through your float profiles
for n in range(len(eddy_ID_anti_winter)):
    #find index of closest longitude in front position
    closest_lon_index_pf = np.nanargmin(np.absolute(pf_lon - np_eddy_ID_anti_winter_lon[n]))
    closest_lon_index_saf = np.nanargmin(np.absolute(saf_lon - np_eddy_ID_anti_winter_lon[n]))
    #print(closest_lon_index)
    #compare front latitude to float latitude
    lat_compare_pf = pf_lat[closest_lon_index_pf]
    lat_compare_saf = saf_lat[closest_lon_index_saf]
    if eddy_ID_anti_winter['Lat [°N]'].values[n] > lat_compare_saf+lat_range:
        front_group_anti_winter[n] = 1 # north of SAF
    elif (eddy_ID_anti_winter['Lat [°N]'].values[n] < lat_compare_saf+lat_range) & (eddy_ID_anti_winter['Lat [°N]'].values[n] >
                                                                                    lat_compare_pf-lat_range):
            front_group_anti_winter[n] = 2 # between saf and pf - in ACC
    elif eddy_ID_anti_winter['Lat [°N]'].values[n] < lat_compare_pf-lat_range:
        front_group_anti_winter[n] = 3 # south of PF
    else:
        front_group_anti_winter[n] = 0 



# ---------- Findnig where Anti-Spring Eddies are compared to ACC -----------
np_eddy_ID_anti_spring_lon = eddy_ID_anti_spring['Lon [°E]'].values
#lon_range = 2
#lon_str = str(lon_range)

front_group_anti_spring = np.empty(len(eddy_ID_anti_spring))
#loop through your float profiles
for n in range(len(eddy_ID_anti_spring)):
    #find index of closest longitude in front position
    closest_lon_index_pf = np.nanargmin(np.absolute(pf_lon - np_eddy_ID_anti_spring_lon[n]))
    closest_lon_index_saf = np.nanargmin(np.absolute(saf_lon - np_eddy_ID_anti_spring_lon[n]))
    #print(closest_lon_index)
    #compare front latitude to float latitude
    lat_compare_pf = pf_lat[closest_lon_index_pf]
    lat_compare_saf = saf_lat[closest_lon_index_saf]
    if eddy_ID_anti_spring['Lat [°N]'].values[n] > lat_compare_saf+lat_range:
        front_group_anti_spring[n] = 1 # north of SAF
    elif (eddy_ID_anti_spring['Lat [°N]'].values[n] < lat_compare_saf+lat_range) & (eddy_ID_anti_spring['Lat [°N]'].values[n] >
                                                                                    lat_compare_pf-lat_range):
            front_group_anti_spring[n] = 2 # between saf and pf - in ACC
    elif eddy_ID_anti_spring['Lat [°N]'].values[n] < lat_compare_pf-lat_range:
        front_group_anti_spring[n] = 3 # south of PF
    else:
        front_group_anti_spring[n] = 0 


# ---------- Findnig where Cyclonic-Winter Eddies are compared to ACC -----------
np_eddy_ID_c_winter_lon = eddy_ID_c_winter['Lon [°E]'].values
#lon_range = 2
#lon_str = str(lon_range)

front_group_c_winter = np.empty(len(eddy_ID_c_winter))
#loop through your float profiles
for n in range(len(eddy_ID_c_winter)):
    #find index of closest longitude in front position
    closest_lon_index_pf = np.nanargmin(np.absolute(pf_lon - np_eddy_ID_c_winter_lon[n]))
    closest_lon_index_saf = np.nanargmin(np.absolute(saf_lon - np_eddy_ID_c_winter_lon[n]))
    #print(closest_lon_index)
    #compare front latitude to float latitude
    lat_compare_pf = pf_lat[closest_lon_index_pf]
    lat_compare_saf = saf_lat[closest_lon_index_saf]
    if eddy_ID_c_winter['Lat [°N]'].values[n] > lat_compare_saf+lat_range:
        front_group_c_winter[n] = 1 # north of SAF
    elif (eddy_ID_c_winter['Lat [°N]'].values[n] < lat_compare_saf+lat_range) & (eddy_ID_c_winter['Lat [°N]'].values[n] >
                                                                                    lat_compare_pf-lat_range):
            front_group_c_winter[n] = 2 # between saf and pf - in ACC
    elif eddy_ID_c_winter['Lat [°N]'].values[n] < lat_compare_pf-lat_range:
        front_group_c_winter[n] = 3 # south of PF
    else:
        front_group_c_winter[n] = 0 


# ---------- Finding where Cyclonic-Spring Eddies are compared to ACC -----------
np_eddy_ID_c_spring_lon = eddy_ID_c_spring['Lon [°E]'].values
#lon_range = 2
#lon_str = str(lon_range)

front_group_c_spring = np.empty(len(eddy_ID_c_spring))
#loop through your float profiles
for n in range(len(eddy_ID_c_spring)):
    #find index of closest longitude in front position
    closest_lon_index_pf = np.nanargmin(np.absolute(pf_lon - np_eddy_ID_c_spring_lon[n]))
    closest_lon_index_saf = np.nanargmin(np.absolute(saf_lon - np_eddy_ID_c_spring_lon[n]))
    #print(closest_lon_index)
    #compare front latitude to float latitude
    lat_compare_pf = pf_lat[closest_lon_index_pf]
    lat_compare_saf = saf_lat[closest_lon_index_saf]
    if eddy_ID_c_spring['Lat [°N]'].values[n] > lat_compare_saf+lat_range:
        front_group_c_spring[n] = 1 # north of SAF
    elif (eddy_ID_c_spring['Lat [°N]'].values[n] < lat_compare_saf+lat_range) & (eddy_ID_c_spring['Lat [°N]'].values[n] >
                                                                                    lat_compare_pf-lat_range):
            front_group_c_spring[n] = 2 # between saf and pf - in ACC
    elif eddy_ID_c_spring['Lat [°N]'].values[n] < lat_compare_pf-lat_range:
        front_group_c_spring[n] = 3 # south of PF
    else:
        front_group_c_spring[n] = 0 




# ---------- Finding where Non-Eddies Winter are compared to ACC -----------
np_non_eddy_winter_lon = non_eddy_winter['Lon [°E]'].values
#lon_range = 2
#lon_str = str(lon_range)

front_group_non_winter = np.empty(len(non_eddy_winter))
#loop through your float profiles
for n in range(len(non_eddy_winter)):
    #find index of closest longitude in front position
    if np.isnan(np_non_eddy_winter_lon[n]):
        continue
    closest_lon_index_pf = np.nanargmin(np.absolute(pf_lon - np_non_eddy_winter_lon[n]))
    closest_lon_index_saf = np.nanargmin(np.absolute(saf_lon - np_non_eddy_winter_lon[n]))
    #print(closest_lon_index)
    #compare front latitude to float latitude
    lat_compare_pf = pf_lat[closest_lon_index_pf]
    lat_compare_saf = saf_lat[closest_lon_index_saf]
    if non_eddy_winter['Lat [°N]'].values[n] > lat_compare_saf+lat_range:
        front_group_non_winter[n] = 1 # north of SAF
    elif (non_eddy_winter['Lat [°N]'].values[n] < lat_compare_saf+lat_range) & (non_eddy_winter['Lat [°N]'].values[n] >
                                                                                    lat_compare_pf-lat_range):
            front_group_non_winter[n] = 2 # between saf and pf - in ACC
    elif non_eddy_winter['Lat [°N]'].values[n] < lat_compare_pf-lat_range:
        front_group_non_winter[n] = 3 # south of PF
    else:
        front_group_non_winter[n] = 0 




# ---------- Finding where Non-Eddies Spring are compared to ACC -----------
np_non_eddy_spring_lon = non_eddy_spring['Lon [°E]'].values
#lon_range = 2
#lon_str = str(lon_range)

front_group_non_spring = np.empty(len(non_eddy_spring))
#loop through your float profiles
for n in range(len(non_eddy_spring)):
    #print(np_non_eddy_spring_lon[n])
    #find index of closest longitude in front position
    if np.isnan(np_non_eddy_spring_lon[n]):
        continue
    closest_lon_index_pf = np.nanargmin(np.absolute(pf_lon - np_non_eddy_spring_lon[n]))
    closest_lon_index_saf = np.nanargmin(np.absolute(saf_lon - np_non_eddy_spring_lon[n]))
    #print(closest_lon_index)
    #compare front latitude to float latitude
    lat_compare_pf = pf_lat[closest_lon_index_pf]
    lat_compare_saf = saf_lat[closest_lon_index_saf]
    if non_eddy_spring['Lat [°N]'].values[n] > lat_compare_saf+lat_range:
        front_group_non_spring[n] = 1 # north of SAF
    elif (non_eddy_spring['Lat [°N]'].values[n] < lat_compare_saf+lat_range) & (non_eddy_spring['Lat [°N]'].values[n] >
                                                                                    lat_compare_pf-lat_range):
            front_group_non_spring[n] = 2 # between saf and pf - in ACC
    elif non_eddy_spring['Lat [°N]'].values[n] < lat_compare_pf-lat_range:
        front_group_non_spring[n] = 3 # south of PF
    else:
        front_group_non_spring[n] = 0 










############ Create a plot showing the front group numbers represented on the map - Anti winter

eddy_fg2_anti_winter = eddy_ID_anti_winter.loc[front_group_anti_winter ==2]

# Set Figure and Map Characteristics
plt.figure(figsize=(9,3))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([0,140,-70,-30], ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
gl = ax.gridlines()
gl.ylabels_left = True
gl.xlabels_bottom = True

# Plot Fronts (uses circle axes from ARGO Float Track)
#ax.set_boundary(circle, transform=ax.transAxes)
#plt.plot(stf['lon'], stf['lat'], color='Red', transform=ccrs.PlateCarree(), 
         #label='Subtropical Front')
plt.plot(saf['lon'], saf['lat'], color='Orange', transform=ccrs.PlateCarree(), 
         label='Subantarctic Front')
plt.plot(pf['lon'], pf['lat'], color='Yellow', transform=ccrs.PlateCarree(), 
         label='Polar Front')
plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree(), 
         label='Southern ACC Front')
#plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree(), 
        # label='Southern Boundary')
#plt.legend()

# Plot ARGO Float Track (taken from ARGO Float Track)
plt.scatter(eddy_fg2_anti_winter['Lon [°E]'], eddy_fg2_anti_winter['Lat [°N]'], 
            color='Black', transform=ccrs.PlateCarree(), s = 5, zorder = 1000) # change dot size
plt.title('Anticyclonic Winter Eddies between the SAF and PF at lat range ' + lat_str)
plt.savefig(output_path +'Eddy_fg2_anti_winter' + lat_str + '.png')



############ Create a plot showing the front group numbers represented on the map - Anti Spring

eddy_fg2_anti_spring = eddy_ID_anti_spring.loc[front_group_anti_spring ==2]

# Set Figure and Map Characteristics
plt.figure(figsize=(9,3))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([0,140,-70,-30], ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
gl = ax.gridlines()
gl.ylabels_left = True
gl.xlabels_bottom = True

# Plot Fronts (uses circle axes from ARGO Float Track)
#ax.set_boundary(circle, transform=ax.transAxes)
#plt.plot(stf['lon'], stf['lat'], color='Red', transform=ccrs.PlateCarree(), 
         #label='Subtropical Front')
plt.plot(saf['lon'], saf['lat'], color='Orange', transform=ccrs.PlateCarree(), 
         label='Subantarctic Front')
plt.plot(pf['lon'], pf['lat'], color='Yellow', transform=ccrs.PlateCarree(), 
         label='Polar Front')
plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree(), 
         label='Southern ACC Front')
#plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree(), 
        # label='Southern Boundary')
#plt.legend()

# Plot ARGO Float Track (taken from ARGO Float Track)
plt.scatter(eddy_fg2_anti_spring['Lon [°E]'], eddy_fg2_anti_spring['Lat [°N]'], 
            color='Black', transform=ccrs.PlateCarree(), s = 5, zorder = 1000) # change dot size
plt.title('Anticyclonic Spring Eddies between the SAF and PF at lat range ' + lat_str)
plt.savefig(output_path +'Eddy_fg2_anti_spring' + lat_str + '.png')


############ Create a plot showing the front group numbers represented on the map - Cyc winter

eddy_fg2_c_winter = eddy_ID_c_winter.loc[front_group_c_winter ==2]

# Set Figure and Map Characteristics
plt.figure(figsize=(9,3))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([0,140,-70,-30], ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
gl = ax.gridlines()
gl.ylabels_left = True
gl.xlabels_bottom = True

# Plot Fronts (uses circle axes from ARGO Float Track)
#ax.set_boundary(circle, transform=ax.transAxes)
#plt.plot(stf['lon'], stf['lat'], color='Red', transform=ccrs.PlateCarree(), 
         #label='Subtropical Front')
plt.plot(saf['lon'], saf['lat'], color='Orange', transform=ccrs.PlateCarree(), 
         label='Subantarctic Front')
plt.plot(pf['lon'], pf['lat'], color='Yellow', transform=ccrs.PlateCarree(), 
         label='Polar Front')
plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree(), 
         label='Southern ACC Front')
#plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree(), 
        # label='Southern Boundary')
#plt.legend()

# Plot ARGO Float Track (taken from ARGO Float Track)
plt.scatter(eddy_fg2_c_winter['Lon [°E]'], eddy_fg2_c_winter['Lat [°N]'], 
            color='Black', transform=ccrs.PlateCarree(), s = 5, zorder = 1000) # change dot size
plt.title('Cyclonic Winter Eddies between the SAF and PF at lat range ' + lat_str)
plt.savefig(output_path +'Eddy_fg2_c_winter' + lat_str + '.png')


############ Create a plot showing the front group numbers represented on the map - Cyc Spring

eddy_fg2_c_spring = eddy_ID_c_spring.loc[front_group_c_spring ==2]

# Set Figure and Map Characteristics
plt.figure(figsize=(9,3))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([0,140,-70,-30], ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
gl = ax.gridlines()
gl.ylabels_left = True
gl.xlabels_bottom = True
ax.set_xlim(10,140)

# Plot Fronts (uses circle axes from ARGO Float Track)
#ax.set_boundary(circle, transform=ax.transAxes)
#plt.plot(stf['lon'], stf['lat'], color='Red', transform=ccrs.PlateCarree(), 
         #label='Subtropical Front')
plt.plot(saf['lon'], saf['lat'], color='Orange', transform=ccrs.PlateCarree(), 
         label='Subantarctic Front')
plt.plot(pf['lon'], pf['lat'], color='Yellow', transform=ccrs.PlateCarree(), 
         label='Polar Front')
plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree(), 
         label='Southern ACC Front')
#plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree(), 
        # label='Southern Boundary')
#plt.legend()

# Plot ARGO Float Track (taken from ARGO Float Track)
plt.scatter(eddy_fg2_c_spring['Lon [°E]'], eddy_fg2_c_spring['Lat [°N]'], 
            color='Black', transform=ccrs.PlateCarree(), s = 5, zorder = 1000) # change dot size
plt.title('Cyclonic Spring Eddies between the SAF and PF at lat range ' + lat_str)
plt.savefig(output_path +'Eddy_fg2_c_spring' + lat_str + '.png')

# 0, 0.5, 1, 2
#print(len(eddy_fg2_anti_winter))
#188, 223, 266, 319
#print(len(eddy_fg2_anti_spring))
#144, 192, 232, 284
#print(len(eddy_fg2_c_winter))
#123, 172, 213, 279
#print(len(eddy_fg2_c_spring))
#160, 192, 211, 294




# ----------------  Organizing the data along the same pressure and average (Anti-Winter) -----------------------

#Import full SOCCOM data to use
SOCCOM = pd.read_pickle('SOCCOM_snapshot.pkl')


#Create New Pressue Array
pnew = np.arange(25,2000,1) #this will make an array with pressures every 1 db from 0 to 2000 db, you could change the increment to e.g. 10

#All Profiles

# Create New Variable Arrays
tnew_aw = np.empty((len(eddy_ID_anti_winter),len(pnew)))
snew_aw = np.empty((len(eddy_ID_anti_winter),len(pnew)))
onew_aw = np.empty((len(eddy_ID_anti_winter),len(pnew)))
nnew_aw = np.empty((len(eddy_ID_anti_winter),len(pnew)))
dnew_aw = np.empty((len(eddy_ID_anti_winter),len(pnew)))

for n in range(len(eddy_ID_anti_winter)): #pandas values
    cruise_n = eddy_ID_anti_winter['Cruise'].values[n]
    station_n = eddy_ID_anti_winter['Station'].values[n]
    SOCCOM_prof = SOCCOM.loc[(SOCCOM['Cruise']==cruise_n) & (SOCCOM['Station']==station_n)]
    #print(np.nanmin(SOCCOM_prof['Pressure[dbar]']))
    if np.all(np.isnan(SOCCOM_prof['Pressure[dbar]'])):
        continue
    if np.nanmin(SOCCOM_prof['Pressure[dbar]'])>= pnew[0]:
        continue
    
    ftemp = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Temperature[°C]']) #, bounds_error = False, fill_value = 0)
    tnew_aw[n,:] = ftemp(pnew)
    
    fsal = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Salinity[pss]']) #, bounds_error = False, fill_value = 0)
    snew_aw[n,:] = fsal(pnew)
    
    fox = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Oxygen[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    onew_aw[n,:] = fox(pnew)
    
    fnit = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Nitrate[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    nnew_aw[n,:] = fnit(pnew)

    fdic = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['DIC_LIAR[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    dnew_aw[n,:] = fdic(pnew)


# Take Average of Profiles
t_avg_aw = np.empty(len(pnew))
t_avg_aw = np.nanmean(tnew_aw, axis = 0)

s_avg_aw = np.empty(len(pnew))
s_avg_aw = np.nanmean(snew_aw, axis = 0)

o_avg_aw = np.empty(len(pnew))
o_avg_aw = np.nanmean(onew_aw, axis = 0)

n_avg_aw = np.empty(len(pnew))
n_avg_aw = np.nanmean(nnew_aw, axis = 0)

d_avg_aw = np.empty(len(pnew))
d_avg_aw = np.nanmean(dnew_aw, axis = 0)

# Take Standard Deviation Profiles
t_std_aw = np.empty(len(pnew))
t_std_aw = np.nanstd(tnew_aw, axis =0)

s_std_aw = np.empty(len(pnew))
s_std_aw = np.nanstd(snew_aw, axis =0)

o_std_aw = np.empty(len(pnew))
o_std_aw = np.nanstd(onew_aw, axis =0)

n_std_aw = np.empty(len(pnew))
n_std_aw = np.nanstd(nnew_aw, axis =0)

d_std_aw = np.empty(len(pnew))
d_std_aw = np.nanstd(dnew_aw, axis =0)


# Plot profiles


## Set min and max values for profiles
#temp_xmin = -2
#temp_xmax = 20

#sal_xmin =
#sal_xmax =

#o_xmin = 
#o_xmax =

#n_xmin =
#n_xmax =

#d_xmin =
#d_xmax =

# Temperature
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(tnew_aw)):
    sc=ax.scatter(tnew_aw[t],pnew, s=0.25)
ax.plot(t_avg_aw, pnew, c='black', linewidth = 2, label = 'Average Temperature Profile')
ax.plot(t_avg_aw + t_std_aw, pnew, t_avg_aw - t_std_aw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Temperature[°C]')
ax.set_xlim(-2,20)
ax.set_ylabel('Pressure[db]')
ax.set_title('Temperature Profiles for Anticyclonic Winter Eddies')
plt.savefig(output_path +'All_Temp_Prof_Anti_winter.png')

# Salinity
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(snew_aw)):
    sc=ax.scatter(snew_aw[t],pnew, s=0.25)
ax.plot(s_avg_aw, pnew, c='black', linewidth = 2, label = 'Average Salinity Profile')
ax.plot(s_avg_aw + s_std_aw, pnew, s_avg_aw - s_std_aw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Salinity[pss]') 
ax.set_ylabel('Pressure[db]')
ax.set_title('Salinity Profiles for Anticyclonic Winter Eddies')
plt.savefig(output_path +'All_Sal_Prof_Anti_winter.png')

# Oxygen
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(onew_aw)):
    sc=ax.scatter(onew_aw[t],pnew, s=0.25)
ax.plot(o_avg_aw, pnew, c='black', linewidth = 2, label = 'Average Oxygen Profile')
ax.plot(o_avg_aw + o_std_aw, pnew, o_avg_aw - o_std_aw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Oxygen[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Oxygen Profiles for Anticyclonic Winter Eddies')
plt.savefig(output_path +'All_Oxy_Prof_Anti_winter.png')

# Nitrate
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(nnew_aw)):
    sc=ax.scatter(nnew_aw[t],pnew, s=0.25)
ax.plot(n_avg_aw, pnew, c='black', linewidth = 2, label = 'Average Nitrate Profile')
ax.plot(n_avg_aw + n_std_aw, pnew, n_avg_aw - n_std_aw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Nitrate[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Nitrate Profiles for Anticyclonic Winter Eddies')
plt.savefig(output_path +'All_Nit_Prof_Anti_winter.png')

# DIC
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(dnew_aw)):
    sc=ax.scatter(dnew_aw[t],pnew, s=0.25)
ax.plot(d_avg_aw, pnew, c='black', linewidth = 2, label = 'Average DIC Profile')
ax.plot(d_avg_aw + d_std_aw, pnew, d_avg_aw - d_std_aw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('DIC_LIAR[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('DIC Profiles for Anticyclonic Winter Eddies')
plt.savefig(output_path +'All_DIC_Prof_Anti_winter.png')




# ------ Front Group Arrays
front_group = 1
front_group_str = str(front_group)

# Create New Variable Arrays
tnew_aw = np.empty((len(eddy_ID_anti_winter.loc[front_group_anti_winter == front_group]),len(pnew)))
snew_aw = np.empty((len(eddy_ID_anti_winter.loc[front_group_anti_winter == front_group]),len(pnew)))
onew_aw = np.empty((len(eddy_ID_anti_winter.loc[front_group_anti_winter == front_group]),len(pnew)))
nnew_aw = np.empty((len(eddy_ID_anti_winter.loc[front_group_anti_winter == front_group]),len(pnew)))
dnew_aw = np.empty((len(eddy_ID_anti_winter.loc[front_group_anti_winter == front_group]),len(pnew)))

for n in range(len(eddy_ID_anti_winter.loc[front_group_anti_winter == front_group])): #pandas values
    cruise_n = eddy_ID_anti_winter['Cruise'].loc[front_group_anti_winter == front_group].values[n]
    station_n = eddy_ID_anti_winter['Station'].loc[front_group_anti_winter == front_group].values[n]
    SOCCOM_prof = SOCCOM.loc[(SOCCOM['Cruise']==cruise_n) & (SOCCOM['Station']==station_n)]
    #print(np.nanmin(SOCCOM_prof['Pressure[dbar]']))
    if np.all(np.isnan(SOCCOM_prof['Pressure[dbar]'])):
        continue
    if np.nanmin(SOCCOM_prof['Pressure[dbar]'])>= pnew[0]:
        continue
    
    ftemp = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Temperature[°C]']) #, bounds_error = False, fill_value = 0)
    tnew_aw[n,:] = ftemp(pnew)
    
    fsal = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Salinity[pss]']) #, bounds_error = False, fill_value = 0)
    snew_aw[n,:] = fsal(pnew)
    
    fox = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Oxygen[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    onew_aw[n,:] = fox(pnew)
    
    fnit = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Nitrate[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    nnew_aw[n,:] = fnit(pnew)

    fdic = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['DIC_LIAR[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    dnew_aw[n,:] = fdic(pnew)


# Take Average of Profiles
t_avg_aw = np.empty(len(pnew))
t_avg_aw = np.nanmean(tnew_aw, axis = 0)

s_avg_aw = np.empty(len(pnew))
s_avg_aw = np.nanmean(snew_aw, axis = 0)

o_avg_aw = np.empty(len(pnew))
o_avg_aw = np.nanmean(onew_aw, axis = 0)

n_avg_aw = np.empty(len(pnew))
n_avg_aw = np.nanmean(nnew_aw, axis = 0)

d_avg_aw = np.empty(len(pnew))
d_avg_aw = np.nanmean(dnew_aw, axis = 0)

# Take Standard Deviation Profiles
t_std_aw = np.empty(len(pnew))
t_std_aw = np.nanstd(tnew_aw, axis =0)

s_std_aw = np.empty(len(pnew))
s_std_aw = np.nanstd(snew_aw, axis =0)

o_std_aw = np.empty(len(pnew))
o_std_aw = np.nanstd(onew_aw, axis =0)

n_std_aw = np.empty(len(pnew))
n_std_aw = np.nanstd(nnew_aw, axis =0)

d_std_aw = np.empty(len(pnew))
d_std_aw = np.nanstd(dnew_aw, axis =0)


# Plot profiles

# Temperature
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(tnew_aw)):
    sc=ax.scatter(tnew_aw[t],pnew, s=0.25)
ax.plot(t_avg_aw, pnew, c='black', linewidth = 2, label = 'Average Temperature Profile')
ax.plot(t_avg_aw + t_std_aw, pnew, t_avg_aw - t_std_aw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Temperature[°C]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Temperature Profiles for Front Group ' + front_group_str +' for Anticyclonic Winter Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Temp_Prof_Anti_winter.png')

# Salinity
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(snew_aw)):
    sc=ax.scatter(snew_aw[t],pnew, s=0.25)
ax.plot(s_avg_aw, pnew, c='black', linewidth = 2, label = 'Average Temperature Profile')
ax.plot(s_avg_aw + s_std_aw, pnew, s_avg_aw - s_std_aw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Salinity[pss]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Salinity Profiles for Front Group ' + front_group_str +' for Anticyclonic Winter Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Sal_Prof_Anti_winter.png')

# Oxygen
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(onew_aw)):
    sc=ax.scatter(onew_aw[t],pnew, s=0.25)
ax.plot(o_avg_aw, pnew, c='black', linewidth = 2, label = 'Average Oxygen Profile')
ax.plot(o_avg_aw + o_std_aw, pnew, o_avg_aw - o_std_aw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Oxygen[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Oxygen Profiles for Front Group ' + front_group_str +' for Anticyclonic Winter Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Oxy_Prof_Anti_winter.png')

# Nitrate
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(nnew_aw)):
    sc=ax.scatter(nnew_aw[t],pnew, s=0.25)
ax.plot(n_avg_aw, pnew, c='black', linewidth = 2, label = 'Average Nitrate Profile')
ax.plot(n_avg_aw + n_std_aw, pnew, n_avg_aw - n_std_aw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Nitrate[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Nitrate Profiles for Front Group ' + front_group_str +' for Anticyclonic Winter Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Nit_Prof_Anti_winter.png')

# DIC
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(dnew_aw)):
    sc=ax.scatter(dnew_aw[t],pnew, s=0.25)
ax.plot(d_avg_aw, pnew, c='black', linewidth = 2, label = 'Average DIC Profile')
ax.plot(d_avg_aw + d_std_aw, pnew, d_avg_aw - d_std_aw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('DIC_LIAR[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('DIC Profiles for Front Group ' + front_group_str +' for Anticyclonic Winter Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_DIC_Prof_Anti_winter.png')






# ----------------  Organizing the data along the same pressure and average (Anti-Spring) -----------------------
#All Profiles

# Create New Variable Arrays
tnew_as = np.empty((len(eddy_ID_anti_spring),len(pnew)))
snew_as = np.empty((len(eddy_ID_anti_spring),len(pnew)))
onew_as = np.empty((len(eddy_ID_anti_spring),len(pnew)))
nnew_as = np.empty((len(eddy_ID_anti_spring),len(pnew)))
dnew_as = np.empty((len(eddy_ID_anti_spring),len(pnew)))

for n in range(len(eddy_ID_anti_spring)): #pandas values
    cruise_n = eddy_ID_anti_spring['Cruise'].values[n]
    station_n = eddy_ID_anti_spring['Station'].values[n]
    SOCCOM_prof = SOCCOM.loc[(SOCCOM['Cruise']==cruise_n) & (SOCCOM['Station']==station_n)]
    #print(np.nanmin(SOCCOM_prof['Pressure[dbar]']))
    if np.all(np.isnan(SOCCOM_prof['Pressure[dbar]'])):
        continue
    if np.nanmin(SOCCOM_prof['Pressure[dbar]'])>= pnew[0]:
        continue
    
    ftemp = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Temperature[°C]']) #, bounds_error = False, fill_value = 0)
    tnew_as[n,:] = ftemp(pnew)
    
    fsal = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Salinity[pss]']) #, bounds_error = False, fill_value = 0)
    snew_as[n,:] = fsal(pnew)
    
    fox = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Oxygen[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    onew_as[n,:] = fox(pnew)
    
    fnit = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Nitrate[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    nnew_as[n,:] = fnit(pnew)

    fdic = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['DIC_LIAR[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    dnew_as[n,:] = fdic(pnew)


# Take Average of Profiles
t_avg_as = np.empty(len(pnew))
t_avg_as = np.nanmean(tnew_as, axis = 0)

s_avg_as = np.empty(len(pnew))
s_avg_as = np.nanmean(snew_as, axis = 0)

o_avg_as = np.empty(len(pnew))
o_avg_as = np.nanmean(onew_as, axis = 0)

n_avg_as = np.empty(len(pnew))
n_avg_as = np.nanmean(nnew_as, axis = 0)

d_avg_as = np.empty(len(pnew))
d_avg_as = np.nanmean(dnew_as, axis = 0)

# Take Standard Deviation Profiles
t_std_as = np.empty(len(pnew))
t_std_as = np.nanstd(tnew_as, axis =0)

s_std_as = np.empty(len(pnew))
s_std_as = np.nanstd(snew_as, axis =0)

o_std_as = np.empty(len(pnew))
o_std_as = np.nanstd(onew_as, axis =0)

n_std_as = np.empty(len(pnew))
n_std_as = np.nanstd(nnew_as, axis =0)

d_std_as = np.empty(len(pnew))
d_std_as = np.nanstd(dnew_as, axis =0)


# Plot profiles
# Temperature
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(tnew_as)):
    sc=ax.scatter(tnew_as[t],pnew, s=0.25)
ax.plot(t_avg_as, pnew, c='black', linewidth = 2, label = 'Average Temperature Profile')
ax.plot(t_avg_as + t_std_as, pnew, t_avg_as - t_std_as, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Temperature[°C]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Temperature Profiles for Anticyclonic Spring Eddies')
plt.savefig(output_path +'All_Temp_Prof_Anti_spring.png')

# Salinity
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(snew_as)):
    sc=ax.scatter(snew_as[t],pnew, s=0.25)
ax.plot(s_avg_as, pnew, c='black', linewidth = 2, label = 'Average Salinity Profile')
ax.plot(s_avg_as + s_std_as, pnew, s_avg_as - s_std_as, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Salinity[pss]') 
ax.set_ylabel('Pressure[db]')
ax.set_title('Salinity Profiles for Anticyclonic Spring Eddies')
plt.savefig(output_path +'All_Sal_Prof_Anti_spring.png')

# Oxygen
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(onew_as)):
    sc=ax.scatter(onew_as[t],pnew, s=0.25)
ax.plot(o_avg_as, pnew, c='black', linewidth = 2, label = 'Average Oxygen Profile')
ax.plot(o_avg_as + o_std_as, pnew, o_avg_as - o_std_as, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Oxygen[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Oxygen Profiles for Anticyclonic Spring Eddies')
plt.savefig(output_path +'All_Oxy_Prof_Anti_spring.png')

# Nitrate
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(nnew_as)):
    sc=ax.scatter(nnew_as[t],pnew, s=0.25)
ax.plot(n_avg_as, pnew, c='black', linewidth = 2, label = 'Average Nitrate Profile')
ax.plot(n_avg_as + n_std_as, pnew, n_avg_as - n_std_as, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Nitrate[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Nitrate Profiles for Anticyclonic Spring Eddies')
plt.savefig(output_path +'All_Nit_Prof_Anti_spring.png')

# DIC
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(dnew_as)):
    sc=ax.scatter(dnew_as[t],pnew, s=0.25)
ax.plot(d_avg_as, pnew, c='black', linewidth = 2, label = 'Average DIC Profile')
ax.plot(d_avg_as + d_std_as, pnew, d_avg_as - d_std_as, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('DIC_LIAR[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('DIC Profiles for Anticyclonic Spring Eddies')
plt.savefig(output_path +'All_DIC_Prof_Anti_spring.png')


# ----- Front Group Arrays
#front_group = 3
#front_group_str = str(front_group)

# Create New Variable Arrays
tnew_as = np.empty((len(eddy_ID_anti_spring.loc[front_group_anti_spring == front_group]),len(pnew)))
snew_as = np.empty((len(eddy_ID_anti_spring.loc[front_group_anti_spring == front_group]),len(pnew)))
onew_as = np.empty((len(eddy_ID_anti_spring.loc[front_group_anti_spring == front_group]),len(pnew)))
nnew_as = np.empty((len(eddy_ID_anti_spring.loc[front_group_anti_spring == front_group]),len(pnew)))
dnew_as = np.empty((len(eddy_ID_anti_spring.loc[front_group_anti_spring == front_group]),len(pnew)))

for n in range(len(eddy_ID_anti_spring.loc[front_group_anti_spring == front_group])): #pandas values
    cruise_n = eddy_ID_anti_spring['Cruise'].loc[front_group_anti_spring == front_group].values[n]
    station_n = eddy_ID_anti_spring['Station'].loc[front_group_anti_spring == front_group].values[n]
    SOCCOM_prof = SOCCOM.loc[(SOCCOM['Cruise']==cruise_n) & (SOCCOM['Station']==station_n)]
    #print(np.nanmin(SOCCOM_prof['Pressure[dbar]']))
    if np.all(np.isnan(SOCCOM_prof['Pressure[dbar]'])):
        continue
    if np.nanmin(SOCCOM_prof['Pressure[dbar]'])>= pnew[0]:
        continue
    
    ftemp = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Temperature[°C]']) #, bounds_error = False, fill_value = 0)
    tnew_as[n,:] = ftemp(pnew)
    
    fsal = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Salinity[pss]']) #, bounds_error = False, fill_value = 0)
    snew_as[n,:] = fsal(pnew)
    
    fox = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Oxygen[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    onew_as[n,:] = fox(pnew)
    
    fnit = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Nitrate[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    nnew_as[n,:] = fnit(pnew)

    fdic = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['DIC_LIAR[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    dnew_as[n,:] = fdic(pnew)


# Take Average of Profiles
t_avg_as = np.empty(len(pnew))
t_avg_as = np.nanmean(tnew_as, axis = 0)

s_avg_as = np.empty(len(pnew))
s_avg_as = np.nanmean(snew_as, axis = 0)

o_avg_as = np.empty(len(pnew))
o_avg_as = np.nanmean(onew_as, axis = 0)

n_avg_as = np.empty(len(pnew))
n_avg_as = np.nanmean(nnew_as, axis = 0)

d_avg_as = np.empty(len(pnew))
d_avg_as = np.nanmean(dnew_as, axis = 0)

# Take Standard Deviation Profiles
t_std_as = np.empty(len(pnew))
t_std_as = np.nanstd(tnew_as, axis =0)

s_std_as = np.empty(len(pnew))
s_std_as = np.nanstd(snew_as, axis =0)

o_std_as = np.empty(len(pnew))
o_std_as = np.nanstd(onew_as, axis =0)

n_std_as = np.empty(len(pnew))
n_std_as = np.nanstd(nnew_as, axis =0)

d_std_as = np.empty(len(pnew))
d_std_as = np.nanstd(dnew_as, axis =0)


# Plot profiles
# Temperature
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(tnew_as)):
    sc=ax.scatter(tnew_as[t],pnew, s=0.25)
ax.plot(t_avg_as, pnew, c='black', linewidth = 2, label = 'Average Temperature Profile')
ax.plot(t_avg_aw + t_std_aw, pnew, t_avg_aw - t_std_aw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Temperature[°C]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Temperature Profiles for Front Group ' + front_group_str +' for Anticyclonic Spring Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Temp_Prof_Anti_spring.png')

# Salinity
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(snew_as)):
    sc=ax.scatter(snew_as[t],pnew, s=0.25)
ax.plot(s_avg_as, pnew, c='black', linewidth = 2, label = 'Average Salinity Profile')
ax.plot(s_avg_as + s_std_as, pnew, s_avg_as - s_std_as, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Salinity[pss]') 
ax.set_ylabel('Pressure[db]')
ax.set_title('Salinity Profiles for Front Group ' + front_group_str +' for Anticyclonic Spring Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Sal_Prof_Anti_spring.png')

# Oxygen
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(onew_as)):
    sc=ax.scatter(onew_as[t],pnew, s=0.25)
ax.plot(o_avg_as, pnew, c='black', linewidth = 2, label = 'Average Oxygen Profile')
ax.plot(o_avg_as + o_std_as, pnew, o_avg_as - o_std_as, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Oxygen[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Oxygen Profiles for Front Group ' + front_group_str +' for Anticyclonic Spring Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Oxy_Prof_Anti_spring.png')

# Nitrate
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(nnew_as)):
    sc=ax.scatter(nnew_as[t],pnew, s=0.25)
ax.plot(n_avg_as, pnew, c='black', linewidth = 2, label = 'Average Nitrate Profile')
ax.plot(n_avg_as + n_std_as, pnew, n_avg_as - n_std_as, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Nitrate[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Nitrate Profiles for Front Group ' + front_group_str +' for Anticyclonic Spring Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Nit_Prof_Anti_spring.png')

# DIC
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(dnew_as)):
    sc=ax.scatter(dnew_as[t],pnew, s=0.25)
ax.plot(d_avg_as, pnew, c='black', linewidth = 2, label = 'Average DIC Profile')
ax.plot(d_avg_as + d_std_as, pnew, d_avg_as - d_std_as, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('DIC_LIAR[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('DIC Profiles for Front Group ' + front_group_str +' for Anticyclonic Spring Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_DIC_Prof_Anti_spring.png')





# ----------------  Organizing the data along the same pressure and average (Cyc-Winter) -----------------------
#All Profiles

# Create New Variable Arrays
tnew_cw = np.empty((len(eddy_ID_c_winter),len(pnew)))
snew_cw = np.empty((len(eddy_ID_c_winter),len(pnew)))
onew_cw = np.empty((len(eddy_ID_c_winter),len(pnew)))
nnew_cw = np.empty((len(eddy_ID_c_winter),len(pnew)))
dnew_cw = np.empty((len(eddy_ID_c_winter),len(pnew)))

for n in range(len(eddy_ID_c_winter)): #pandas values
    cruise_n = eddy_ID_c_winter['Cruise'].values[n]
    station_n = eddy_ID_c_winter['Station'].values[n]
    SOCCOM_prof = SOCCOM.loc[(SOCCOM['Cruise']==cruise_n) & (SOCCOM['Station']==station_n)]
    #print(np.nanmin(SOCCOM_prof['Pressure[dbar]']))
    if np.all(np.isnan(SOCCOM_prof['Pressure[dbar]'])):
        continue
    if np.nanmin(SOCCOM_prof['Pressure[dbar]'])>= pnew[0]:
        continue
    
    ftemp = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Temperature[°C]']) #, bounds_error = False, fill_value = 0)
    tnew_cw[n,:] = ftemp(pnew)
    
    fsal = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Salinity[pss]']) #, bounds_error = False, fill_value = 0)
    snew_cw[n,:] = fsal(pnew)
    
    fox = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Oxygen[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    onew_cw[n,:] = fox(pnew)
    
    fnit = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Nitrate[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    nnew_cw[n,:] = fnit(pnew)

    fdic = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['DIC_LIAR[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    dnew_cw[n,:] = fdic(pnew)


# Take Average of Profiles
t_avg_cw = np.empty(len(pnew))
t_avg_cw = np.nanmean(tnew_cw, axis = 0)

s_avg_cw = np.empty(len(pnew))
s_avg_cw = np.nanmean(snew_cw, axis = 0)

o_avg_cw = np.empty(len(pnew))
o_avg_cw = np.nanmean(onew_cw, axis = 0)

n_avg_cw = np.empty(len(pnew))
n_avg_cw = np.nanmean(nnew_cw, axis = 0)

d_avg_cw = np.empty(len(pnew))
d_avg_cw = np.nanmean(dnew_cw, axis = 0)

# Take Standard Deviation Profiles
t_std_cw = np.empty(len(pnew))
t_std_cw = np.nanstd(tnew_cw, axis =0)

s_std_cw = np.empty(len(pnew))
s_std_cw = np.nanstd(snew_cw, axis =0)

o_std_cw = np.empty(len(pnew))
o_std_cw = np.nanstd(onew_cw, axis =0)

n_std_cw = np.empty(len(pnew))
n_std_cw = np.nanstd(nnew_cw, axis =0)

d_std_cw = np.empty(len(pnew))
d_std_cw = np.nanstd(dnew_cw, axis =0)


# Plot profiles
# Temperature
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(tnew_cw)):
    sc=ax.scatter(tnew_cw[t],pnew, s=0.25)
ax.plot(t_avg_cw, pnew, c='black', linewidth = 2, label = 'Average Temperature Profile')
ax.plot(t_avg_cw + t_std_cw, pnew, t_avg_cw - t_std_cw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Temperature[°C]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Temperature Profiles for Cyclonic Winter Eddies')
plt.savefig(output_path +'All_Temp_Prof_Cyc_winter.png')

# Salinity
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(snew_cw)):
    sc=ax.scatter(snew_cw[t],pnew, s=0.25)
ax.plot(s_avg_cw, pnew, c='black', linewidth = 2, label = 'Average Salinity Profile')
ax.plot(s_avg_cw + s_std_cw, pnew, s_avg_cw - s_std_cw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Salinity[pss]') 
ax.set_ylabel('Pressure[db]')
ax.set_title('Salinity Profiles for Cyclonic Winter Eddies')
plt.savefig(output_path +'All_Sal_Prof_Cyc_winter.png')

# Oxygen
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(onew_cw)):
    sc=ax.scatter(onew_cw[t],pnew, s=0.25)
ax.plot(o_avg_cw, pnew, c='black', linewidth = 2, label = 'Average Oxygen Profile')
ax.plot(o_avg_cw + o_std_cw, pnew, o_avg_cw - o_std_cw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Oxygen[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Oxygen Profiles for Cyclonic Winter Eddies')
plt.savefig(output_path +'All_Oxy_Prof_Cyc_winter.png')

# Nitrate
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(nnew_cw)):
    sc=ax.scatter(nnew_cw[t],pnew, s=0.25)
ax.plot(n_avg_cw, pnew, c='black', linewidth = 2, label = 'Average Nitrate Profile')
ax.plot(n_avg_cw + n_std_cw, pnew, n_avg_cw - n_std_cw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Nitrate[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Nitrate Profiles for Cyclonic Winter Eddies')
plt.savefig(output_path +'All_Nit_Prof_Cyc_winter.png')

# DIC
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(dnew_cw)):
    sc=ax.scatter(dnew_cw[t],pnew, s=0.25)
ax.plot(d_avg_cw, pnew, c='black', linewidth = 2, label = 'Average DIC Profile')
ax.plot(d_avg_cw + d_std_cw, pnew, d_avg_cw - d_std_cw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('DIC_LIAR[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('DIC Profiles for Cyclonic Winter Eddies')
plt.savefig(output_path +'All_DIC_Prof_Cyc_winter.png')


# ------ Front Group Arrays
#front_group = 3
#front_group_str = str(front_group)

# Create New Variable Arrays
tnew_cw = np.empty((len(eddy_ID_c_winter.loc[front_group_c_winter == front_group]),len(pnew)))
snew_cw = np.empty((len(eddy_ID_c_winter.loc[front_group_c_winter == front_group]),len(pnew)))
onew_cw = np.empty((len(eddy_ID_c_winter.loc[front_group_c_winter == front_group]),len(pnew)))
nnew_cw = np.empty((len(eddy_ID_c_winter.loc[front_group_c_winter == front_group]),len(pnew)))
dnew_cw = np.empty((len(eddy_ID_c_winter.loc[front_group_c_winter == front_group]),len(pnew)))

for n in range(len(eddy_ID_c_winter.loc[front_group_c_winter == front_group])): #pandas values
    cruise_n = eddy_ID_c_winter['Cruise'].loc[front_group_c_winter == front_group].values[n]
    station_n = eddy_ID_c_winter['Station'].loc[front_group_c_winter == front_group].values[n]
    SOCCOM_prof = SOCCOM.loc[(SOCCOM['Cruise']==cruise_n) & (SOCCOM['Station']==station_n)]
    #print(np.nanmin(SOCCOM_prof['Pressure[dbar]']))
    if np.all(np.isnan(SOCCOM_prof['Pressure[dbar]'])):
        continue
    if np.nanmin(SOCCOM_prof['Pressure[dbar]'])>= pnew[0]:
        continue
    
    ftemp = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Temperature[°C]']) #, bounds_error = False, fill_value = 0)
    tnew_cw[n,:] = ftemp(pnew)
    
    fsal = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Salinity[pss]']) #, bounds_error = False, fill_value = 0)
    snew_cw[n,:] = fsal(pnew)
    
    fox = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Oxygen[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    onew_cw[n,:] = fox(pnew)
    
    fnit = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Nitrate[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    nnew_cw[n,:] = fnit(pnew)

    fdic = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['DIC_LIAR[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    dnew_cw[n,:] = fdic(pnew)


# Take Average of Profiles
t_avg_cw = np.empty(len(pnew))
t_avg_cw = np.nanmean(tnew_cw, axis = 0)

s_avg_cw = np.empty(len(pnew))
s_avg_cw = np.nanmean(snew_cw, axis = 0)

o_avg_cw = np.empty(len(pnew))
o_avg_cw = np.nanmean(onew_cw, axis = 0)

n_avg_cw = np.empty(len(pnew))
n_avg_cw = np.nanmean(nnew_cw, axis = 0)

d_avg_cw = np.empty(len(pnew))
d_avg_cw = np.nanmean(dnew_cw, axis = 0)

# Take Standard Deviation Profiles
t_std_cw = np.empty(len(pnew))
t_std_cw = np.nanstd(tnew_cw, axis =0)

s_std_cw = np.empty(len(pnew))
s_std_cw = np.nanstd(snew_cw, axis =0)

o_std_cw = np.empty(len(pnew))
o_std_cw = np.nanstd(onew_cw, axis =0)

n_std_cw = np.empty(len(pnew))
n_std_cw = np.nanstd(nnew_cw, axis =0)

d_std_cw = np.empty(len(pnew))
d_std_cw = np.nanstd(dnew_cw, axis =0)


# Plot profiles
# Temperature
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(tnew_cw)):
    sc=ax.scatter(tnew_cw[t],pnew, s=0.25)
ax.plot(t_avg_cw, pnew, c='black', linewidth = 2, label = 'Average Temperature Profile')
ax.plot(t_avg_cw + t_std_cw, pnew, t_avg_cw - t_std_cw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Temperature[°C]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Temperature Profiles for Front Group ' + front_group_str +' for Cyclonic Winter Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Temp_Prof_Cyc_winter.png')

# Salinity
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(snew_cw)):
    sc=ax.scatter(snew_cw[t],pnew, s=0.25)
ax.plot(s_avg_cw, pnew, c='black', linewidth = 2, label = 'Average Salinity Profile')
ax.plot(s_avg_cw + s_std_cw, pnew, s_avg_cw - s_std_cw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Salinity[pss]') 
ax.set_ylabel('Pressure[db]')
ax.set_title('Salinity Profiles for Front Group ' + front_group_str +' for Cyclonic Winter Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Sal_Prof_Cyc_winter.png')

# Oxygen
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(onew_cw)):
    sc=ax.scatter(onew_cw[t],pnew, s=0.25)
ax.plot(o_avg_cw, pnew, c='black', linewidth = 2, label = 'Average Oxygen Profile')
ax.plot(o_avg_cw + o_std_cw, pnew, o_avg_cw - o_std_cw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Oxygen[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Oxygen Profiles for Front Group ' + front_group_str +' for Cyclonic Winter Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Oxy_Prof_Cyc_winter.png')

# Nitrate
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(nnew_cw)):
    sc=ax.scatter(nnew_cw[t],pnew, s=0.25)
ax.plot(n_avg_cw, pnew, c='black', linewidth = 2, label = 'Average Nitrate Profile')
ax.plot(n_avg_cw + n_std_cw, pnew, n_avg_cw - n_std_cw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Nitrate[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Nitrate Profiles for Front Group ' + front_group_str +' for Cyclonic Winter Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Nit_Prof_Cyc_winter.png')

# DIC
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(dnew_cw)):
    sc=ax.scatter(dnew_cw[t],pnew, s=0.25)
ax.plot(d_avg_cw, pnew, c='black', linewidth = 2, label = 'Average DIC Profile')
ax.plot(d_avg_cw + d_std_cw, pnew, d_avg_cw - d_std_cw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('DIC_LIAR[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('DIC Profiles for Front Group ' + front_group_str +' for Cyclonic Winter Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_DIC_Prof_Cyc_winter.png')




# ----------------  Organizing the data along the same pressure and average (Cyc-Spring) -----------------------
#All Profiles

# Create new pnew because x range is greater than 25 db which is the bound for pnew for other columsn
#pnew = np.arange(50,2000,1)

# Create New Variable Arrays
tnew_cs = np.empty((len(eddy_ID_c_spring),len(pnew)))
snew_cs = np.empty((len(eddy_ID_c_spring),len(pnew)))
onew_cs = np.empty((len(eddy_ID_c_spring),len(pnew)))
nnew_cs = np.empty((len(eddy_ID_c_spring),len(pnew)))
dnew_cs = np.empty((len(eddy_ID_c_spring),len(pnew)))

for n in range(len(eddy_ID_c_spring)): #pandas values
    cruise_n = eddy_ID_c_spring['Cruise'].values[n]
    station_n = eddy_ID_c_spring['Station'].values[n]
    SOCCOM_prof = SOCCOM.loc[(SOCCOM['Cruise']==cruise_n) & (SOCCOM['Station']==station_n)]
    #print(np.nanmin(SOCCOM_prof['Pressure[dbar]']))
    if np.all(np.isnan(SOCCOM_prof['Pressure[dbar]'])):
        continue
    if np.nanmin(SOCCOM_prof['Pressure[dbar]'])>= pnew[0]:
        continue
    
    ftemp = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Temperature[°C]']) #, bounds_error = False, fill_value = 0)
    tnew_cs[n,:] = ftemp(pnew)
    
    fsal = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Salinity[pss]']) #, bounds_error = False, fill_value = 0)
    snew_cs[n,:] = fsal(pnew)
    
    fox = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Oxygen[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    onew_cs[n,:] = fox(pnew)
    
    fnit = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Nitrate[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    nnew_cs[n,:] = fnit(pnew)

    fdic = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['DIC_LIAR[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    dnew_cs[n,:] = fdic(pnew)


# Take Average of Profiles
t_avg_cs = np.empty(len(pnew))
t_avg_cs = np.nanmean(tnew_cs, axis = 0)

s_avg_cs = np.empty(len(pnew))
s_avg_cs = np.nanmean(snew_cs, axis = 0)

o_avg_cs = np.empty(len(pnew))
o_avg_cs = np.nanmean(onew_cs, axis = 0)

n_avg_cs = np.empty(len(pnew))
n_avg_cs = np.nanmean(nnew_cs, axis = 0)

d_avg_cs = np.empty(len(pnew))
d_avg_cs = np.nanmean(dnew_cs, axis = 0)

# Take Standard Deviation Profiles
t_std_cs = np.empty(len(pnew))
t_std_cs = np.nanstd(tnew_cs, axis =0)

s_std_cs = np.empty(len(pnew))
s_std_cs = np.nanstd(snew_cs, axis =0)

o_std_cs = np.empty(len(pnew))
o_std_cs = np.nanstd(onew_cs, axis =0)

n_std_cs = np.empty(len(pnew))
n_std_cs = np.nanstd(nnew_cs, axis =0)

d_std_cs = np.empty(len(pnew))
d_std_cs = np.nanstd(dnew_cs, axis =0)


# Plot profiles
# Temperature
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(tnew_cs)):
    sc=ax.scatter(tnew_cs[t],pnew, s=0.25)
ax.plot(t_avg_cs, pnew, c='black', linewidth = 2, label = 'Average Temperature Profile')
ax.plot(t_avg_cs + t_std_cs, pnew, t_avg_cs - t_std_cs, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Temperature[°C]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Temperature Profiles for Cyclonic Spring Eddies')
plt.savefig(output_path +'All_Temp_Prof_Cyc_spring.png')

# Salinity
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(snew_cs)):
    sc=ax.scatter(snew_cs[t],pnew, s=0.25)
ax.plot(s_avg_cs, pnew, c='black', linewidth = 2, label = 'Average Salinity Profile')
ax.plot(s_avg_cs + s_std_cs, pnew, s_avg_cs - s_std_cs, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Salinity[pss]') 
ax.set_ylabel('Pressure[db]')
ax.set_title('Salinity Profiles for Cyclonic Spring Eddies')
plt.savefig(output_path +'All_Sal_Prof_Cyc_spring.png')

# Oxygen
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(onew_cs)):
    sc=ax.scatter(onew_cs[t],pnew, s=0.25)
ax.plot(o_avg_cs, pnew, c='black', linewidth = 2, label = 'Average Oxygen Profile')
ax.plot(o_avg_cs + o_std_cs, pnew, o_avg_cs - o_std_cs, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Oxygen[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Oxygen Profiles for Cyclonic Spring Eddies')
plt.savefig(output_path +'All_Oxy_Prof_Cyc_spring.png')

# Nitrate
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(nnew_cs)):
    sc=ax.scatter(nnew_cs[t],pnew, s=0.25)
ax.plot(n_avg_cs, pnew, c='black', linewidth = 2, label = 'Average Nitrate Profile')
ax.plot(n_avg_cs + n_std_cs, pnew, n_avg_cs - n_std_cs, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Nitrate[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Nitrate Profiles for Cyclonic Spring Eddies')
plt.savefig(output_path +'All_Nit_Prof_Cyc_spring.png')

# DIC
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(dnew_cs)):
    sc=ax.scatter(dnew_cs[t],pnew, s=0.25)
ax.plot(d_avg_cs, pnew, c='black', linewidth = 2, label = 'Average DIC Profile')
ax.plot(d_avg_cs + d_std_cs, pnew, d_avg_cs - d_std_cs, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('DIC_LIAR[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('DIC Profiles for Cyclonic Spring Eddies')
plt.savefig(output_path +'All_DIC_Prof_Cyc_spring.png')


# ----- Front Group Arrays
#front_group = 3
#front_group_str = str(front_group)

# Create New Variable Arrays
tnew_cs = np.empty((len(eddy_ID_c_spring.loc[front_group_c_spring == front_group]),len(pnew)))
snew_cs = np.empty((len(eddy_ID_c_spring.loc[front_group_c_spring == front_group]),len(pnew)))
onew_cs = np.empty((len(eddy_ID_c_spring.loc[front_group_c_spring == front_group]),len(pnew)))
nnew_cs = np.empty((len(eddy_ID_c_spring.loc[front_group_c_spring == front_group]),len(pnew)))
dnew_cs = np.empty((len(eddy_ID_c_spring.loc[front_group_c_spring == front_group]),len(pnew)))

for n in range(len(eddy_ID_c_spring.loc[front_group_c_spring == front_group])): #pandas values
    cruise_n = eddy_ID_c_spring['Cruise'].loc[front_group_c_spring == front_group].values[n]
    station_n = eddy_ID_c_spring['Station'].loc[front_group_c_spring == front_group].values[n]
    SOCCOM_prof = SOCCOM.loc[(SOCCOM['Cruise']==cruise_n) & (SOCCOM['Station']==station_n)]
    #print(np.nanmin(SOCCOM_prof['Pressure[dbar]']))
    if np.all(np.isnan(SOCCOM_prof['Pressure[dbar]'])):
        continue
    if np.nanmin(SOCCOM_prof['Pressure[dbar]'])>= pnew[0]:
        continue    
    
    ftemp = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Temperature[°C]']) #, bounds_error = False, fill_value = 0)
    tnew_cs[n,:] = ftemp(pnew)
    
    fsal = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Salinity[pss]']) #, bounds_error = False, fill_value = 0)
    snew_cs[n,:] = fsal(pnew)
    
    fox = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Oxygen[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    onew_cs[n,:] = fox(pnew)
    
    fnit = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Nitrate[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    nnew_cs[n,:] = fnit(pnew)

    fdic = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['DIC_LIAR[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    dnew_cs[n,:] = fdic(pnew)


# Take Average of Profiles
t_avg_cs = np.empty(len(pnew))
t_avg_cs = np.nanmean(tnew_cs, axis = 0)

s_avg_cs = np.empty(len(pnew))
s_avg_cs = np.nanmean(snew_cs, axis = 0)

o_avg_cs = np.empty(len(pnew))
o_avg_cs = np.nanmean(onew_cs, axis = 0)

n_avg_cs = np.empty(len(pnew))
n_avg_cs = np.nanmean(nnew_cs, axis = 0)

d_avg_cs = np.empty(len(pnew))
d_avg_cs = np.nanmean(dnew_cs, axis = 0)

# Take Standard Deviation Profiles
t_std_cs = np.empty(len(pnew))
t_std_cs = np.nanstd(tnew_cs, axis =0)

s_std_cs = np.empty(len(pnew))
s_std_cs = np.nanstd(snew_cs, axis =0)

o_std_cs = np.empty(len(pnew))
o_std_cs = np.nanstd(onew_cs, axis =0)

n_std_cs = np.empty(len(pnew))
n_std_cs = np.nanstd(nnew_cs, axis =0)

d_std_cs = np.empty(len(pnew))
d_std_cs = np.nanstd(dnew_cs, axis =0)

# Plot profiles

# Temperature
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(tnew_cs)):
    sc=ax.scatter(tnew_cs[t],pnew, s=0.25)
ax.plot(t_avg_cs, pnew, c='black', linewidth = 2, label = 'Average Temperature Profile')
ax.plot(t_avg_cs + t_std_cs, pnew, t_avg_cs - t_std_cs, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Temperature[°C]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Temperature Profiles for Front Group ' + front_group_str +' for Cyclonic Spring Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Temp_Prof_Cyc_spring.png')

# Salinity
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(snew_cs)):
    sc=ax.scatter(snew_cs[t],pnew, s=0.25)
ax.plot(s_avg_cs, pnew, c='black', linewidth = 2, label = 'Average Salinity Profile')
ax.plot(s_avg_cs + s_std_cs, pnew, s_avg_cs - s_std_cs, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Salinity[pss]') 
ax.set_ylabel('Pressure[db]')
ax.set_title('Salinity Profiles for Front Group ' + front_group_str +' for Cyclonic Spring Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Sal_Prof_Cyc_spring.png')

# Oxygen
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(onew_cs)):
    sc=ax.scatter(onew_cs[t],pnew, s=0.25)
ax.plot(o_avg_cs, pnew, c='black', linewidth = 2, label = 'Average Oxygen Profile')
ax.plot(o_avg_cs + o_std_cs, pnew, o_avg_cs - o_std_cs, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Oxygen[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Oxygen Profiles for Front Group ' + front_group_str +' for Cyclonic Spring Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Oxy_Prof_Cyc_spring.png')

# Nitrate
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(nnew_cs)):
    sc=ax.scatter(nnew_cs[t],pnew, s=0.25)
ax.plot(n_avg_cs, pnew, c='black', linewidth = 2, label = 'Average Nitrate Profile')
ax.plot(n_avg_cs + n_std_cs, pnew, n_avg_cs - n_std_cs, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Nitrate[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Nitrate Profiles for Front Group ' + front_group_str +' for Cyclonic Spring Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Nit_Prof_Cyc_spring.png')

# DIC
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(dnew_cs)):
    sc=ax.scatter(dnew_cs[t],pnew, s=0.25)
ax.plot(d_avg_cs, pnew, c='black', linewidth = 2, label = 'Average DIC Profile')
ax.plot(d_avg_cs + d_std_cs, pnew, d_avg_cs - d_std_cs, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('DIC_LIAR[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('DIC Profiles for Front Group ' + front_group_str +' for Cyclonic Spring Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_DIC_Prof_Cyc_spring.png')




# ----------------  Organizing the data along the same pressure and average (Non-Eddy Winter) ---------------------
# ---  While previous sections include all profiles together and plots, this section only
#       includes the data organizing by front groups and no plots.  This is because we will
#       be using this data to test anomaly calculations


# ----- Front Group Arrays
#front_group = 3
#front_group_str = str(front_group)

# Create New Variable Arrays
tnew_nw = np.empty((len(non_eddy_winter.loc[front_group_non_winter == front_group]),len(pnew)))
snew_nw = np.empty((len(non_eddy_winter.loc[front_group_non_winter == front_group]),len(pnew)))
onew_nw = np.empty((len(non_eddy_winter.loc[front_group_non_winter == front_group]),len(pnew)))
nnew_nw = np.empty((len(non_eddy_winter.loc[front_group_non_winter == front_group]),len(pnew)))
dnew_nw = np.empty((len(non_eddy_winter.loc[front_group_non_winter == front_group]),len(pnew)))

for n in range(len(non_eddy_winter.loc[front_group_non_winter == front_group])): #pandas values
    cruise_n = non_eddy_winter['Cruise'].loc[front_group_non_winter == front_group].values[n]
    station_n = non_eddy_winter['Station'].loc[front_group_non_winter == front_group].values[n]
    SOCCOM_prof = SOCCOM.loc[(SOCCOM['Cruise']==cruise_n) & (SOCCOM['Station']==station_n)]
    #print(np.nanmin(SOCCOM_prof['Pressure[dbar]']))
    if np.all(np.isnan(SOCCOM_prof['Pressure[dbar]'])):
        continue
    if np.nanmin(SOCCOM_prof['Pressure[dbar]'])>= pnew[0]:
        continue
    ftemp = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Temperature[°C]']) #, bounds_error = False, fill_value = 0)
    tnew_nw[n,:] = ftemp(pnew)
    
    fsal = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Salinity[pss]']) #, bounds_error = False, fill_value = 0)
    snew_nw[n,:] = fsal(pnew)
    
    fox = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Oxygen[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    onew_nw[n,:] = fox(pnew)
    
    fnit = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Nitrate[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    nnew_nw[n,:] = fnit(pnew)

    fdic = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['DIC_LIAR[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    dnew_nw[n,:] = fdic(pnew)


# Take Average of Profiles
t_avg_nw = np.empty(len(pnew))
t_avg_nw = np.nanmean(tnew_nw, axis = 0)

s_avg_nw = np.empty(len(pnew))
s_avg_nw = np.nanmean(snew_nw, axis = 0)

o_avg_nw = np.empty(len(pnew))
o_avg_nw = np.nanmean(onew_nw, axis = 0)

n_avg_nw = np.empty(len(pnew))
n_avg_nw = np.nanmean(nnew_nw, axis = 0)

d_avg_nw = np.empty(len(pnew))
d_avg_nw = np.nanmean(dnew_nw, axis = 0)

# Take Standard Deviation Profiles
#t_std_as = np.empty(len(pnew))
#t_std_as = np.nanstd(tnew_as, axis =0)

#s_std_as = np.empty(len(pnew))
#s_std_as = np.nanstd(snew_as, axis =0)

#o_std_as = np.empty(len(pnew))
#o_std_as = np.nanstd(onew_as, axis =0)

#n_std_as = np.empty(len(pnew))
#n_std_as = np.nanstd(nnew_as, axis =0)

#d_std_as = np.empty(len(pnew))
#d_std_as = np.nanstd(dnew_as, axis =0)





# ----------------  Organizing the data along the same pressure and average (Non-Eddy Spring) ---------------------
# ---  While previous sections include all profiles together and plots, this section only
#       includes the data organizing by front groups and no plots.  This is because we will
#       be using this data to test anomaly calculations


# ----- Front Group Arrays
#front_group = 3
#front_group_str = str(front_group)

# Create New Variable Arrays
tnew_ns = np.empty((len(non_eddy_spring.loc[front_group_non_spring == front_group]),len(pnew)))
snew_ns = np.empty((len(non_eddy_spring.loc[front_group_non_spring == front_group]),len(pnew)))
onew_ns = np.empty((len(non_eddy_spring.loc[front_group_non_spring == front_group]),len(pnew)))
nnew_ns = np.empty((len(non_eddy_spring.loc[front_group_non_spring == front_group]),len(pnew)))
dnew_ns = np.empty((len(non_eddy_spring.loc[front_group_non_spring == front_group]),len(pnew)))

for n in range(len(non_eddy_spring.loc[front_group_non_spring == front_group])): #pandas values
    cruise_n = non_eddy_spring['Cruise'].loc[front_group_non_spring == front_group].values[n]
    station_n = non_eddy_spring['Station'].loc[front_group_non_spring == front_group].values[n]
    SOCCOM_prof = SOCCOM.loc[(SOCCOM['Cruise']==cruise_n) & (SOCCOM['Station']==station_n)]
    #print(np.nanmin(SOCCOM_prof['Pressure[dbar]']))
    if np.all(np.isnan(SOCCOM_prof['Pressure[dbar]'])):
        continue
    if np.nanmin(SOCCOM_prof['Pressure[dbar]'])>= pnew[0]:
        continue
    
    ftemp = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Temperature[°C]']) #, bounds_error = False, fill_value = 0)
    tnew_ns[n,:] = ftemp(pnew)
    
    fsal = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Salinity[pss]']) #, bounds_error = False, fill_value = 0)
    snew_ns[n,:] = fsal(pnew)
    
    fox = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Oxygen[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    onew_ns[n,:] = fox(pnew)
    
    fnit = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Nitrate[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    nnew_ns[n,:] = fnit(pnew)

    fdic = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['DIC_LIAR[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    dnew_ns[n,:] = fdic(pnew)


# Take Average of Profiles
t_avg_ns = np.empty(len(pnew))
t_avg_ns = np.nanmean(tnew_ns, axis = 0)

s_avg_ns = np.empty(len(pnew))
s_avg_ns = np.nanmean(snew_ns, axis = 0)

o_avg_ns = np.empty(len(pnew))
o_avg_ns = np.nanmean(onew_ns, axis = 0)

n_avg_ns = np.empty(len(pnew))
n_avg_ns = np.nanmean(nnew_ns, axis = 0)

d_avg_ns = np.empty(len(pnew))
d_avg_ns = np.nanmean(dnew_ns, axis = 0)

# Take Standard Deviation Profiles
#t_std_as = np.empty(len(pnew))
#t_std_as = np.nanstd(tnew_as, axis =0)

#s_std_as = np.empty(len(pnew))
#s_std_as = np.nanstd(snew_as, axis =0)

#o_std_as = np.empty(len(pnew))
#o_std_as = np.nanstd(onew_as, axis =0)

#n_std_as = np.empty(len(pnew))
#n_std_as = np.nanstd(nnew_as, axis =0)

#d_std_as = np.empty(len(pnew))
#d_std_as = np.nanstd(dnew_as, axis =0)




# --------------------- Calculate the Anomalies between Eddies and Non Eddies ------------- #
# The mean values of the non-eddy variables will be used as a baseline for calculating anomalies
#   anticyclonic and cyclonic eddies.  Cyc/anti profiles will have the non-eddy mean subtracted
#   from the profile and will be plotted, and a standard deviation will be calculated from the 
#   created anomaly profile.

# Plots included



# ----------- Anticyclonic Winter ---------- #

# Calculate Anomalies

# Temperature (degrees C)
t_anom_aw = np.empty(len(pnew))
t_anom_aw = tnew_aw-t_avg_nw

# Salinity 
s_anom_aw = np.empty(len(pnew))
s_anom_aw = snew_aw-s_avg_nw

# Oxygen 
o_anom_aw = np.empty(len(pnew))
o_anom_aw = onew_aw-o_avg_nw

# Nitrate
n_anom_aw = np.empty(len(pnew))
n_anom_aw = nnew_aw-n_avg_nw

# DIC
d_anom_aw = np.empty(len(pnew))
d_anom_aw = dnew_aw-d_avg_nw


# Calculate Mean of Anomalies

# Temperature (degrees C)
t_anom_m_aw = np.empty(len(pnew))
t_anom_m_aw = np.nanmean(t_anom_aw, axis = 0)

# Salinity 
s_anom_m_aw = np.empty(len(pnew))
s_anom_m_aw = np.nanmean(s_anom_aw, axis = 0)

# Oxygen 
o_anom_m_aw = np.empty(len(pnew))
o_anom_m_aw = np.nanmean(o_anom_aw, axis = 0)

# Nitrate
n_anom_m_aw = np.empty(len(pnew))
n_anom_m_aw = np.nanmean(n_anom_aw, axis = 0)

# DIC
d_anom_m_aw = np.empty(len(pnew))
d_anom_m_aw = np.nanmean(d_anom_aw, axis = 0)


# Take Standard Deviation Profiles

# Temperature
t_anom_std_aw = np.empty(len(pnew))
t_anom_std_aw = np.nanstd(t_anom_aw, axis =0)

# Salinity
s_anom_std_aw = np.empty(len(pnew))
s_anom_std_aw = np.nanstd(s_anom_aw, axis =0)

# Oxygen
o_anom_std_aw = np.empty(len(pnew))
o_anom_std_aw = np.nanstd(o_anom_aw, axis =0)

# Nitrate
n_anom_std_aw = np.empty(len(pnew))
n_anom_std_aw = np.nanstd(n_anom_aw, axis =0)

# DIC
d_anom_std_aw = np.empty(len(pnew))
d_anom_std_aw = np.nanstd(d_anom_aw, axis =0)



# ------ Plot Profiles


# Set bounds for profiles (these bounds will be used for all anomaly profile plots to compare)

# Temp x bounds
txmin = -12
txmax = 12

# Sal x bounds
sxmin = -2
sxmax = 2

# Oxygen x bounds
oxmin = -225
oxmax = 150


# Nitrate x bounds
nxmin = -25
nxmax = 20

# DIC x bounds
dxmin = 0
dxmax = 0



# Temperature
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(t_anom_aw)):
    sc=ax.scatter(t_anom_aw[t],pnew, s=0.25)
ax.plot(t_anom_m_aw, pnew, c='black', linewidth = 2, label = 'Average Temperature Profile')
ax.plot(t_anom_m_aw + t_anom_std_aw, pnew, t_anom_m_aw - t_anom_std_aw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Temperature[°C]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Anomaly Temperature Profiles for Front Group ' + front_group_str +' for Anticyclonic Winter Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Anom_Temp_Prof_Anti_Winter.png')

# Salinity
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(s_anom_aw)):
    sc=ax.scatter(s_anom_aw[t],pnew, s=0.25)
ax.plot(s_anom_m_aw, pnew, c='black', linewidth = 2, label = 'Average Salinity Profile')
ax.plot(s_anom_m_aw + s_anom_std_aw, pnew, s_anom_m_aw - s_anom_std_aw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Salinity[pss]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Anomaly Salinity Profiles for Front Group ' + front_group_str +' for Anticyclonic Winter Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Anom_Sal_Prof_Anti_Winter.png')

# Oxygen
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(o_anom_aw)):
    sc=ax.scatter(o_anom_aw[t],pnew, s=0.25)
ax.plot(o_anom_m_aw, pnew, c='black', linewidth = 2, label = 'Average Oxygen Profile')
ax.plot(o_anom_m_aw + o_anom_std_aw, pnew, o_anom_m_aw - o_anom_std_aw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Oxygen[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Anomaly Oxygen Profiles for Front Group ' + front_group_str +' for Anticyclonic Winter Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Anom_Oxy_Prof_Anti_winter.png')

# Nitrate
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(n_anom_aw)):
    sc=ax.scatter(n_anom_aw[t],pnew, s=0.25)
ax.plot(n_anom_m_aw, pnew, c='black', linewidth = 2, label = 'Average Nitrate Profile')
ax.plot(n_anom_m_aw + n_anom_std_aw, pnew, n_anom_m_aw - n_anom_std_aw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Nitrate[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Anomaly Nitrate Profiles for Front Group ' + front_group_str +' for Anticyclonic Winter Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Anom_Nit_Prof_Anti_winter.png')

# DIC
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(d_anom_aw)):
    sc=ax.scatter(d_anom_aw[t],pnew, s=0.25)
ax.plot(d_anom_m_aw, pnew, c='black', linewidth = 2, label = 'Average DIC Profile')
ax.plot(d_anom_m_aw + d_anom_std_aw, pnew, d_anom_m_aw - d_anom_std_aw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('DIC_LIAR[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Anomaly DIC Profiles for Front Group ' + front_group_str +' for Anticyclonic Winter Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Anom_DIC_Prof_Anti_winter.png')




# ----------- Anticyclonic Spring ---------- #

# Calculate Anomalies

# Temperature (degrees C)
t_anom_as = np.empty(len(pnew))
t_anom_as = tnew_as-t_avg_ns

# Salinity 
s_anom_as = np.empty(len(pnew))
s_anom_as = snew_as-s_avg_ns

# Oxygen 
o_anom_as = np.empty(len(pnew))
o_anom_as = onew_as-o_avg_ns

# Nitrate
n_anom_as = np.empty(len(pnew))
n_anom_as = nnew_as-n_avg_ns

# DIC
d_anom_as = np.empty(len(pnew))
d_anom_as = dnew_as-d_avg_ns


# Calculate Mean of Anomalies

# Temperature (degrees C)
t_anom_m_as = np.empty(len(pnew))
t_anom_m_as = np.nanmean(t_anom_as, axis = 0)

# Salinity 
s_anom_m_as = np.empty(len(pnew))
s_anom_m_as = np.nanmean(s_anom_as, axis = 0)

# Oxygen 
o_anom_m_as = np.empty(len(pnew))
o_anom_m_as = np.nanmean(o_anom_as, axis = 0)

# Nitrate
n_anom_m_as = np.empty(len(pnew))
n_anom_m_as = np.nanmean(n_anom_as, axis = 0)

# DIC
d_anom_m_as = np.empty(len(pnew))
d_anom_m_as = np.nanmean(d_anom_as, axis = 0)


# Take Standard Deviation Profiles

# Temperature
t_anom_std_as = np.empty(len(pnew))
t_anom_std_as = np.nanstd(t_anom_as, axis =0)

# Salinity
s_anom_std_as = np.empty(len(pnew))
s_anom_std_as = np.nanstd(s_anom_as, axis =0)

# Oxygen
o_anom_std_as = np.empty(len(pnew))
o_anom_std_as = np.nanstd(o_anom_as, axis =0)

# Nitrate
n_anom_std_as = np.empty(len(pnew))
n_anom_std_as = np.nanstd(n_anom_as, axis =0)

# DIC
d_anom_std_as = np.empty(len(pnew))
d_anom_std_as = np.nanstd(d_anom_as, axis =0)


# ----- Plot Profiles

# Temperature
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(t_anom_as)):
    sc=ax.scatter(t_anom_as[t],pnew, s=0.25)
ax.plot(t_anom_m_as, pnew, c='black', linewidth = 2, label = 'Average Temperature Profile')
ax.plot(t_anom_m_as + t_anom_std_as, pnew, t_anom_m_as- t_anom_std_as, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Temperature[°C]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Anomaly Temperature Profiles for Front Group ' + front_group_str +' for Anticyclonic Spring Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Anom_Temp_Prof_Anti_Spring.png')

# Salinity
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(s_anom_as)):
    sc=ax.scatter(s_anom_as[t],pnew, s=0.25)
ax.plot(s_anom_m_as, pnew, c='black', linewidth = 2, label = 'Average Salinity Profile')
ax.plot(s_anom_m_as + s_anom_std_as, pnew, s_anom_m_as - s_anom_std_as, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Salinity[pss]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Anomaly Salinity Profiles for Front Group ' + front_group_str +' for Anticyclonic Spring Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Anom_Sal_Prof_Anti_Spring.png')

# Oxygen
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(o_anom_as)):
    sc=ax.scatter(o_anom_as[t],pnew, s=0.25)
ax.plot(o_anom_m_as, pnew, c='black', linewidth = 2, label = 'Average Oxygen Profile')
ax.plot(o_anom_m_as + o_anom_std_as, pnew, o_anom_m_as - o_anom_std_as, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Oxygen[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Anomaly Oxygen Profiles for Front Group ' + front_group_str +' for Anticyclonic Spring Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Anom_Oxy_Prof_Anti_spring.png')

# Nitrate
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(n_anom_as)):
    sc=ax.scatter(n_anom_as[t],pnew, s=0.25)
ax.plot(n_anom_m_as, pnew, c='black', linewidth = 2, label = 'Average Nitrate Profile')
ax.plot(n_anom_m_as + n_anom_std_as, pnew, n_anom_m_as- n_anom_std_as, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Nitrate[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Anomaly Nitrate Profiles for Front Group ' + front_group_str +' for Anticyclonic Spring Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Anom_Nit_Prof_Anti_spring.png')

# DIC
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(d_anom_as)):
    sc=ax.scatter(d_anom_as[t],pnew, s=0.25)
ax.plot(d_anom_m_as, pnew, c='black', linewidth = 2, label = 'Average DIC Profile')
ax.plot(d_anom_m_as + d_anom_std_as, pnew, d_anom_m_as - d_anom_std_as, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('DIC_LIAR[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Anomaly DIC Profiles for Front Group ' + front_group_str +' for Anticyclonic Spring Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Anom_DIC_Prof_Anti_spring.png')


# ----------- Cyclonic Winter ---------- #

# Calculate Anomalies

# Temperature (degrees C)
t_anom_cw = np.empty(len(pnew))
t_anom_cw = tnew_cw-t_avg_nw

# Salinity 
s_anom_cw = np.empty(len(pnew))
s_anom_cw = snew_cw-s_avg_nw

# Oxygen 
o_anom_cw = np.empty(len(pnew))
o_anom_cw = onew_cw-o_avg_nw

# Nitrate
n_anom_cw = np.empty(len(pnew))
n_anom_cw = nnew_cw-n_avg_nw

# DIC
d_anom_cw = np.empty(len(pnew))
d_anom_cw = dnew_cw-d_avg_nw


# Calculate Mean of Anomalies

# Temperature (degrees C)
t_anom_m_cw = np.empty(len(pnew))
t_anom_m_cw = np.nanmean(t_anom_cw, axis = 0)

# Salinity 
s_anom_m_cw = np.empty(len(pnew))
s_anom_m_cw = np.nanmean(s_anom_cw, axis = 0)

# Oxygen 
o_anom_m_cw = np.empty(len(pnew))
o_anom_m_cw = np.nanmean(o_anom_cw, axis = 0)

# Nitrate
n_anom_m_cw = np.empty(len(pnew))
n_anom_m_cw = np.nanmean(n_anom_cw, axis = 0)

# DIC
d_anom_m_cw = np.empty(len(pnew))
d_anom_m_cw = np.nanmean(d_anom_cw, axis = 0)


# Take Standard Deviation Profiles

# Temperature
t_anom_std_cw = np.empty(len(pnew))
t_anom_std_cw = np.nanstd(t_anom_cw, axis =0)

# Salinity
s_anom_std_cw = np.empty(len(pnew))
s_anom_std_cw = np.nanstd(s_anom_cw, axis =0)

# Oxygen
o_anom_std_cw = np.empty(len(pnew))
o_anom_std_cw = np.nanstd(o_anom_cw, axis =0)

# Nitrate
n_anom_std_cw = np.empty(len(pnew))
n_anom_std_cw = np.nanstd(n_anom_cw, axis =0)

# DIC
d_anom_std_cw = np.empty(len(pnew))
d_anom_std_cw = np.nanstd(d_anom_cw, axis =0)



# --- Plot Profiles

# Temperature
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(t_anom_cw)):
    sc=ax.scatter(t_anom_cw[t],pnew, s=0.25)
ax.plot(t_anom_m_cw, pnew, c='black', linewidth = 2, label = 'Average Temperature Profile')
ax.plot(t_anom_m_cw + t_anom_std_cw, pnew, t_anom_m_cw - t_anom_std_cw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Temperature[°C]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Anomaly Temperature Profiles for Front Group ' + front_group_str +' for Cyclonic Winter Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Anom_Temp_Prof_Cyc_winter.png')

# Salinity
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(s_anom_cw)):
    sc=ax.scatter(s_anom_cw[t],pnew, s=0.25)
ax.plot(s_anom_m_cw, pnew, c='black', linewidth = 2, label = 'Average Salinity Profile')
ax.plot(s_anom_m_cw + s_anom_std_cw, pnew, s_anom_m_cw - s_anom_std_cw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Salinity[pss]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Anomaly Salinity Profiles for Front Group ' + front_group_str +' for Cyclonic Winter Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Anom_Sal_Prof_Cyc_winter.png')

# Oxygen
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(o_anom_cw)):
    sc=ax.scatter(o_anom_cw[t],pnew, s=0.25)
ax.plot(o_anom_m_cw, pnew, c='black', linewidth = 2, label = 'Average Oxygen Profile')
ax.plot(o_anom_m_cw + o_anom_std_cw, pnew, o_anom_m_cw - o_anom_std_cw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Oxygen[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Anomaly Oxygen Profiles for Front Group ' + front_group_str +' for Cyclonic Winter Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Anom_Oxy_Prof_Cyc_winter.png')

# Nitrate
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(n_anom_cw)):
    sc=ax.scatter(n_anom_cw[t],pnew, s=0.25)
ax.plot(n_anom_m_cw, pnew, c='black', linewidth = 2, label = 'Average Nitrate Profile')
ax.plot(n_anom_m_cw + n_anom_std_cw, pnew, n_anom_m_cw - n_anom_std_cw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Nitrate[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Anomaly Nitrate Profiles for Front Group ' + front_group_str +' for Cyclonic Winter Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Anom_Nit_Prof_Cyc_winter.png')

# DIC
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(d_anom_cw)):
    sc=ax.scatter(d_anom_cw[t],pnew, s=0.25)
ax.plot(d_anom_m_cw, pnew, c='black', linewidth = 2, label = 'Average DIC Profile')
ax.plot(d_anom_m_cw + d_anom_std_cw, pnew, d_anom_m_cw - d_anom_std_cw, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('DIC_LIAR[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Anomaly DIC Profiles for Front Group ' + front_group_str +' for Cyclonic Winter Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Anom_DIC_Prof_Cyc_winter.png')



# ----------- Cyclonic Spring ---------- #

# Calculate Anomalies

# Temperature (degrees C)
t_anom_cs = np.empty(len(pnew))
t_anom_cs = tnew_cs-t_avg_ns

# Salinity 
s_anom_cs = np.empty(len(pnew))
s_anom_cs = snew_cs-s_avg_ns

# Oxygen 
o_anom_cs = np.empty(len(pnew))
o_anom_cs = onew_cs-o_avg_ns

# Nitrate
n_anom_cs = np.empty(len(pnew))
n_anom_cs = nnew_cs-n_avg_ns

# DIC
d_anom_cs = np.empty(len(pnew))
d_anom_cs = dnew_cs-d_avg_ns


# Calculate Mean of Anomalies

# Temperature (degrees C)
t_anom_m_cs = np.empty(len(pnew))
t_anom_m_cs = np.nanmean(t_anom_cs, axis = 0)

# Salinity 
s_anom_m_cs = np.empty(len(pnew))
s_anom_m_cs = np.nanmean(s_anom_cs, axis = 0)

# Oxygen 
o_anom_m_cs = np.empty(len(pnew))
o_anom_m_cs = np.nanmean(o_anom_cs, axis = 0)

# Nitrate
n_anom_m_cs = np.empty(len(pnew))
n_anom_m_cs = np.nanmean(n_anom_cs, axis = 0)

# DIC
d_anom_m_cs = np.empty(len(pnew))
d_anom_m_cs = np.nanmean(d_anom_cs, axis = 0)


# Take Standard Deviation Profiles

# Temperature
t_anom_std_cs = np.empty(len(pnew))
t_anom_std_cs = np.nanstd(t_anom_cs, axis =0)

# Salinity
s_anom_std_cs = np.empty(len(pnew))
s_anom_std_cs = np.nanstd(s_anom_cs, axis =0)

# Oxygen
o_anom_std_cs = np.empty(len(pnew))
o_anom_std_cs = np.nanstd(o_anom_cs, axis =0)

# Nitrate
n_anom_std_cs = np.empty(len(pnew))
n_anom_std_cs = np.nanstd(n_anom_cs, axis =0)

# DIC
d_anom_std_cs = np.empty(len(pnew))
d_anom_std_cs = np.nanstd(d_anom_cs, axis =0)



# --- Plot Profiles

# Temperature
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(t_anom_cs)):
    sc=ax.scatter(t_anom_cs[t],pnew, s=0.25)
ax.plot(t_anom_m_cs, pnew, c='black', linewidth = 2, label = 'Average Temperature Profile')
ax.plot(t_anom_m_cs + t_anom_std_cs, pnew, t_anom_m_cs - t_anom_std_cs, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Temperature[°C]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Anomaly Temperature Profiles for Front Group ' + front_group_str +' for Cyclonic Spring Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Anom_Temp_Prof_Cyc_spring.png')

# Salinity
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(s_anom_cs)):
    sc=ax.scatter(s_anom_cs[t],pnew, s=0.25)
ax.plot(s_anom_m_cs, pnew, c='black', linewidth = 2, label = 'Average Salinity Profile')
ax.plot(s_anom_m_cs + s_anom_std_cs, pnew, s_anom_m_cs - s_anom_std_cs, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Salinity[pss]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Anomaly Salinity Profiles for Front Group ' + front_group_str +' for Cyclonic Spring Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Anom_Sal_Prof_Cyc_Wspring.png')

# Oxygen
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(o_anom_cs)):
    sc=ax.scatter(o_anom_cs[t],pnew, s=0.25)
ax.plot(o_anom_m_cs, pnew, c='black', linewidth = 2, label = 'Average Oxygen Profile')
ax.plot(o_anom_m_cs + o_anom_std_cs, pnew, o_anom_m_cs - o_anom_std_cs, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Oxygen[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Anomaly Oxygen Profiles for Front Group ' + front_group_str +' for Cyclonic Spring Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Anom_Oxy_Prof_Cyc_spring.png')

# Nitrate
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(n_anom_cs)):
    sc=ax.scatter(n_anom_cs[t],pnew, s=0.25)
ax.plot(n_anom_m_cs, pnew, c='black', linewidth = 2, label = 'Average Nitrate Profile')
ax.plot(n_anom_m_cs + n_anom_std_cs, pnew, n_anom_m_cs - n_anom_std_cs, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('Nitrate[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Anomaly Nitrate Profiles for Front Group ' + front_group_str +' for Cyclonic Spring Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Anom_Nit_Prof_Cyc_spring.png')

# DIC
fig = plt.figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
for t in range(len(d_anom_cs)):
    sc=ax.scatter(d_anom_cs[t],pnew, s=0.25)
ax.plot(d_anom_m_cs, pnew, c='black', linewidth = 2, label = 'Average DIC Profile')
ax.plot(d_anom_m_cs + d_anom_std_cs, pnew, d_anom_m_cs - d_anom_std_cs, pnew, c='blue', linewidth = 2, label = 'Standard Deviations')
plt.gca().invert_yaxis()
ax.set_xlabel('DIC_LIAR[µmol/kg]')
ax.set_ylabel('Pressure[db]')
ax.set_title('Anomaly DIC Profiles for Front Group ' + front_group_str +' for Cyclonic Spring Eddies')
plt.savefig(output_path + 'Front_Group_' + front_group_str +'_Anom_DIC_Prof_Cyc_spring.png')





