#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 13:48:25 2021

@author: nicolausf
"""

# Import operations
#import os
import re
import datetime
import shapely
#import cmocean
import cmocean.cm as cmo
#os.environ["PROJ_LIB"] = "C:\\Utilities\\Python\\Anaconda\\Library\\share";
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.path as mpath
#from numpy import linspace
#from numpy import meshgrid
import cartopy.geodesic 
import cartopy.crs as ccrs  # chose to use cartopy over basemap due to data format
import cartopy.feature as cf
import xarray as xr
#import gsw
from PIL import Image
import glob

from numpy import linspace, meshgrid
#from matplotlib.mlab import griddata

import eddy_matchup
from BGC_plot import BGC_plot

import gsw

### Hello, Welcome to my code! :) 

#Boop


#------------------- Import ARGO and Eddy Data ---------------------
#  Note:  This is not the only location in the code where data is imported
##### Import ARGO Float File #####

# Currently set up to pull data from a locally downloaded source
# In this code, all current files are being downloaded from the local desktop of 
# the laptop

# Location where you want figures to save:
output_path = 'generated/'

## Identify the file of the float you want to use:
# Link to float ID matchups - http://soccom.ucsd.edu/floats/SOCCOM_data_ref.html
    
# Pick a float
floatnum='5904693' # Operatore Webpage InstID - 9634
floatpath='SOCCOM_HiResQC_LIAR_05May2021_odvtxt/'  
floatsuffix='_HRQC.txt'


## Create function that sorts through data file defined with variables above, 
##   and replaces '/' with '#'
# The replacement is made because '/' is used elsewhere in the file, while # is not.
# For the December Updated SOCCOM_LIAR High Res txt files, UTF-8 encoding works,
#   but a different encoding value may be needed for earlier data (For example, "latin1")

# Input file
fin = open(floatpath+floatnum+floatsuffix,'rt', encoding='UTF-8')

# Output File
fout = open('ARGOFloat_9634.txt', 'wt')

# For loop to replace '//' with '#'
for line in fin:
    fout.write(line.replace('//','#'))
fin.close()
fout.close()


## Import Argo Float Data
# Data file is filled with additional info, texts, and comments, so the file is 
#   loaded into pandas pull useful data, with the loop above taking out the extra details

# Read Sorted File that was output in Previous Loop
ARGO = pd.read_csv("ARGOFloat_9634.txt", error_bad_lines=False, comment='#', 
                   delimiter='\t', na_values =-1E10)
# Makes Lon coordinates in terms of positive and negative values do to an issue
#   with the cartopy map for the Eddy Mercator Plots
ARGO['Lon [°E]'].loc[ARGO['Lon [°E]']>180] = ARGO['Lon [°E]'].loc[ARGO['Lon [°E]']>180]-360



### Import Eddy Data given from Dr. Chambers
# Info about the file above:
    # Line 1 - eddy ID, time of first observation in modified Julian Days
        # number of eddies in track (n), flag for direction of eddy (-1 = cyclonic,
        # 1 = anticyclonic)
    # Line 2: n values of longitude
    # Line 3: n values of latitude
    # Line 4: radius to max vel from center (in km)
    # Line 5: amplitude of SSH at center (m)
# EDDY size is 4 rows x 1 column
# to call data: EDDY.iloc[#_for_row,0]

EDDY = pd.read_csv("SO_eddy_63878.txt",error_bad_lines=False, header=None, 
                   skiprows=1, delimiter='\s+').T
EDDY.columns = ('Lon', 'Lat', 'Radius_Max_Velocity_From_Center[km]', 'SSH_Amp[m]')
# Makes Lon coordinates in terms of positive and negative values do to an issue
#   with the cartopy map for the Eddy Mercator Plots
EDDY['Lon'].loc[EDDY['Lon']>180] = EDDY['Lon'].loc[EDDY['Lon']>180]-360



### Saildrone Physical Data
ds = xr.open_dataset('data/saildrone-gen_5-antarctica_circumnavigation_2019-sd1020-20190119T040000-20190803T043000-1_minutes-v1.1620360815446.nc')
Saildrone_phys = ds.to_dataframe()




#ARGO.loc[ARGO['Station'] == 120,'date']

#---------- Sorting Through Data Quality Flags ---------------------


##### Sort Through Quality Flagged Data with a For Loop #####

## For Loop designed to replace any bad data with NaN (determined by Quality Flags (QF)) 
# Only want to keep the "good" data which has a corresponding QF = 0
# For any value that has a QF not equal to 0, replace that specific data value with NaN
# THE EXCEPTION : NITRATE - the sensor experienced issues where moderate data (QF=1)
#    is likely still useful data.


## Create a Regular Expression to make sure For Loop isn't reading anything but useful 
##  data

# Regular Expression
noDataRE = '^("|#)'


## For Loop to read through QF values and replace appropriate values
for line in ARGO:
    if re.search(noDataRE,line): #Insures nothing but data is read into loop
        continue
    for var in ARGO:
        coord = None
        var= 'pCO2_LIAR[µatm]' 
        ARGO[var] = np.where(ARGO.iloc[:,ARGO.columns.get_loc(var)+1] == 0,
                             ARGO[var], np.nan) #*************
        # rge get_loc(var)+1 takes you to the Quality Flag coulmn
        # if QF = 0, the variable value remains what it was when it was read
        # if QF does not equal 0, the variable value is replaced with NaN
        var2= 'Depth[m]'
        ARGO[var2] = np.where(ARGO.iloc[:,ARGO.columns.get_loc(var2)+1] == 0,
                              ARGO[var2], np.nan)
        var3= 'Sigma_theta[kg/m^3]'
        ARGO['Sigma_theta[kg/m^3]'] = np.where(ARGO.iloc[:,ARGO.columns.get_loc(var3)+1] == 0,
                              ARGO[var3], np.nan)
        #changed name so it was easier to understand on posters
        var4= 'DIC_LIAR[µmol/kg]'
        ARGO[var4] = np.where(ARGO.iloc[:,ARGO.columns.get_loc(var4)+1] == 0,
                              ARGO[var4], np.nan)
        var5= 'Nitrate[µmol/kg]' # may need to adjust quality flags due to sensor issue
        ARGO[var5] = np.where(ARGO.iloc[:,ARGO.columns.get_loc(var5)+1] <= 8,
                              ARGO[var5], np.nan) 
        ARGO[var5] = np.where(ARGO.iloc[:,ARGO.columns.get_loc(var5)] >= 0, #try to get rid of any negative values
                              ARGO[var5], np.nan)
        ARGO[var5] = np.where(ARGO.iloc[:,ARGO.columns.get_loc('Station')] <= 120, #try to get rid of any negative values
                              ARGO[var5], np.nan)
        var6= 'Oxygen[µmol/kg]'
        ARGO[var6] = np.where(ARGO.iloc[:,ARGO.columns.get_loc(var6)+1] == 0,
                              ARGO[var6], np.nan)
        var7= 'Temperature[°C]'
        ARGO[var7] = np.where(ARGO.iloc[:,ARGO.columns.get_loc(var7)+1] == 0,
                              ARGO[var7], np.nan)
        var8= 'Salinity[pss]'
        ARGO[var8] = np.where(ARGO.iloc[:,ARGO.columns.get_loc(var8)+1] == 0,
                              ARGO[var8], np.nan)
        var9= 'Chl_a_corr[mg/m^3]'
        ARGO[var9] = np.where(ARGO.iloc[:,ARGO.columns.get_loc(var9)+1] == 0,
                              ARGO[var9], np.nan)
        var10= 'OxygenSat[%]'
        ARGO[var10] = np.where(ARGO.iloc[:,ARGO.columns.get_loc(var10)+1] == 0,
                               ARGO[var10], np.nan)
        var11= 'Oxygen[µmol/kg]'
        ARGO[var11] = np.where(ARGO.iloc[:,ARGO.columns.get_loc(var11)+1] == 0,
                               ARGO[var11], np.nan)
        var12= 'Pressure[dbar]'
        ARGO[var11] = np.where(ARGO.iloc[:,ARGO.columns.get_loc(var12)+1] == 0,
                               ARGO[var11], np.nan)



# Skeleton format to add new variables:
    
#var= ''
#ARGO[var] = np.where(ARGO.iloc[:,ARGO.columns.get_loc(var)+1] == 0,ARGO[var], np.nan)





###### Make Additional ARGO Variables #####

# Combine Month Day Year and Hour and Minute to get full Date Time of ARGO Float Data
ARGO['date'] = pd.to_datetime(ARGO['mon/day/yr'] +' '+ ARGO['hh:mm'])

# Convert Oxygen[µmol/kg] to Oxygen[ml/L] (to match Orsi et al. 1995 for front analysis)
# https://ocean.ices.dk/tools/UnitConversion.aspx
# https://www.nodc.noaa.gov/OC5/WOD/wod18-notes.html
#   1 ml/l of O2 is approximately 43.570 µmol/kg (assumes a molar volume of O2 of 22.392 l/mole 
#       and a constant seawater potential density of 1025 kg/m3).
#   The conversion of units on a per-volume (e.g., per liter) to a per-mass (e.g., per kilogram) 
#       basis assumes a constant seawater potential density of 1025 kg/m3.
ARGO['Oxygen[mL/L]'] = ARGO['Oxygen[µmol/kg]'] / 43.570 

# Argo Oxygen mmol/L created to match plots by Talley et al. 2019
ARGO['Oxygen[µmol/L]'] = ARGO['Oxygen[µmol/kg]']*ARGO['Sigma_theta[kg/m^3]']

ARGO['row']=ARGO.index

# Log CHL 
ARGO['logCHL'] = np.log(ARGO['Chl_a_corr[mg/m^3]'])
#https://matplotlib.org/stable/gallery/images_contours_and_fields/contourf_log.html



###### List of Variables #####

# Only typed so they can be copied and pasted throughout the rest of the code

# ARGO['Lon [°E]']
# ARGO['Lat [°N]']
# ARGO['pCO2_LIAR[µatm]']
# ARGO['Depth[m]']
# ARGO['Sigma_theta[kg/m^3]']
# Density = sigma theta + 1000 ******* (kg/m^3)
# ARGO['DIC_LIAR[µmol/kg]']
# ARGO['Nitrate[µmol/kg]']
# ARGO['Oxygen[µmol/kg]']
# ARGO['Temperature[°C]']
# ARGO['Salinity[pss]']
# ARGO['Chl_a_corr[mg/m^3]']
# ARGO['OxygenSat[%]']
# ARGO['Oxygen[µmol/kg]']
# ARGO['Oxygen[mL/L]']          # Created in code, not in data
# ARGO['date']                  # Created in code, not in data








#------------ Exploring Map of ARGO and Eddy ---------------------

##### Plotting Argo Float Track with Cartopy #####
# (1 Figure)

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

# Set Circle Axes Coordinates to Boundary
ax.set_boundary(circle, transform=ax.transAxes)

# Plot Argo Float Lon and Lat Data
plt.plot(ARGO['Lon [°E]'], ARGO['Lat [°N]'], transform=ccrs.PlateCarree(), marker='o', markerfacecolor='yellow', markevery=[0], zorder=2)
plt.plot(ARGO['Lon [°E]'], ARGO['Lat [°N]'], transform=ccrs.PlateCarree(), marker='o', markerfacecolor='red', markevery=[-1], zorder=1)
plt.title('Argo Float '+floatnum)
plt.savefig(output_path + floatnum + 'FloatPath' + '.png')





##### Plotting Fronts and Fronts with Argo Float Track with Cartopy #####
# (2 Figures)

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


## Plot Fronts (uses similar figure, map characteristics, and circle axes coordinates
##   as the ARGO Float Track)

# Set Figure and Map Characteristics
plt.figure(figsize=(6, 6))
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([-180,180,-90,-30],ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
ax.gridlines()

# Plot Fronts (uses circle axes from ARGO Float Track)
ax.set_boundary(circle, transform=ax.transAxes)
plt.plot(stf['lon'], stf['lat'], color='Red', transform=ccrs.PlateCarree())
plt.plot(saf['lon'], saf['lat'], color='Orange', transform=ccrs.PlateCarree())
plt.plot(pf['lon'], pf['lat'], color='Yellow', transform=ccrs.PlateCarree())
plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree())
plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree())
plt.title('Fronts in the Southern Ocean')
plt.savefig(output_path + floatnum + 'SouthernOceanFronts' + '.png')


## Plot Fronts with ARGO Float Track (uses similar figure, map characteristics, 
#   and circle axes coordinates as the ARGO Float Track)

# Set Figure and Map Characteristics
plt.figure(figsize=(6, 6))
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([-180,180,-90,-30], ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
ax.gridlines()

# Plot Fronts (uses circle axes from ARGO Float Track)
ax.set_boundary(circle, transform=ax.transAxes)
plt.plot(stf['lon'], stf['lat'], color='Red', transform=ccrs.PlateCarree())
plt.plot(saf['lon'], saf['lat'], color='Orange', transform=ccrs.PlateCarree())
plt.plot(pf['lon'], pf['lat'], color='Yellow', transform=ccrs.PlateCarree())
plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree())
plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree())

# Plot ARGO Float Track (taken from ARGO Float Track)
plt.plot(ARGO['Lon [°E]'], ARGO['Lat [°N]'], color='Black', transform=ccrs.PlateCarree())
plt.title('Float '+floatnum+' Track and Fronts in the Southern Ocean')
plt.savefig(output_path + floatnum + 'TrackFronts' + '.png')


## Plot Fronts with ARGO Float Track and emphasized profile (uses similar figure,
#    map characteristics, and circle axes coordinates as the ARGO Float Track)   

## Set Profile Range 
# Want to range data between profiles 90 and 110 for Float 6934
#   (float encountered eddy at 92 and left at 107)
start_nprof = 90  
start_dateprof = ARGO['date'].loc[(ARGO['Station']==start_nprof)].mean()

start_Lonprof = ARGO['Lon [°E]'].loc[(ARGO['Station']==start_nprof)].mean()
start_Latprof = ARGO['Lat [°N]'].loc[(ARGO['Station']==start_nprof)].mean()

end_nprof = 110 
end_dateprof = ARGO['date'].loc[(ARGO['Station']==end_nprof)].mean()

end_Lonprof = ARGO['Lon [°E]'].loc[(ARGO['Station']==end_nprof)].mean()
end_Latprof = ARGO['Lat [°N]'].loc[(ARGO['Station']==end_nprof)].mean()

# Set Figure and Map Characteristics
plt.figure(figsize=(7, 7))
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([-180,180,-90,-30], ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
ax.gridlines()

# Plot Fronts (uses circle axes from ARGO Float Track)
ax.set_boundary(circle, transform=ax.transAxes)
plt.plot(stf['lon'], stf['lat'], color='Purple', transform=ccrs.PlateCarree(), label='Subtropical Front', linewidth = 1)
plt.plot(saf['lon'], saf['lat'], color='Red', transform=ccrs.PlateCarree(), label='Subantarctic Front', linewidth = 1)
plt.plot(pf['lon'], pf['lat'], color='Green', transform=ccrs.PlateCarree(), label='Polar Front', linewidth = 1)
#plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree())
#plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree())

# Plot ARGO Float Track (taken from ARGO Float Track)
plt.plot(ARGO['Lon [°E]'], ARGO['Lat [°N]'], color='Black', transform=ccrs.PlateCarree(), zorder=1001, label = 'Float Track', linewidth = 2)
plt.plot(ARGO['Lon [°E]'], ARGO['Lat [°N]'], color='Black', transform=ccrs.PlateCarree(), marker='o', markerfacecolor='yellow', markevery=[0], zorder=1200, label = 'Float Start Point')
plt.plot(ARGO['Lon [°E]'], ARGO['Lat [°N]'], color='Black', transform=ccrs.PlateCarree(), marker='o', markerfacecolor='red', markevery=[-1], zorder=1100, label = 'Float End Point')
# Plot profile highlighted over top of track
ax.scatter(start_Lonprof, start_Latprof, color='Cyan', s=20, transform=ccrs.PlateCarree(), zorder=1300, label = 'Eddy Encounter') #maybe make marker
ax.scatter(end_Lonprof, end_Latprof, color='Cyan', s=20, transform=ccrs.PlateCarree(), zorder=1400)
# Plot Saildrone Path
#plt.scatter(Saildrone_phys.longitude, Saildrone_phys.latitude,
#           transform=ccrs.PlateCarree(), c='green', s=2, label='Saildrone', zorder=1000)
plt.legend(loc='lower right')
#plt.title('Float '+floatnum+' Track, Saildrone Track and Fronts')
plt.title('Float ' + floatnum +' Track and Fronts')
plt.savefig(output_path + floatnum + 'TrackFrontsSaildroneAndEddyProfile' + '.png')


##### Plot of Eddy that Float and Saildrone traveled through #####

# Set Figure and Map Characteristics
plt.figure(figsize=(6, 6))
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([-160,-190,-55,-45], ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
ax.gridlines()

# Create Circle Axes Coordinates
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

# Set Circle Axes Coordinates to Boundary
ax.set_boundary(circle, transform=ax.transAxes)

# Plot Eddy Lon and Lat Data
plt.plot(EDDY['Lon'], EDDY['Lat'], transform=ccrs.PlateCarree())
plt.title('Eddy for ARGO Float '+floatnum)
plt.savefig(output_path + floatnum + 'EddyTrack' + '.png')







#------------------- Exploring Radius of Eddy ---------------------

######### MAKE A LOOP FOR PLOTTING RADIUS CIRCLES ########
#radius = cartopy.geodesic.Geodesic().circle(lon=EDDY['Lon'], lat=EDDY['Lat'], radius=rads)
#geom = shapely.geometry.Polygon(circle_points)
#ax.add_geometries((geom,), crs=cartopy.crs.PlateCarree(), facecolor='red', edgecolor='none', linewidth=0)

plt.figure(figsize=(6, 6))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_extent([-150,-144,-55,-50], ccrs.Mercator())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
ax.gridlines()
plt.plot(EDDY['Lon'], EDDY['Lat'], color='Blue', transform=ccrs.Mercator())
plt.plot(ARGO['Lon [°E]'], ARGO['Lat [°N]'], color='Black', transform=ccrs.Mercator())
plt.title('Eddy and Eddy Radius for ARGO Float '+floatnum)
geom = []
for i in range(78):
    radius = cartopy.geodesic.Geodesic().circle(lon=EDDY['Lon'][i], lat=EDDY['Lat'][i], radius=(EDDY['Radius_Max_Velocity_From_Center[km]'][i])*1000, endpoint=True)
    print(radius)
    geom = shapely.geometry.Polygon(radius) 
    ax.add_geometries([geom], crs=ccrs.Mercator(), facecolor='red', edgecolor='black', linewidth=1, alpha=1) 
plt.savefig(output_path + 'EddyTrackAndRadiusAndArgoTrack' + '.png')

### Example from Nancy
#circle_points = cartopy.geodesic.Geodesic().circle(lon=lon, lat=lat, radius=radius_in_meters, n_samples=n_points, endpoint=False)
#geom = shapely.geometry.Polygon(circle_points)
#ax.add_geometries((geom,), crs=cartopy.crs.PlateCarree(), facecolor='red', edgecolor='none', linewidth=0)


## Radius plot of eddies

plt.figure(figsize=(6, 6))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_extent([-155,-144,-55,-50], ccrs.Mercator())
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
ax.gridlines()
plt.plot(EDDY['Lon'], EDDY['Lat'], color='Red', transform=ccrs.Mercator(),zorder=1000)
plt.plot(ARGO['Lon [°E]'], ARGO['Lat [°N]'], color='Black', transform=ccrs.Mercator())
#radius = cartopy.geodesic.Geodesic().circle(lon=EDDY['Lon'], lat=EDDY['Lat'], radius=rads)
#plt.Circle(EDDY['Lon'], EDDY['Lat'], radius=radius)
#plt.plot(EDDY['Lon'], EDDY['Lat'], transform=ccrs.PlateCarree())
plt.title('Eddy for ARGO Float '+floatnum)
plt.savefig(output_path + floatnum + 'EddyAndArgoTrack' + '.png')





#----------- Exploring Random Variables in ARGO ---------------------

##### Generic Plots to Begin Sifting Through Data #####


## PCO2 vs. Depth using Scatter

# Was initially using plt.plot(), but the data format did not work well with that format,
#   instead it only plotted correctly with using plt.scatter().
plt.figure()
plt.title('PCO2 vs. Depth')
plt.xlabel('PCO2 (µatm)')
plt.ylabel('Depth (m)')
plt.scatter(ARGO['pCO2_LIAR[µatm]'], ARGO['Depth[m]'], s=1)
plt.gca().invert_yaxis()
plt.savefig(output_path + floatnum + 'pCO2vsDepth' + '.png')


## Sigma Theta vs. Depth using Scatter
# Potential plot for Gary *****
plt.figure()  
plt.title('Sigma Theta vs. Depth')
plt.xlabel('Sigma Theta (kg/m^3)')
plt.ylabel('Depth (m)')
plt.scatter(ARGO['Sigma_theta[kg/m^3]'], ARGO['Depth[m]'], s=1)
plt.gca().invert_yaxis()
plt.savefig(output_path + floatnum + 'SigmaThetaVsDepth' + '.png')


## PCO2 vs. Depth for a single specific profile indicated by nprof

# For Float 6934, we want to target profiles just before and after when the float
#   encountered an eddy.
# Those values are first encounter at 92 and last encounter at 107
# For single profile, 97 is a good profile within the eddy
nprof = 97 #Specify a profile to plot
pCO2prof = np.where(ARGO['Station']==nprof, ARGO['pCO2_LIAR[µatm]'], np.nan)
plt.figure()
plt.scatter(pCO2prof, ARGO['Depth[m]'], s=10)  
plt.xlabel('pCO2 (µatm)')
plt.ylabel('Depth (m)')
plt.title ('Profile Specific PCO2 vs. Depth')
plt.gca().invert_yaxis()
plt.savefig(output_path + floatnum + 'profilePCO2vsDepth' + '.png') #make number specific?

## PCO2 vs. Date using Scatter
plt.figure()
plt.title('Day Vs. PCO2')
plt.scatter(ARGO['date'], ARGO['pCO2_LIAR[µatm]'], s=1)
plt.savefig(output_path + floatnum + 'pCO2vsDate' + '.png')

## Sigma Theta versus Depth for Gary
nprof = 97 #Specify a profile to plot
sigmaprof = np.where(ARGO['Station']==nprof, ARGO['Sigma_theta[kg/m^3]'], np.nan)
plt.figure()
plt.scatter(sigmaprof, ARGO['Depth[m]'], s=10)  
plt.xlabel('Sigma Theta (kg/m^3)')
plt.ylabel('Depth (m)')
plt.title ('Sigma Theta  vs. Depth')
plt.gca().invert_yaxis()
plt.savefig(output_path + floatnum + 'profileSigmaThetaVsDepth' + '.png') #make number specific?


# Plots for Dr. Chambers to look at Temp Noise

##### Temperature Profiles for Dr. Chambers 6/1/21 #####
# These plots were used to confirm noise in the temperature data that caused 
# spiked in the MLD temperature gradient profiles.

## Temp profiles for Dr. Chambers
nprof = 110 # Corresponding to spike on 2/18/20
tempprof = np.where(ARGO['Station']==nprof, ARGO['Temperature[°C]'], np.nan)
plt.figure()
plt.title('Temperature vs. Depth at Station '+str(nprof))
plt.xlabel('Temperature (°C)')
plt.ylabel('Depth (m)')
plt.scatter(tempprof, ARGO['Depth[m]'], s=1)
plt.gca().invert_yaxis()
plt.savefig(output_path + floatnum + 'profileTempVsDepth1' + '.png')
#plt.savefig(output_path + 'profile' + nprof + 'TempVsDepth' + '.png')

nprof = 113 # Corresponding to spike between 2/19/2019 and 2/20/2019
tempprof = np.where(ARGO['Station']==nprof, ARGO['Temperature[°C]'], np.nan)
plt.figure()
plt.title('Temperature vs. Depth at Station '+str(nprof))
plt.xlabel('Temperature (°C)')
plt.ylabel('Depth (m)')
plt.scatter(tempprof, ARGO['Depth[m]'], s=1)
plt.gca().invert_yaxis()
plt.savefig(output_path + floatnum + 'profileTempVsDepth2' + '.png')
#plt.savefig(output_path + floatnum + 'profile' + nprof + 'TempVsDepth' + '.png')







#------------------- Exploring the MLD ---------------------

##### Mixed Layer Depth (MLD) #####

## Nancy's instruction about MLD to her class:
#It would be helpful to know where the mixed layer is when we're thinking 
#about surface ocean seasonality. Typically, mixed layer is calculated by 
#looking at the density relative to the surface. As you move down the water 
#column, the density increases and you can choose a threshold above which you 
#are no longer in the mixed layer. This is typically 0.03 kg/m3 
#(https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2004JC002378). and Holte 2017 paper ~ First
# let's look at density. Sigma theta is actually the density minus 1000 kg/m3. #
#The true density is closer to 1027 kg/m3.


## Calculate MLD for each station using density threshold
# We want to find where in a given station the density is greater than 0.03
#    greater than the surface density.  The shallowest of those depths is the
#    mixed layer depth.

# Create empty dataframe to drop values corresponding to the MLD into
MLD = []

# Use For Loop to loop through densities of each station meeting the requirements 
#   for calculating MLD as stated above.
for station in ARGO['Station'].unique():
    surfacedens = ARGO['Sigma_theta[kg/m^3]'].loc[(ARGO['Station'] == station)].min()
    MLD.append([
      station,ARGO['date'].loc[(ARGO['Station'] == station) & (ARGO['Sigma_theta[kg/m^3]']-surfacedens>0.03)].min(),
      ARGO['Depth[m]'].loc[(ARGO['Station'] == station) & (ARGO['Sigma_theta[kg/m^3]']-surfacedens>0.03)].min(),
      ARGO['Lon [°E]'].loc[(ARGO['Station'] == station)].mean(),
      ARGO['Lat [°N]'].loc[(ARGO['Station'] == station)].mean()
      ])

# Put MLD in terms of a pandas dataframe to match all other files used
MLD = pd.DataFrame(data = MLD, columns = ['Station', 'date', 'MLD','Lon [°E]','Lat [°N]'])


## Set Depth Range for Biogeochemical Plots
# Want to Utilize MLD
MLDdepth = MLD['MLD'].max() + 50  





###### Mixed Layer Depth (MLD) Focused on Sigma Theta ########
# When looking at the mixed layer, the threshold may not be the most accurate
# depiction.  The next few for loops test approaches based on 
# sigma theta, temperature, and salinity gradients as indicators of the MLD
# Excluding the top 10 m of data as per Holte et al. (2017) where diurnal 
# heating may be an issue

MLDgrad = []
diurnal = 20

for station in ARGO['Station'].unique():
    ARGO['DepSig'] = (
      ARGO['Sigma_theta[kg/m^3]'].loc[(ARGO['Station'] == station) & (
          ARGO['Depth[m]'] > diurnal)].diff()
      )/(
      ARGO['Depth[m]'].loc[(ARGO['Station']==station) & (
          ARGO['Depth[m]'] > diurnal)].diff()
      )
    MLDgrad.append([
      station,
      ARGO['date'].loc[(ARGO['Station']==station) & (
          ARGO['Depth[m]'] > diurnal) & (
          (ARGO['DepSig'])==(ARGO['DepSig'].max()))].min(),
      ARGO['Depth[m]'].loc[(ARGO['Station']==station) & (
          ARGO['Depth[m]'] > diurnal) & (
          (ARGO['DepSig'])==(ARGO['DepSig'].max()))].min(),
      ARGO['Lon [°E]'].loc[(ARGO['Station']==station) & (
          ARGO['Depth[m]'] > diurnal)].mean(),
      ARGO['Lat [°N]'].loc[(ARGO['Station']==station) & (
          ARGO['Depth[m]'] > diurnal)].mean()
      ])


# Put MLD in terms of a pandas dataframe to match all other files used
MLDgrad = pd.DataFrame(data=MLDgrad, columns=['Station', 'date', 'MLD','Lon [°E]','Lat [°N]'])

# MLD in Xarray - added just incase I want to attempt this
# MLDgradXR = MLDgrad.to_xarray() #, columns=['Station', 'date', 'MLD','Lon [°E]','Lat [°N]'])
# MLDgradXR.set_coords('date')


## Set Depth Range for Biogeochemical Plots
# Want to Utilize MLD
MLDgraddepth=MLDgrad['MLD'].max()+50  



###### Mixed Layer Depth (MLD) Focused on Temperature ########
# Excluding the top 10 m of data as per Holte et al. (2017) where diurnal 
# heating may be an issue

# Create empty dataframe
MLDtemp = []
diurnal = 20

for station in ARGO['Station'].unique():
    ARGO['TempStat'] = (
        ARGO['Temperature[°C]'].loc[(ARGO['Station']==station) & (
            ARGO['Depth[m]'] > diurnal)].diff()
        )/(
        ARGO['Depth[m]'].loc[(ARGO['Station']==station) & (
            ARGO['Depth[m]'] > diurnal)].diff())
    MLDtemp.append(
        [station, 
        ARGO['date'].loc[(ARGO['Station']==station) & (
            ARGO['Depth[m]'] > diurnal) & (
            (ARGO['TempStat'])==(ARGO['TempStat'].max()))].min(),
        ARGO['Depth[m]'].loc[(ARGO['Station']==station) & (
            ARGO['Depth[m]'] > diurnal) & (
            (ARGO['TempStat'])==(ARGO['TempStat'].max()))].min(),
        ARGO['Lon [°E]'].loc[(ARGO['Station']==station) & (
          ARGO['Depth[m]'] > diurnal)].mean(),
      ARGO['Lat [°N]'].loc[(ARGO['Station']==station) & (
          ARGO['Depth[m]'] > diurnal)].mean()
      ])


# Put MLD in terms of a pandas dataframe to match all other files used
MLDtemp = pd.DataFrame(data=MLDtemp, columns=['Station', 'date', 'MLD','Lon [°E]','Lat [°N]'])

## Set Depth Range for Biogeochemical Plots
# Want to Utilize MLD
MLDtempdepth=MLDtemp['MLD'].max()+50  





###### Mixed Layer Depth (MLD) Focused on Salinity ########
# When looking at the mixed layer
# Excluding the top 10 m of data as per Holte et al. (2017) where diurnal 
# heating may be an issue

# Create empty dataframe
MLDsal = []
diurnal = 20
#(ARGO['Sigma_theta[kg/m^3]'].diff()/ARGO['Depth[m]'].diff())

for station in ARGO['Station'].unique():
    ARGO['SalStat'] = (
        ARGO['Salinity[pss]'].loc[(ARGO['Station']==station) & (
            ARGO['Depth[m]'] > diurnal)].diff()
        )/(
        ARGO['Depth[m]'].loc[(ARGO['Station']==station) & (
            ARGO['Depth[m]'] > diurnal)].diff())
    MLDsal.append(
        [station, 
        ARGO['date'].loc[(ARGO['Station']==station) & (
            ARGO['Depth[m]'] > diurnal) & (
            (ARGO['SalStat'])==(ARGO['SalStat'].max()))].min(),
        ARGO['Depth[m]'].loc[(ARGO['Station']==station) & (
            ARGO['Depth[m]'] > diurnal) & (
            (ARGO['SalStat'])==(ARGO['SalStat'].max()))].min(),
        ARGO['Lon [°E]'].loc[(ARGO['Station']==station) & (
          ARGO['Depth[m]'] > diurnal)].mean(),
      ARGO['Lat [°N]'].loc[(ARGO['Station']==station) & (
          ARGO['Depth[m]'] > diurnal)].mean()
      ])


# Put MLD in terms of a pandas dataframe to match all other files used
MLDsal = pd.DataFrame(data=MLDsal, columns=['Station', 'date', 'MLD','Lon [°E]','Lat [°N]'])


## Set Depth Range for Biogeochemical Plots
# Want to Utilize MLD
MLDsaldepth=MLDsal['MLD'].max()+50  









#------------------- Biogeochemical Plots ---------------------
### This process was used before the creation of the BGC_plot function
## Set Profile Range- Appears earlier in code: This is a reminder ******
# Want to range data between profiles 90 and 110 for Float 6934
#   (float encountered eddy at 92 and left at 107)
# start_nprof = 90  
# start_dateprof = ARGO['date'].loc[(ARGO['Station']==start_nprof)].mean()
# end_nprof = 110 
# end_dateprof = ARGO['date'].loc[(ARGO['Station']==end_nprof)].mean()


## Plot DIC, CHL, pCO2, nitrate, temp, salinity, and density
# Some sections are commented out in order to make it easier to override colorbar
#   scales, xlims, etc.
# All plots can be adjusted by figure size, date range, profile range, etc. in order
#   to zoom in or out to focus on various features found in the plots.

# Quality flagging within plots has been commented out - is already sorted through in the for
#   loop created earlier in the code, not necessary in the plotting code.

# DIC
# https://stackoverflow.com/questions/34113083/numpy-contour-typeerror-input-z-must-be-a-2d-array
# Website above is example for turning 1D into a 2D array

var='DIC_LIAR[µmol/kg]'
fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
sc=ax.scatter(ARGO['date'],ARGO['Depth[m]'], c=ARGO[var], cmap = 'cool')
#ax.plot(MLD['date'], MLD['MLD'], c='blue', label = 'MLD Density Threshold')
ax.plot(MLDgrad['date'], MLDgrad['MLD'], c='green', label='MLD Density Gradient')
#plt.axvline(x=start)
ax.invert_yaxis()
ax.set_title(var+' for Float '+floatnum)
#ax.set_ylim([MLDdepth, 0])
ax.set_xlim([datetime.date(2019,2,1), datetime.date(2019,3,1)])
#ax.set_xlim([datetime.date(2019,2,8), datetime.date(2019,3,5)])
#ax.set_xlim([pd.to_datetime(start_dateprof), pd.to_datetime(end_dateprof)])
plt.xticks(rotation = 60)
plt.legend()
cb=plt.colorbar(sc)
cb.set_label(var)
# automatically adjusts the colorbar based on the range of values youre plotting
#sc.set_clim(vmin = ARGO[var].loc[(ARGO['Depth[m]']<MLDdepth)].min(), 
#            vmax = ARGO[var].loc[(ARGO['Depth[m]']<MLDdepth)].max()) 
#sc.set_clim(vmin = 2090, vmax = 2130)
plt.savefig(output_path + floatnum + 'bgcDIC' + '.png')


##### Example of how to get individual Station Biogeochemical plots #####
#var='Temperature[°C]'
#fig = plt.figure(num=None, figsize=(10,5), dpi=80, facecolor='w', edgecolor='k')
#ax = fig.add_axes([0.1, 0.1, .8, .8])
#nprof = 113
#prof = np.where(ARGO['Station']==nprof, ARGO[var], np.nan)
#sc=ax.scatter(ARGO['date'], ARGO['Depth[m]'], c=prof, cmap = 'bwr')
#ax.invert_yaxis()
#ax.set_title('Temperature for Float '+floatnum+' at Station '+str(nprof))
#ax.set_ylim([250, 0])
#ax.set_xlim([datetime.date(2019,2,15), datetime.date(2019,2,21)])
#ax.set_xlim([pd.to_datetime(start_dateprof), pd.to_datetime(end_dateprof)])
#plt.xticks(rotation = 60)
#cb=plt.colorbar(sc)
#cb.set_label(var)
#sc.set_clim(vmin = ARGO[var].loc[(ARGO['Depth[m]']<MLDdepth)].min(), 
#            vmax = ARGO[var].loc[(ARGO['Depth[m]']<MLDdepth)].max()) 
#sc.set_clim(vmin = 2, vmax = 15) # (was roughly 0 to 14)






#------------------- Biogeochemical Plots with BGC_plot ---------------------


#--Oxygen
var = 'Oxygen[µmol/kg]'
BGC_plot(ARGO, var, 'Station', cmap = 'Blues', eddy_station = True) #GSS POSTER
BGC_plot(ARGO, var, 'Station', cmap = 'Blues', station_max = 120,
         eddy_station = True, depth_max = 500) #GSS POSTER
BGC_plot(ARGO, var, 'Station', cmap = 'Blues', map = True)
BGC_plot(ARGO, var, 'Station', cmap = 'Blues', map = True, MLD = True)
BGC_plot(ARGO, var, 'Station', cmap = 'Blues', depth_max = 250, MLD = True)
BGC_plot(ARGO, var, 'Station', cmap = 'Blues', station_min = 90, station_max =130)
BGC_plot(ARGO, var, 'Station', cmap = 'Blues', station_min = 90, station_max =130,
         depth_max = 250, MLD = True, eddy_station = True) #GSS POSTER
#plt.savefig(output_path + floatnum + 'BGC_plot_Oxygen' + '.png')

#--Nitrate
var = 'Nitrate[µmol/kg]'
BGC_plot(ARGO, var, 'Station', cmap = 'Purples') #GSS POSTER
BGC_plot(ARGO, var, 'Station', cmap = 'Purples', station_max = 120,
         eddy_station = True, depth_max = 500) #GSS POSTER
BGC_plot(ARGO, var, 'Station', cmap = 'Purples', map = True)
BGC_plot(ARGO, var, 'Station', cmap = 'Purples', map = True, MLD = True)
BGC_plot(ARGO, var, 'Station', cmap = 'Purples', depth_max = 250, MLD = True)
BGC_plot(ARGO, var, 'Station', cmap = 'Purples', station_min = 90, station_max =130)
BGC_plot(ARGO, var, 'Station', cmap = 'Purples', station_min = 90, station_max =130,
         depth_max = 250, MLD = True, eddy_station = True) #GSS POSTER
#plt.savefig(output_path + floatnum + 'BGC_plot_Nitrate' + '.png')

#--DIC
var = 'DIC_LIAR[µmol/kg]'
BGC_plot(ARGO, var, 'Station', cmap = 'Spectral') 
BGC_plot(ARGO, var, 'Station', cmap = 'Spectral', station_max = 120,
         eddy_station = True, depth_max = 500) #GSS POSTER
BGC_plot(ARGO, var, 'Station', cmap = 'Spectral', map = True)
BGC_plot(ARGO, var, 'Station', cmap = 'Spectral', map = True, MLD = True)
BGC_plot(ARGO, var, 'Station', cmap = 'Spectral', depth_max = 250, MLD = True)
BGC_plot(ARGO, var, 'Station', cmap = 'Spectral', station_min = 90, station_max =120)
BGC_plot(ARGO, var, 'Station', cmap = 'Spectral', station_min = 90, station_max =120,
         depth_max = 250, MLD = True, eddy_station = True) #GSS POSTER
#plt.savefig(output_path + floatnum + 'BGC_plot_DIC_LIAR' + '.png')

#--Temperature
var ='Temperature[°C]'
BGC_plot(ARGO, var, 'Station', cmap = 'coolwarm', eddy_station = True) # GSS POSTER
BGC_plot(ARGO, var, 'Station', cmap = 'coolwarm', map = True)
BGC_plot(ARGO, var, 'Station', cmap = 'coolwarm', station_max = 120,
         eddy_station = True, depth_max = 500) #GSS POSTER
BGC_plot(ARGO, var, 'Station', cmap = 'coolwarm', map = True, MLD = True)
BGC_plot(ARGO, var, 'Station', cmap = 'coolwarm', depth_max = 250, MLD = True)
BGC_plot(ARGO, var, 'Station', cmap = 'coolwarm', station_min = 90, station_max =130)
BGC_plot(ARGO, var, 'Station', cmap = 'coolwarm', station_min = 90, station_max =130,
         depth_max = 250, MLD = True, eddy_station = True) #GSS POSTER
#plt.savefig(output_path + floatnum + 'BGC_plot_Temp' + '.png')

#--CHL 
var ='Chl_a_corr[mg/m^3]'
BGC_plot(ARGO, var, 'Station', cmap = 'Greens') 
BGC_plot(ARGO, var, 'Station', cmap = 'Greens', station_max = 120,
         eddy_station = True, depth_max = 500) #GSS POSTER
BGC_plot(ARGO, var, 'Station', cmap = 'Greens', station_max = 120,
         eddy_station = True, depth_max = 250) #GSS POSTER
BGC_plot(ARGO, var, 'Station', cmap = 'Greens', map = True)
BGC_plot(ARGO, var, 'Station', cmap = 'Greens', map = True, MLD = True)
BGC_plot(ARGO, var, 'Station', cmap = 'Greens', depth_max = 250, MLD = True)
BGC_plot(ARGO, var, 'Station', cmap = 'Greens', station_min = 90, station_max =130)
BGC_plot(ARGO, var, 'Station', cmap = 'Greens', station_min = 90, station_max =120,
         depth_max = 250, MLD = True, eddy_station = True) #GSS POSTER
#plt.savefig(output_path + floatnum + 'BGC_plot_CHL' + '.png')


#--Salinity
var ='Salinity[pss]'
BGC_plot(ARGO, var, 'Station', cmap = 'Oranges') 
BGC_plot(ARGO, var, 'Station', cmap = 'Oranges', station_max = 120,
         eddy_station = True, depth_max = 500) #GSS POSTER
BGC_plot(ARGO, var, 'Station', cmap = 'Oranges', map = True) #GSS POSTER
BGC_plot(ARGO, var, 'Station', cmap = 'Oranges', map = True, MLD = True)
BGC_plot(ARGO, var, 'Station', cmap = 'Oranges', depth_max = 250, MLD = True)
BGC_plot(ARGO, var, 'Station', cmap = 'Oranges', station_min = 90, station_max =130)
BGC_plot(ARGO, var, 'Station', cmap = 'Oranges', station_min = 90, station_max =130,
         depth_max = 250, MLD = True, eddy_station = True) #GSS POSTER
#plt.savefig(output_path + floatnum + 'BGC_plot_Sal' + '.png')


#--Sigma Theta
var ='Sigma_theta[kg/m^3]'
BGC_plot(ARGO, var, 'Station', cmap = cmo.dense) #GSS POSTER
BGC_plot(ARGO, var, 'Station', cmap = cmo.dense, MLD = True, legend = True,
         eddy_station = True) #GSS POSTER
BGC_plot(ARGO, var, 'Station', cmap = cmo.dense, map = True)
BGC_plot(ARGO, var, 'Station', cmap = cmo.dense, map = True, MLD = True)
BGC_plot(ARGO, var, 'Station', cmap = cmo.dense, depth_max = 250, MLD = True)
BGC_plot(ARGO, var, 'Station', cmap = cmo.dense, station_min = 90, station_max =130)
BGC_plot(ARGO, var, 'Station', cmap = cmo.dense, station_min = 90, station_max =130,
         depth_max = 250, MLD = True, eddy_station = True, legend = True) #GSS POSTER
#plt.savefig(output_path + floatnum + 'BGC_plot_SigmaTheta' + '.png')















#----------- U and V Ekman Calculations using Collocated Data ---------------------

##### Ekman Transport Calculation #####
# Calculating Ekman Transport in order to create a greater understanding of the
#   biogeochemical float profiles 
# Aiming to look for regions of upwelling or downwelling which would help explain
#   some oddities in the MLD calculations

### At this point in time, the code is working backwards to derive ekman transport
# For the first iteration, wind stress will be held constant
# Moving forward, gradualy changes will be made to complete the wind stress calculation



### Reading in CCMP wind data for Wind Stress / Ekman Transport equation ###
### This data must be downloaded prior to using this part of the code

### Identify file path and month folder you want to look at ###
CCMPmonth='M' # Format is in M##/ where the numbers are the month number
CCMPpath='CCMPdata/'  

### Open dataset using xr.open_mfdataset()
CCMPall = xr.open_mfdataset(CCMPpath+CCMPmonth+'*/*.nc')


### Import Surface Ocean data - Copernicus ###
OceanPath = 'Copernicus/'
Ocean = xr.open_mfdataset(OceanPath +'*.nc')

### Import Surface Ocean data - Oscar ###
OscPath = 'Oscar/'
Oscar = xr.open_mfdataset(OscPath +'*.nc')

##### Colocating CCMP wind data to Argo Float Surface Data #####

### Get Argo Surface Data ###
ARGO_Surface = []

# Use For Loop to loop through densities of each station meeting the requirements 
#   for calculating MLD as stated above.
for station in ARGO['Station'].unique():
    surface = ARGO['Depth[m]'].loc[(ARGO['Station'] == station)].min()
    ARGO_Surface.append([
      station,ARGO['date'].loc[(ARGO['Station'] == station) & (ARGO['Depth[m]']==surface)].min(),
      ARGO['Depth[m]'].loc[(ARGO['Station'] == station) & (ARGO['Depth[m]']==surface)].min(),
      ARGO['Lon [°E]'].loc[(ARGO['Station'] == station) & (ARGO['Depth[m]']==surface)].mean(),
      ARGO['Lat [°N]'].loc[(ARGO['Station'] == station) & (ARGO['Depth[m]']==surface)].mean()
      ])

ARGO_Surface = pd.DataFrame(data=ARGO_Surface, columns=['Station', 'date', 'Depth[m]','Lon [°E]','Lat [°N]'])
ARGO_Surface = ARGO_Surface.loc[(ARGO_Surface['Station'] >= 90) & (ARGO_Surface['Station'] <= 125)]
# Stations 97-121 are the target stations, but wanted to include a little more 
#   data from before and after those times due to averaging later

### Set Boundaries on Time, Lat and Lon ###
time_min,time_max = ARGO_Surface['date'].min(),ARGO_Surface['date'].max() # ARGO

# Lon
lon_min,lon_max = ARGO_Surface['Lon [°E]'].min()+360,ARGO_Surface['Lon [°E]'].max()+360

# Lat
lat_min,lat_max = ARGO_Surface['Lat [°N]'].min(),ARGO_Surface['Lat [°N]'].max()
#lat_min,lat_max = window['latitude'].min(), window['latitude'].max() #window



### Subset CCMP data to match time, lat and lon boundaries, and collocate
subset = CCMPall.sel(
    longitude = slice(lon_min-2,lon_max+2),
    latitude = slice(lat_min-2,lat_max+2),
    time = slice(time_min, time_max)) 
# Time Max is March 5 but your wind are only from February
# Using "linear" interpolation it will fill in a value from the wrong day.
# Need to add winds and ocean currents for March
subset.load()

# Pinpoint locations of interest within ARGO_Surface #
lat_points = xr.DataArray(ARGO_Surface['Lat [°N]'], dims = 'points')
lon_points = xr.DataArray(ARGO_Surface['Lon [°E]']+360, dims = 'points')
time_points = xr.DataArray(ARGO_Surface['date'], dims = 'points')

#
CCMPall_collocated = subset.interp(
    latitude = lat_points,
    longitude = lon_points,
    time = time_points,
    method = 'linear')



### Subset and Collocate Ocean Data ###
subsetOcean = Ocean.sel(
    longitude = slice(lon_min-2,lon_max+2),
    latitude = slice(lat_min-2,lat_max+2),
    time = slice(time_min, time_max))
subsetOcean.load()


Ocean_collocated = subsetOcean.interp(
    latitude = lat_points,
    longitude = lon_points,
    time = time_points,
    method = 'linear')


### Subset and Collocate Ocean Data ###
subsetOscar = Oscar.sel(
    longitude = slice(lon_min-2,lon_max+2),
    latitude = slice(lat_max+2,lat_min-2),
    time = slice(time_min, time_max))
subsetOscar.load()


######## Exploring the impact of wind and ocean currents - are ocean currents significcant? #####

#### ------------ Wind Stress Calculation - Without Ocean currents CCMP collocated -----------  ####
# https://marine.rutgers.edu/dmcs/ms501/2004/Notes/Wilkin20041014.htm
# T = Cd * rho_air * U^2

Cd = 0.0013 # dimensionless drag coefficient
rho_air = 1.22 # (kg/m^3)

Tx = Cd * rho_air * (CCMPall_collocated['uwnd']**2)
Ty = Cd * rho_air * (CCMPall_collocated['vwnd']**2)



### Ekman Transport Calculations ###
## Uses wind stress calculated above from CCMP data ##

## Constants
# Omega
omega = 7.2921*10E-5

# Wind Stress Constants (commented out, where used to test ekman calculation before
#       wind stress was calculated)
#Tx = 0
#Ty = 0.1

## Mean Density of the Profile
# Veronica suggested using 1025 kg/m^3, but that is lower than the lowest value
#rho = (ARGO['Sigma_theta[kg/m^3]'].mean() + 1000)
# True density is sigma theta + 1000
rho = 1035

## Calculate Coriolis Parameter
#f = 2 * omega * np.sin(np.radians(ARGO['Lat [°N]'])) # f based on Argo Lat
f = 2 * omega * np.sin(np.radians(CCMPall_collocated['latitude'])) # f based on CCMP Lat


## Solving for Ekman Transport ##
CCMPall_collocated['V_ekman'] = -Tx / (rho * f)
CCMPall_collocated.V_ekman.attrs['units'] = 'm^2/s'
CCMPall_collocated['U_ekman'] = Ty / (rho * f)
CCMPall_collocated.U_ekman.attrs['units'] = 'm^2/s'


##### Wind Stress with Ocean #####

CCMPall_collocated['u_wind'] = CCMPall_collocated['uwnd'] - Ocean_collocated['ugos']
CCMPall_collocated['v_wind'] = CCMPall_collocated['vwnd'] - Ocean_collocated['vgos']

Tx_o = Cd * rho_air * (CCMPall_collocated['u_wind']**2)
Ty_o = Cd * rho_air * (CCMPall_collocated['v_wind']**2)


CCMPall_collocated['V_ekmanOce'] = -Tx_o / (rho * f)
CCMPall_collocated.V_ekmanOce.attrs['units'] = 'm^2/s'
CCMPall_collocated['U_ekmanOce'] = Ty_o / (rho * f)
CCMPall_collocated.U_ekmanOce.attrs['units'] = 'm^2/s'


##### Figures of Collocated Ekman Data #####

# V_ekman single plot
fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
CCMPall_collocated.V_ekman.plot(x="time", label = 'V_ekman', color = 'red')
CCMPall_collocated.V_ekmanOce.plot(x="time", label = 'V_ekmanOce', color = 'blue')
plt.legend()
plt.title('V_ekman with Ocean Data collocated to Argo Float')
plt.savefig(output_path + 'Collocated_V_ekmanVSV_ekmanOce' + '.png')

# U_ekman single plot
fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
CCMPall_collocated.U_ekman.plot(x="time", label = 'U_ekman', color = 'red')
CCMPall_collocated.U_ekmanOce.plot(x="time", label = 'U_ekmanOce', color = 'blue')
plt.legend()
plt.title('U_ekman vs U_ekmanOce')
plt.savefig(output_path + 'Collocated_U_ekmanVSU_ekmanOce' + '.png')



### Confirmation Plots ###

# Magnitude of Uwnd and Vwnd from Ocean Data
fig, axs = plt.subplots(2)
#plt.tight_layout()
CCMPall_collocated.uwnd.plot(x="time", ax=axs[0], label = 'vwnd')
axs[0].set_xlim([datetime.date(2019,2,7), datetime.date(2019,2,25)])
axs[0].get_xaxis().set_visible(False)
CCMPall_collocated.vwnd.plot(x="time", ax=axs[1], label = 'uwnd')
axs[1].set_xlim([datetime.date(2019,2,7), datetime.date(2019,2,25)])
#axs[1].set_xlim([window_collocated.time.min(), window_collocated.time.max()])
#axs[1].invert_yaxis()
#axs[1].set_ylim([150, 0])
#axs[1].set_ylabel('Depth [m]')
axs[0].title.set_text('vwnd [m/s]')
axs[1].title.set_text('uwnd [m/s]')
plt.xticks(rotation = 60)
plt.savefig(output_path + 'CCMPwindMagnitude' + '.png')

# Magnitude of Vgos and Ugos from Ocean Data
fig, axs = plt.subplots(2)
#plt.tight_layout()
Ocean_collocated.ugos.plot(x="time", ax=axs[0], label = 'vwnd')
axs[0].set_xlim([datetime.date(2019,2,7), datetime.date(2019,2,25)])
axs[0].get_xaxis().set_visible(False)
Ocean_collocated.vgos.plot(x="time", ax=axs[1], label = 'uwnd')
axs[1].set_xlim([datetime.date(2019,2,7), datetime.date(2019,2,25)])
#axs[1].set_xlim([window_collocated.time.min(), window_collocated.time.max()])
#axs[1].invert_yaxis()
#axs[1].set_ylim([150, 0])
#axs[1].set_ylabel('Depth [m]')
axs[0].title.set_text('vgos [m/s]')
axs[1].title.set_text('ugos [m/s]')
plt.savefig(output_path + 'OceanGeosMagnitude' + '.png')

# Wind Quiver Plot
fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
#CCMPall_collocated.plot.quiver(CCMPall_collocated.latitude, CCMPall_collocated.longitude, CCMPall_collocated.uwnd, CCMPall_collocated.vwnd)
CCMPall_collocated.plot.quiver(x = "latitude", y = "longitude", u = "uwnd", v = "vwnd")
plt.savefig(output_path + 'CCMPwndQuiver' + '.png')









#----------- Z_ekman Calculation ---------------------

# The correct approach to finding z_ekman is not while using data collocated to 
#   Argo data.  Argo was taken below the surface, for true Z_ekman, we want to work 
#   with just the true surface data, specified in the subsets of CCMP (wind) and 
#   Ocean (Copernicus data, although would like to change this to Oscar potentially)

# The CCMP wind data and Ocean data have different ranges in when reduced to their
# subsets (the subsets are established by the boundaries of the Argo Data).  subsetOcean
# has fewer points, and therefore it can be interpolated to CCMP wind in similar
# style as CCMP was collocated to the Argo Float.


# Establish Points of CCMP (subset) to focus on
#lat_points = xr.DataArray(ARGO_Surface['Lat [°N]'], dims = 'points')
#lon_points = xr.DataArray(ARGO_Surface['Lon [°E]']+360, dims = 'points')
#time_points = xr.DataArray(ARGO_Surface['date'], dims = 'points')

# Could omit the term points and just use variable names, but is useful if we 
# wanted to change lat points at all without touching the interp code
lat_points = subset.latitude
lon_points = subset.longitude
time_points = subset.time

# Interpolate lower resolution datasets to CCMP (higher resolution)
# Use in future a rolling mean over the time frame desired

# Copernicus interpolated to CCMP
IntOcean = subsetOcean.interp(
    latitude = lat_points,
    longitude = lon_points,
    time = time_points,
    method = 'linear')

# Oscar interpolated to CCMP
IntOscar = subsetOscar.interp(
    latitude = lat_points,
    longitude = lon_points,
    time = time_points,
    method = 'linear')

##### Wind Stress Calculation for Z_ekman #####
### There are multiple forms of wind stress here to be able to compare Z_ekman with wind,
### with wind and copernicus ocean data, and with wind and Oscar ocean data.

# https://marine.rutgers.edu/dmcs/ms501/2004/Notes/Wilkin20041014.htm
# T = Cd * rho_air * U^2


# Impact of Ocean currents from Copernicus on wind
subset['u_Cop'] = subset['uwnd'] - IntOcean['ugos']
subset['v_Cop'] = subset['vwnd'] - IntOcean['vgos']

# Impact of Ocean currents from Oscar on wind 
subset['u_Osc'] = subset['uwnd'] - IntOscar['u'] 
subset['v_Osc'] = subset['vwnd'] - IntOscar['v'] 


## Wind Stress Calculations for Wind, Copernicus, and Oscar data

Cd = 0.0013 # dimensionless drag coefficient
rho_air = 1.22 # (kg/m^3)

# Wind stres with just subset u and v wind components just to look
Tx_wnd = Cd * rho_air * (subset['uwnd']**2)
Ty_wnd = Cd * rho_air * (subset['vwnd']**2)

# Wind Stress with Copernicus
Tx_Cop = Cd * rho_air * (subset['u_Cop']**2)
Ty_Cop = Cd * rho_air * (subset['v_Cop']**2)

# Wind Stress with Oscar
Tx_Osc = Cd * rho_air * (subset['u_Osc']**2) #replace with Oscar variables
Ty_Osc = Cd * rho_air * (subset['v_Osc']**2) #replace with Oscar variables





##################### Ekman Transport Calculations ###
# Uses wind stress calculated above #

## Constants
# Omega
omega = 7.2921*10E-5

## Mean Density of the Profile
# Veronica suggested using 1035 kg/m^3, but that is lower than the lowest value
#rho = (ARGO['Sigma_theta[kg/m^3]'].mean() + 1000)
# True density is sigma theta + 1000
rho = 1035

## Calculate Coriolis Parameter
f = 2 * omega * np.sin(np.radians(subset['latitude'])) # f based on CCMP Lat


## Solving for Ekman Transport - Wind Only ##
subset['V_ekman'] = -Tx_wnd / (rho * f)
subset.V_ekman.attrs['units'] = 'm^2/s'
subset['U_ekman'] = Ty_wnd / (rho * f)
subset.U_ekman.attrs['units'] = 'm^2/s'

## Solving for Ekman Transport - Wind and Copernicus ##
subset['V_ekman_Cop'] = -Tx_Cop / (rho * f)
subset.V_ekman_Cop.attrs['units'] = 'm^2/s'
subset['U_ekman_Cop'] = Ty_Cop / (rho * f)
subset.U_ekman_Cop.attrs['units'] = 'm^2/s'

## Solving for Ekman Transport - Wind and Oscar ##
subset['V_ekman_Osc'] = -Tx_Osc / (rho * f)
subset.V_ekman_Osc.attrs['units'] = 'm^2/s'
subset['U_ekman_Osc'] = Ty_Osc / (rho * f)
subset.U_ekman_Osc.attrs['units'] = 'm^2/s'



################ Ekman Pumping - Wind Only ###

# Change in Ekman
subset["dU_ek"] = subset.U_ekman.diff(dim="longitude")
subset.dU_ek.attrs['units'] = 'm^2/s'
subset["dV_ek"] = subset.V_ekman.diff(dim="latitude")
subset.dV_ek.attrs['units'] = 'm^2/s'

# Change in Distance (need to change to a center differencing product)
dr = 2*np.pi*(6371.136*1000)/360.0 #radius of earth in m

subset["dx"] = subset.longitude.diff(dim="longitude")*dr*np.cos(np.deg2rad(subset.latitude))
subset.dx.attrs['units'] = 'm'
subset["dy"] = subset.latitude.diff(dim="latitude")*dr
subset.dy.attrs['units'] = 'm'

# Create Lat and Lon midpoints
#subset["difflon"] = subset.longitude.diff(dim="longitude")#*np.cos(np.deg2rad(subset.latitude))
#subset["midlon"] = subset.longitude + (subset.difflon/2)
#subset["difflat"] = subset.latitude.diff(dim="latitude")
#subset["midlat"] = subset.latitude + (subset.difflat/2)

# Combine Terms
subset["dUdx"] = (subset.dU_ek / subset.dx) # diff along longitude
subset.dUdx.attrs['units'] = 'm/s'
subset["dVdy"] = (subset.dV_ek / subset.dy) # diff along latitude
subset.dVdy.attrs['units'] = 'm/s'

# Z_ekman
subset['Z_ekman'] = (subset.dUdx + subset.dVdy) * 86400
# Multiplying by 86400 changes units from m/s to m/day
subset.Z_ekman.attrs['units'] = 'm/day'

# Create new DataArray to plot Z_ekman on midpoints
#mid_plotZ = xr.DataArray(
#    data = subset.Z_ekman,
#    dims=["time", "latitude", "longitude"],
#    coords = dict(
#        longitude = subset.midlon,
#        latitude = subset.midlat,
#        time = subset.time))
#mid_plotZ['Z_ekman'] = subset.Z_ekman
#mid_plotZ.Z_ekman.sortby(mid_plotZ.time, ascending = True)
#mid_plotZ.Z_ekman.reset_coords()
#mid_plotZ.sortby('time', ascending=True)
#mid_plotZ.reindex(time=sorted(mid_plotZ.time))



################ Ekman Pumping - Wind and Copernicus ###

### Change in Ekman
subset["dU_ek_Cop"] = subset.U_ekman_Cop.diff(dim="longitude")
subset.dU_ek_Cop.attrs['units'] = 'm^2/s'

subset["dU_ek_Cop_center"] = subset.U_ekman_Cop.differentiate("longitude") #center diff
subset.dU_ek_Cop_center.attrs['units'] = 'm^2/s'

subset["dV_ek_Cop"] = subset.V_ekman_Cop.diff(dim="latitude")
subset.dV_ek_Cop.attrs['units'] = 'm^2/s'

subset["dV_ek_Cop_center"] = subset.V_ekman_Cop.differentiate("latitude") #center diff
subset.dV_ek_Cop_center.attrs['units'] = 'm^2/s'

### Change in Distance (need to change to a center differencing product)
subset["dx_Cop"] = subset.longitude.diff(dim="longitude")*dr*np.cos(np.deg2rad(subset.latitude))
subset.dx_Cop.attrs['units'] = 'm'

subset["dy_Cop"] = subset.latitude.diff(dim="latitude")*dr
subset.dy_Cop.attrs['units'] = 'm'

### Combine Terms
subset["dUdx_Cop"] = (subset.dU_ek_Cop / subset.dx_Cop) # diff along longitude
subset.dUdx_Cop.attrs['units'] = 'm/s'

subset["dVdy_Cop"] = (subset.dV_ek_Cop / subset.dy_Cop) # diff along latitude
subset.dVdy_Cop.attrs['units'] = 'm/s'

subset["dUdx_Cop_center"] = (subset.dU_ek_Cop_center / subset.dx_Cop) # diff along longitude
subset.dUdx_Cop_center.attrs['units'] = 'm/s'

subset["dVdy_Cop_center"] = (subset.dV_ek_Cop_center / subset.dy_Cop) # diff along latitude
subset.dVdy_Cop_center.attrs['units'] = 'm/s'

### Z_ekman
subset['Z_ekman_Cop'] = (subset.dUdx_Cop + subset.dVdy_Cop) * 86400
# Multiplying by 86400 changes units from m/s to m/day
subset.Z_ekman_Cop.attrs['units'] = 'm/day'

subset['Z_ekman_Cop_center'] = (subset.dUdx_Cop_center + subset.dVdy_Cop_center) * 86400
# Multiplying by 86400 changes units from m/s to m/day
subset.Z_ekman_Cop_center.attrs['units'] = 'm/day'


################ Ekman Pumping - Wind and Oscar ###

# Change in Ekman
subset["dU_ek_Osc"] = subset.U_ekman_Osc.diff(dim="longitude")
subset.dU_ek_Osc.attrs['units'] = 'm^2/s'
subset["dV_ek_Osc"] = subset.V_ekman_Osc.diff(dim="latitude")
subset.dV_ek_Osc.attrs['units'] = 'm^2/s'

# Change in Distance (need to change to a center differencing product)
subset["dx_Osc"] = subset.longitude.diff(dim="longitude")*dr*np.cos(np.deg2rad(subset.latitude))
subset.dx_Osc.attrs['units'] = 'm'
subset["dy_Osc"] = subset.latitude.diff(dim="latitude")*dr
subset.dy_Osc.attrs['units'] = 'm'

# Combine Terms
subset["dUdx_Osc"] = (subset.dU_ek_Osc / subset.dx_Osc) # diff along longitude
subset.dUdx_Osc.attrs['units'] = 'm/s'
subset["dVdy_Osc"] = (subset.dV_ek_Osc / subset.dy_Osc) # diff along latitude
subset.dVdy_Osc.attrs['units'] = 'm/s'

# Z_ekman
subset['Z_ekman_Osc'] = (subset.dUdx_Osc + subset.dVdy_Osc) * 86400 
# Multiplying by 86400 changes units from m/s to m/day
subset.Z_ekman_Osc.attrs['units'] = 'm/day'





######### Plotting Subset Ekman Components #####

# V_ekman - just wind
var='V_ekman'
t = '2019-02-18'
fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
sc = subset[var].sel(time = t).mean(dim = 'time').plot(cmap="bwr")
plt.title('V_ekman mean for ' + t)
ax.set_ylim([-54, -51])
ax.set_xlim([211.5, 213.5])
sc.set_clim(vmin = -0.05, vmax = 0.05)
plt.savefig(output_path + t + 'V_ekman_wnd' + '.png')

# U_ekman - just wind
var='U_ekman'
t = '2019-02-18'
fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
sc = subset[var].sel(time = t).mean(dim = 'time').plot(cmap="bwr")
plt.title('U_ekman mean for ' + t)
ax.set_ylim([-54, -51])
ax.set_xlim([211.5, 213.5])
sc.set_clim(vmin = -0.1, vmax = 0.1)
plt.savefig(output_path + t + 'U_ekman_wnd' + '.png')

# Z_ekman - just wind
var='Z_ekman'
t = '2019-02-18'
fig = plt.figure(num=None, dpi=80, figsize=(8,3), facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
sc = subset[var].sel(time = t).mean(dim = 'time').plot(cmap="bwr")
plt.title('Z_ekman for ' + t)
ax.set_ylim([-54, -50])
ax.set_xlim([211.5, 217])
#sc.set_clim(vmin = -0.9e-8, vmax = 0.9e-8)
plt.savefig(output_path + t + 'Z_ekman_wnd' + '.png') #GSS POSTER

# Z_ekman_Cop - Wind and Copernicus
var='Z_ekman_Cop'
t = '2019-02-18'
fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
sc = subset[var].sel(time = t).mean(dim = 'time').plot(cmap="bwr")
plt.title('Z_ekman_Cop for ' + t)
ax.set_ylim([-54, -50])
ax.set_xlim([211.5, 217])
sc.set_clim(vmin = -0.4, vmax = 0.4)
plt.savefig(output_path + t + 'Z_ekman_Cop' + '.png')

# Z_ekman_Cop_center - Wind and Copernicus with center differencing
var='Z_ekman_Cop_center'
t = '2019-02-18'
fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
sc = subset[var].sel(time = t).mean(dim = 'time').plot(cmap="bwr")
plt.title('Z_ekman_Cop_Center for ' + t)
ax.set_ylim([-54, -50])
ax.set_xlim([211.5, 217])
sc.set_clim(vmin = -0.4, vmax = 0.4)
plt.savefig(output_path + t + 'Z_ekman_Cop_center' + '.png')

# Z_ekman_Osc - Wind and Osc
var='Z_ekman_Osc'
t = '2019-02-18'
fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
sc = subset[var].sel(time = t).mean(dim = 'time').plot(cmap="bwr")
plt.title('Z_ekman_Osc for ' + t)
ax.set_ylim([-54, -50])
ax.set_xlim([211.5, 217])
#sc.set_clim(vmin = -0.9e-8, vmax = 0.9e-8)
plt.savefig(output_path + t + 'Z_ekman_Osc' + '.png')


# MLD Vs. Time corresponding to t in Z_ekman plot - just wind
fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
plt.plot(MLDgrad['date'], MLDgrad['MLD'], c='green', label='MLD Density Gradient')
plt.axvline(t, c ='red', linestyle='dashed')
ax.invert_yaxis()
ax.set_ylim([MLDdepth, 0])
ax.set_xlim([datetime.date(2019,2,8), datetime.date(2019,3,5)])
plt.title('MLD [m] with Time, Dashed Line at Time = ' + t)
plt.savefig(output_path + t + 'MLD_TimeMarkforZ_ekman' + '.png')
plt.show()



# PLotting Z_ekman on midpoints - this is the preferred method
# If works, this method will be applied to the gif
#var='Z_ekman'
#t = '2019-02-18'
#fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
#ax = fig.add_axes([0.1, 0.1, .8, .8])
#sc = mid_plotZ[var].sel(time = t).mean(dim = 'time').plot(cmap="bwr")
#plt.title('Z_ekman for ' + t)
#ax.set_ylim([-54, -50])
#ax.set_xlim([211.5, 217])
#sc.set_clim(vmin = -0.9e-8, vmax = 0.9e-8)
#plt.savefig(output_path + t + 'Midpoint_Z_ekman' + '.png')


### Attempt at gif for Z_ekman - currently using wind only Z_ekman
# Will try to add vectors as well

# Z_ekman just wind
for i in range(100):
    subset.Z_ekman[i,:,:].plot(figsize=(8,3), cmap="bwr") 
    #subset[i,:,:].plot.quiver(x = "latitude", y = "longitude", u = "uwnd", v = "vwnd")
    ax = fig.add_axes([0.1, 0.1, .8, .8])
    ax.set_ylim([-54, -50])
    ax.set_xlim([211.5, 217])
    plt.title("Z_ekman (CCMP Only) for Time = " + str(subset.coords['time'].values[i])[:13])
    #plt.show()
    plt.savefig(output_path + f"gif_plots/Z_Ekman_Animation_01_frame_{i:03}.png")
    plt.close()

# Z_ekman_Cop
for i in range(100):
    subset.Z_ekman_Cop[i,:,:].plot(figsize=(8,3), cmap="bwr") 
    #subset[i,:,:].plot.quiver(x = "latitude", y = "longitude", u = "uwnd", v = "vwnd")
    ax = fig.add_axes([0.1, 0.1, .8, .8])
    ax.set_ylim([-54, -50])
    ax.set_xlim([211.5, 217])
    plt.title("Z_ekman (CCMP and Copernicus) for Time = " + str(subset.coords['time'].values[i])[:13])
    #plt.show()
    plt.savefig(output_path + f"gif_plots/Z_Ekman_Cop_Animation_01_frame_{i:03}.png")
    plt.close()
    
# Z_ekman_Osc
for i in range(100):
    subset.Z_ekman_Osc[i,:,:].plot(figsize=(8,3), cmap="bwr") 
    #subset[i,:,:].plot.quiver(x = "latitude", y = "longitude", u = "uwnd", v = "vwnd")
    ax = fig.add_axes([0.1, 0.1, .8, .8])
    ax.set_ylim([-54, -50])
    ax.set_xlim([211.5, 217])
    plt.title("Z_ekman (CCMP and Oscar) for Time = " + str(subset.coords['time'].values[i])[:13])
    #plt.show()
    plt.savefig(output_path + f"gif_plots/Z_Ekman_Osc_Animation_01_frame_{i:03}.png")
    plt.close()


#### Currently using command line tools to create gifs
# Must be in the directory where the .png files are located in order to copy and paste this
# Also must have ffmpeg installed (I did so using conda)

# ffmpeg -framerate 3 -i CCMP_Z_Ekman_Animation_01_frame_%02d.png test.mp4
# CCMP_Z_Ekman_Animation_01_frame_###.png
#!ffmpeg -framerate 2 -pattern_type glob -i "generated/gif_plots/Z_Ekman_Animation_01_frame_%03d.png" -q:v 0 -vcodec mpeg4 -r 20 ./generated/gif_plots/Zekman.mp4 -y
#!ffmpeg -framerate 2 -pattern_type glob -i "CCMP_Z_Ekman_Animation_01_frame_*.png" -q:v 0 -vcodec mpeg4 -r 20 Zekman.mp4 -y





# ----------------------- Eddy MAtchups ----------
# Not all of the changes in the MLD seem to be explained by ekman pumping alone,
# so it is likely that the float was moving around in different portions of the eddy.
# By doing eddy-matchups, we can see where the flaot was compared to the center of the eddy,
# and compare that to Biogeochemical plots, MLD, and Ekman pumping.

##### Eddy Matchups to Argo Data #####
# Eddy match up may only take values of xr, Argo is in pd
#eddy = eddy_matchup.match(
#    SD.longitude,
#    SD.latitude,
#    SD.time,
#    latmin=SD.latitude.min(),
#    latmax=SD.latitude.max(),
#    radiusrange=1.2)

ARGO_Subset = ARGO.loc[(ARGO['Station'] >= 97) & (ARGO['Station'] <= 121)]

ArgoEddies = eddy_matchup.match(
    ARGO_Subset['Lon [°E]'],
    ARGO_Subset['Lat [°N]'],
    ARGO_Subset['date'],
    latmin=ARGO_Subset['Lat [°N]'].min(),
    latmax=ARGO_Subset['Lat [°N]'].max(),
    radiusrange=1.5) # Was 1.2 Change threshold value on this **************


### Distance From Center Vs. Time
fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
plt.plot(ArgoEddies.time, ArgoEddies.eddy_dist_to_ctr)
ax.set_xlim([datetime.date(2019,2,8), datetime.date(2019,3,5)])
plt.title('Distance from Eddy Center with Time')
plt.xlabel('Time')
plt.ylabel('Distance from Eddy Center (m)')
plt.savefig(output_path + 'ArgoDistance_from_EddyCenter' + '.png')
plt.show()
plt.close()





##### Plotting MLD and distance Float Distance from Eddy #####
# Might add Z_ekman to that subplot in the future

#MLD depth
var='Z_ekman'
t = '2019-02-18'
fig = plt.subplot(2,1,2)
plt.tight_layout()
sc = subset[var].sel(time = t).mean(dim = 'time').plot(cmap="bwr")
plt.gca().set_ylim([-54, -50])
plt.gca().set_xlim([211.5, 217])
#plt.xticks(rotation = 60)
plt.gca().title.set_text("Z_ekman for Time = " + t)

fig = plt.subplot(2,2,1)
plt.tight_layout()
plt.plot(MLDgrad['date'], MLDgrad['MLD'], c='green', label='MLD Density Gradient')
plt.axvline(t, c ='red', linestyle='dashed')
plt.gca().invert_yaxis()
plt.gca().set_ylim([MLDdepth, 0])
plt.gca().set_xlim([datetime.date(2019,2,8), datetime.date(2019,3,5)])
#plt.xticks(rotation = 60)
#plt.gca().get_xaxis().set_visible(False)
plt.gca().title.set_text('MLD [m] with Time')

#Distance from Eddy Center
fig = plt.subplot(2,2,2)
plt.tight_layout()
#plt.axvline(t, c = 'red', linestyle='dashed')
plt.plot(ArgoEddies.time, ArgoEddies.eddy_dist_to_ctr)
plt.gca().set_xlim([datetime.date(2019,2,8), datetime.date(2019,3,5)])
#plt.xticks(rotation = 60)
plt.gca().title.set_text('Distance from Eddy Center with Time')
plt.savefig(output_path + t + 'MLD+EddyDistFromCenter+Z_ekman_Subplots' + '.png')
plt.show()
plt.close()



# -------- Rolling Means of Z_ekman -------- #
#http://xarray.pydata.org/en/stable/generated/xarray.DataArray.rolling.html
#https://stackoverflow.com/questions/45992983/python-xarray-rolling-mean-example

# Z_ekman using just wind
subset["Z_ekman_roll"] = subset.Z_ekman.rolling(time=4).mean()

# Z_ekman using Copernicus ocean currents
subset["Z_ekman_Cop_roll"] = subset.Z_ekman_Cop.rolling(time=4).mean()

# Z_ekman using Oscar ocean currents
subset["Z_ekman_Osc_roll"] = subset.Z_ekman_Osc.rolling(time=4).mean()

### Plot time series to compare rolling averaging
dsloc = subset.sel(longitude=-146.706+360,latitude=-51.407,method='nearest')

# Z_ekman based on wind alone
fig = plt.figure(num=None, figsize=(10,5), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
dsloc['Z_ekman_roll'].plot(label = "Daily Rolling Mean")
dsloc['Z_ekman'].plot(label = "6-Hourly")
ax.set_title('Z_ekman Timeseries to compare Rolling Mean')
ax.set_xlim([datetime.date(2019,2,8), datetime.date(2019,3,5)])
ax.set_ylim(-0.3, 0.4)
plt.legend()
plt.savefig(output_path + 'Z_ekman_RollingMean_Comparison' + '.png')
plt.show()

# Z_ekman Copernicus
fig = plt.figure(num=None, figsize=(10,5), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
dsloc['Z_ekman_Cop_roll'].plot(label = "Daily Rolling Mean")
dsloc['Z_ekman_Cop'].plot(label = "6-Hourly")
ax.set_title('Z_ekman_Cop Timeseries to compare Rolling Mean')
ax.set_xlim([datetime.date(2019,2,8), datetime.date(2019,3,5)])
ax.set_ylim(-0.3, 0.4)
plt.legend()
plt.savefig(output_path + 'Z_ekman_Cop_RollingMean_Comparison' + '.png')
plt.show()

# Z_ekman Oscar
fig = plt.figure(num=None, figsize=(10,5), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
dsloc['Z_ekman_Osc_roll'].plot(label = "Daily Rolling Mean")
dsloc['Z_ekman_Osc'].plot(label = "6-Hourly")
ax.set_title('Z_ekman_Osc Timeseries to compare Rolling Mean')
ax.set_xlim([datetime.date(2019,2,8), datetime.date(2019,3,5)])
ax.set_ylim(-0.3, 0.4)
plt.legend()
plt.savefig(output_path + 'Z_ekman_Osc_RollingMean_Comparison' + '.png')
plt.show()


#subset['x'] = subset.dU_ek_Cop - subset.dU_ek_Cop_center
dsloc['Z_ekman_Cop'].plot(label = "Forward Diff")
dsloc['Z_ekman_Cop_center'].plot(label = "Center Diff")
#dsloc['Cop_center_diff'].plot(label = "difference between the two")
x = dsloc['Z_ekman_Cop']-dsloc['Z_ekman_Cop_center']
plt.plot(subset.time, x, label = "Difference")
plt.legend()
plt.show()






# ------------------- Equation of State ------------------- #

# Calculating density from equation of state and comparing it to density given
# in Argo Float file

# https://teos-10.github.io/GSW-Python/intro.html 

# Need:
    # sea pressure in dbar (sea pressure = absolute pressure - 10.1325 )
    # absolute salinity in g/kg
    # conservative temperature in degree C


###### This attempt is assuming that the pressure given is not sea pressure, but absolute pressure
### If density is significantly different from my float density, will try nest without changing pressure



# Sea Pressure 
ARGO['sea_press[dbar]'] = ARGO['Pressure[dbar]'] - 10.1325

# Salinity pss (Practical Salinity Scale) to Absolute salinity (g/kg)
ARGO['abs_sal[g/kg]'] = gsw.SA_from_SP(ARGO['Salinity[pss]'], ARGO['sea_press[dbar]'],
                                       ARGO['Lon [°E]'], ARGO['Lat [°N]']) # Absolute Salinity

# Conservative Temperature
ARGO['con_temp(C)'] = gsw.CT_from_t(ARGO['abs_sal[g/kg]'], ARGO['Temperature[°C]'],
                                    ARGO['sea_press[dbar]']) # Conservative Temperature

# In situ Desnity
ARGO['insitu_dens(kg/m)'] = gsw.density.rho(ARGO['abs_sal[g/kg]'], ARGO['con_temp(C)'], ARGO['sea_press[dbar]']) # In Situ Density

### Compare In situ Density (kg/m) to Density given (kg/m^3)
diff = ARGO['insitu_dens(kg/m)'] - (ARGO['Density [kg/m^3]']+1000)

# In situ Desnity without changing pressure
ARGO['insitu_dens(kg/m)2'] = gsw.density.rho(ARGO['abs_sal[g/kg]'], ARGO['con_temp(C)'], ARGO['Pressure[dbar]']) # In Situ Density

### Compare In situ Density (kg/m) to Density given (kg/m^3)
diff2 = ARGO['insitu_dens(kg/m)2'] - (ARGO['Density [kg/m^3]']+1000)

difftot = diff2-diff


#### Create Scatter Plots of the calculated density vs. given density (density rho calculation not sigma0!)
fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
plt.scatter(ARGO['Station'],ARGO['insitu_dens(kg/m)2'], label = 'Calculated Density')
plt.scatter(ARGO['Station'],ARGO['Sigma_theta[kg/m^3]']+1000, label = 'Density given by BGC Argo')
plt.legend()
ax.invert_yaxis()
ax.set_title('Calculated Vs. Given Density')
ax.set_xlabel('Station Number')
ax.set_ylabel('Density [kg/m^3]')

var='Sigma_theta[kg/m^3]'
fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
sc=ax.scatter(ARGO['date'],ARGO['Depth[m]'], c=ARGO[var]+1000, cmap = 'cool')
#ax.plot(MLD['date'], MLD['MLD'], c='blue', label = 'MLD Density Threshold')
ax.plot(MLDgrad['date'], MLDgrad['MLD'], c='green', label='MLD Density Gradient')
#plt.axvline(x=start)
ax.invert_yaxis()
ax.set_title(var+' for Float '+floatnum)
#ax.set_ylim([MLDdepth, 0])
#ax.set_xlim([datetime.date(2019,2,1), datetime.date(2019,3,1)])
#ax.set_xlim([datetime.date(2019,2,8), datetime.date(2019,3,5)])
#ax.set_xlim([pd.to_datetime(start_dateprof), pd.to_datetime(end_dateprof)])
plt.xticks(rotation = 60)
plt.legend()
cb=plt.colorbar(sc)
cb.set_label(var)
# automatically adjusts the colorbar based on the range of values youre plotting
#sc.set_clim(vmin = ARGO[var].loc[(ARGO['Depth[m]']<MLDdepth)].min(), 
#            vmax = ARGO[var].loc[(ARGO['Depth[m]']<MLDdepth)].max()) 
#sc.set_clim(vmin = 2090, vmax = 2130)
#plt.savefig(output_path + floatnum + 'bgcDIC' + '.png')

var='insitu_dens(kg/m)2'
fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
sc=ax.scatter(ARGO['date'],ARGO['Depth[m]'], c=ARGO[var], cmap = 'cool')
#ax.plot(MLD['date'], MLD['MLD'], c='blue', label = 'MLD Density Threshold')
ax.plot(MLDgrad['date'], MLDgrad['MLD'], c='green', label='MLD Density Gradient')
#plt.axvline(x=start)
ax.invert_yaxis()
ax.set_title(var+' for Float '+floatnum)
#ax.set_ylim([MLDdepth, 0])
#ax.set_xlim([datetime.date(2019,2,1), datetime.date(2019,3,1)])
#ax.set_xlim([datetime.date(2019,2,8), datetime.date(2019,3,5)])
#ax.set_xlim([pd.to_datetime(start_dateprof), pd.to_datetime(end_dateprof)])
plt.xticks(rotation = 60)
plt.legend()
cb=plt.colorbar(sc)
cb.set_label(var)
# automatically adjusts the colorbar based on the range of values youre plotting
#sc.set_clim(vmin = ARGO[var].loc[(ARGO['Depth[m]']<MLDdepth)].min(), 
#            vmax = ARGO[var].loc[(ARGO['Depth[m]']<MLDdepth)].max()) 
#sc.set_clim(vmin = 2090, vmax = 2130)
#plt.savefig(output_path + floatnum + 'bgcDIC' + '.png')

### Scatter Plots showing Density Vs. Temp and Vs. Sal
# Temp
fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
plt.scatter(ARGO['Temperature[°C]'],ARGO['insitu_dens(kg/m)2'], label = 'Calculated Density')
plt.scatter(ARGO['Temperature[°C]'],ARGO['Sigma_theta[kg/m^3]']+1000, label = 'Density given by BGC Argo', c='orange')
plt.legend()
ax.invert_yaxis()
ax.set_title('Density Vs. Temperature')
ax.set_xlabel('Temperature[°C]')
ax.set_ylabel('Density [kg/m^3]')

fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
plt.scatter(ARGO['Temperature[°C]'],ARGO['insitu_dens(kg/m)2'], label = 'Calculated Density')
#plt.scatter(ARGO['Temperature[°C]'],ARGO['Sigma_theta[kg/m^3]']+1000, label = 'Density given by BGC Argo', c='orange')
#plt.legend()
ax.invert_yaxis()
ax.set_title('Calculated Density Vs. Temperature')
ax.set_xlabel('Temperature[°C]')
ax.set_ylabel('Density [kg/m^3]')

fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
#plt.scatter(ARGO['Temperature[°C]'],ARGO['insitu_dens(kg/m)2'], label = 'Calculated Density')
plt.scatter(ARGO['Temperature[°C]'],ARGO['Sigma_theta[kg/m^3]']+1000, label = 'Density given by BGC Argo', c ='orange')
#plt.legend()
ax.invert_yaxis()
ax.set_title('Given Density Vs. Temperature')
ax.set_xlabel('Temperature[°C]')
ax.set_ylabel('Density [kg/m^3]')


# Salinity
fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
plt.scatter(ARGO['Salinity[pss]'],ARGO['insitu_dens(kg/m)2'], label = 'Calculated Density')
plt.scatter(ARGO['Salinity[pss]'],ARGO['Sigma_theta[kg/m^3]']+1000, label = 'Density given by BGC Argo', c='orange')
plt.legend()
ax.invert_yaxis()
ax.set_title('Density Vs. Salinity')
ax.set_xlabel('Salinity[pss]')
ax.set_ylabel('Density [kg/m^3]')

fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
plt.scatter(ARGO['Salinity[pss]'],ARGO['insitu_dens(kg/m)2'], label = 'Calculated Density')
#plt.scatter(ARGO['Salinity[pss]'],ARGO['Sigma_theta[kg/m^3]']+1000, label = 'Density given by BGC Argo', c='orange')
#plt.legend()
ax.invert_yaxis()
ax.set_title('Calculated Density Vs. Salinity')
ax.set_xlabel('Salinity[pss]')
ax.set_ylabel('Density [kg/m^3]')

fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
#plt.scatter(ARGO['Salinity[pss]'],ARGO['insitu_dens(kg/m)2'], label = 'Calculated Density')
plt.scatter(ARGO['Salinity[pss]'],ARGO['Sigma_theta[kg/m^3]']+1000, label = 'Density given by BGC Argo', c='orange')
#plt.legend()
ax.invert_yaxis()
ax.set_title('Given Density Vs. Salinity')
ax.set_xlabel('Salinity[pss]')
ax.set_ylabel('Density [kg/m^3]')



#### -------- Calculating Sigma0 from equation of State -------
#Above, density was calculated which is not a proper comparisson to the potential density given 
# by SOCCOM.

# Sigma0 is calculated below using the conservative temperature and absolute salinity
# calculated for density rho

### Calculating sigma0
ARGO['sigma_calc'] = gsw.sigma0(ARGO['abs_sal[g/kg]'], ARGO['con_temp(C)'])

#### Create Scatter Plots of the calculated sigma vs. given sigma 
fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
plt.scatter(ARGO['Station'],ARGO['sigma_calc'], label = 'Calculated Sigma')
plt.scatter(ARGO['Station'],ARGO['Sigma_theta[kg/m^3]'], label = 'Sigma given by BGC Argo')
plt.legend()
ax.invert_yaxis()
ax.set_title('Calculated Vs. Given Sigma')
ax.set_xlabel('Station Number')
ax.set_ylabel('sigma0 [kg/m^3]')

var='Sigma_theta[kg/m^3]'
fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
sc=ax.scatter(ARGO['date'],ARGO['Depth[m]'], c=ARGO[var], cmap = 'cool')
#ax.plot(MLD['date'], MLD['MLD'], c='blue', label = 'MLD Density Threshold')
ax.plot(MLDgrad['date'], MLDgrad['MLD'], c='green', label='MLD Density Gradient')
#plt.axvline(x=start)
ax.invert_yaxis()
ax.set_title(var+' for Float '+floatnum)
#ax.set_ylim([MLDdepth, 0])
#ax.set_xlim([datetime.date(2019,2,1), datetime.date(2019,3,1)])
#ax.set_xlim([datetime.date(2019,2,8), datetime.date(2019,3,5)])
#ax.set_xlim([pd.to_datetime(start_dateprof), pd.to_datetime(end_dateprof)])
plt.xticks(rotation = 60)
plt.legend()
cb=plt.colorbar(sc)
cb.set_label(var)
# automatically adjusts the colorbar based on the range of values youre plotting
#sc.set_clim(vmin = ARGO[var].loc[(ARGO['Depth[m]']<MLDdepth)].min(), 
#            vmax = ARGO[var].loc[(ARGO['Depth[m]']<MLDdepth)].max()) 
#sc.set_clim(vmin = 2090, vmax = 2130)
#plt.savefig(output_path + floatnum + 'bgcDIC' + '.png')

var='sigma_calc'
fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
sc=ax.scatter(ARGO['date'],ARGO['Depth[m]'], c=ARGO[var], cmap = 'cool')
#ax.plot(MLD['date'], MLD['MLD'], c='blue', label = 'MLD Density Threshold')
ax.plot(MLDgrad['date'], MLDgrad['MLD'], c='green', label='MLD Density Gradient')
#plt.axvline(x=start)
ax.invert_yaxis()
ax.set_title(var+' for Float '+floatnum)
#ax.set_ylim([MLDdepth, 0])
#ax.set_xlim([datetime.date(2019,2,1), datetime.date(2019,3,1)])
#ax.set_xlim([datetime.date(2019,2,8), datetime.date(2019,3,5)])
#ax.set_xlim([pd.to_datetime(start_dateprof), pd.to_datetime(end_dateprof)])
plt.xticks(rotation = 60)
plt.legend()
cb=plt.colorbar(sc)
cb.set_label(var)
# automatically adjusts the colorbar based on the range of values youre plotting
#sc.set_clim(vmin = ARGO[var].loc[(ARGO['Depth[m]']<MLDdepth)].min(), 
#            vmax = ARGO[var].loc[(ARGO['Depth[m]']<MLDdepth)].max()) 
#sc.set_clim(vmin = 2090, vmax = 2130)
#plt.savefig(output_path + floatnum + 'bgcDIC' + '.png')

### Scatter Plots showing Density Vs. Temp and Vs. Sal
# Temp
fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
plt.scatter(ARGO['Temperature[°C]'],ARGO['sigma_calc'], label = 'Calculated sigma')
plt.scatter(ARGO['Temperature[°C]'],ARGO['Sigma_theta[kg/m^3]'], label = 'Sigma given by BGC Argo', c='orange')
plt.legend()
ax.invert_yaxis()
ax.set_title('Sigma0 Vs. Temperature')
ax.set_xlabel('Temperature[°C]')
ax.set_ylabel('Sigma0 [kg/m^3]')

fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
plt.scatter(ARGO['Temperature[°C]'],ARGO['sigma_calc'], label = 'Calculated sigma')
#plt.scatter(ARGO['Temperature[°C]'],ARGO['Sigma_theta[kg/m^3]']+1000, label = 'Density given by BGC Argo', c='orange')
#plt.legend()
ax.invert_yaxis()
ax.set_title('Calculated Sigma Vs. Temperature')
ax.set_xlabel('Temperature[°C]')
ax.set_ylabel('Sigma [kg/m^3]')

fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
#plt.scatter(ARGO['Temperature[°C]'],ARGO['insitu_dens(kg/m)2'], label = 'Calculated Density')
plt.scatter(ARGO['Temperature[°C]'],ARGO['Sigma_theta[kg/m^3]'], label = 'Sigma given by BGC Argo', c ='orange')
#plt.legend()
ax.invert_yaxis()
ax.set_title('Given Sigma Vs. Temperature')
ax.set_xlabel('Temperature[°C]')
ax.set_ylabel('Density [kg/m^3]')


# Salinity
fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
plt.scatter(ARGO['Salinity[pss]'],ARGO['sigma_calc'], label = 'Calculated Sigma')
plt.scatter(ARGO['Salinity[pss]'],ARGO['Sigma_theta[kg/m^3]'], label = 'Sigma given by BGC Argo', c='orange')
plt.legend()
ax.invert_yaxis()
ax.set_title('Sigma Vs. Salinity')
ax.set_xlabel('Salinity[pss]')
ax.set_ylabel('Sigma [kg/m^3]')

fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
plt.scatter(ARGO['Salinity[pss]'],ARGO['sigma_calc'], label = 'Calculated Sigma')
#plt.scatter(ARGO['Salinity[pss]'],ARGO['Sigma_theta[kg/m^3]'], label = 'Density given by BGC Argo', c='orange')
#plt.legend()
ax.invert_yaxis()
ax.set_title('Calculated Sigma Vs. Salinity')
ax.set_xlabel('Salinity[pss]')
ax.set_ylabel('Sigma [kg/m^3]')

fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.1, 0.1, .8, .8])
#plt.scatter(ARGO['Salinity[pss]'],ARGO['insitu_dens(kg/m)2'], label = 'Calculated Density')
plt.scatter(ARGO['Salinity[pss]'],ARGO['Sigma_theta[kg/m^3]'], label = 'Sigma given by BGC Argo', c='orange')
#plt.legend()
ax.invert_yaxis()
ax.set_title('Given Sigma Vs. Salinity')
ax.set_xlabel('Salinity[pss]')
ax.set_ylabel('Sigma [kg/m^3]')

