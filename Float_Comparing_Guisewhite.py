#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 16:54:32 2022

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

import scipy.io
import scipy.interpolate as interp






# ************************** Import and Organize Data **************************




#------------------- Import Single Argo Data with Eddy ---------------------

#  Note:  This is not the only location in the code where data is imported
##### Import ARGO Float File #####

# Currently set up to pull data from a locally downloaded source
# In this code, all current files are being downloaded from the local desktop of 
# the laptop

## Identify the file of the float you want to use:
# Link to float ID matchups - http://soccom.ucsd.edu/floats/SOCCOM_data_ref.html

## Link to downloaded SOCCOM data 
# The only downloaded data used for this code is for the single float 5904693
# Other float data is pulled from a pickle created from the SOCCOMfloats4Saildrone code (see more info
#   in "loading SOCCOM fleet data")
    
# Location where you want figures to save:
output_path = 'generated2/'


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







#---------- Sorting Through Data Quality Flags - Single Float ---------------------


##### Sort Through Quality Flagged Data with a For Loop #####

## For Loop designed to replace any bad data with NaN (determined by Quality Flags (QF)) 
# Only want to keep the "good" data which has a corresponding QF = 0
# For any value that has a QF not equal to 0, replace that specific data value with NaN
# THE EXCEPTION : NITRATE - the sensor experienced issues where moderate data (QF=1)
#    is likely still useful data.


## Create a Regular Expression to make sure For Loop isn't reading anything but useful 
##      data

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
        var4= 'DIC_LIAR[µmol/kg]'
        ARGO[var4] = np.where(ARGO.iloc[:,ARGO.columns.get_loc(var4)+1] == 0,
                              ARGO[var4], np.nan)
        var5= 'Nitrate[µmol/kg]' # may need to adjust quality flags due to sensor issue
        ARGO[var5] = np.where(ARGO.iloc[:,ARGO.columns.get_loc(var5)+1] <= 8,
                              ARGO[var5], np.nan) 
        ARGO[var5] = np.where(ARGO.iloc[:,ARGO.columns.get_loc(var5)] >= 0, #try to get rid of any negative values
                              ARGO[var5], np.nan)
#        ARGO[var5] = np.where(ARGO.iloc[:,ARGO.columns.get_loc('Station')] <= 120, #staion adjustment based on stations with bad quality flags
#                              ARGO[var5], np.nan)
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


###### Make Additional ARGO Variables within Single Float Dataframe#####

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







#----------------- Import Eddy Data from Dr. Chambers File ---------------------
# Data about the eddy encountered by the single float

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

# The data in this file is not used for statistcal analysis, rather for first look
#   with simple plots of the eddy location and movement during flot observations

EDDY = pd.read_csv("SO_eddy_63878.txt",error_bad_lines=False, header=None, 
                   skiprows=1, delimiter='\s+').T
EDDY.columns = ('Lon', 'Lat', 'Radius_Max_Velocity_From_Center[km]', 'SSH_Amp[m]')
# Makes Lon coordinates in terms of positive and negative values do to an issue
#   with the cartopy map for the Eddy Mercator Plots
EDDY['Lon'].loc[EDDY['Lon']>180] = EDDY['Lon'].loc[EDDY['Lon']>180]-360







# -------------------------- Import SOCCOM FLEET Data --------------------- #


# Read in pickle created in SOCCOM_eddy_matchups (Pickle created via other functions
#   SOCCOMFloats4Saildrone, SubsetEddies, SOCCOM_eddy_matchup)
# The SOCCOMFloats4Saildrone pulls SOCCOM fleet data from ftp://ftp.mbari.org/pub/SOCCOM/FloatVizData/QC
#   and organizes the data into a pickle snapshot of the full data, only pulling
#   data where QF = 0 (good)

#Import full SOCCOM Fleet data
SOCCOM = pd.read_pickle('/Users/nicolausf/Desktop/Chambers_Argo/SOCCOM_snapshot.pkl')
SOCCOM['Lon [°E]'].loc[SOCCOM['Lon [°E]']>180] = SOCCOM['Lon [°E]'].loc[SOCCOM['Lon [°E]']>180]-360

# The pickle is run through SubsetEddies, which creates a smaller pickle with unique 
#   stationsto be used for identifying profiles within eddies in SOCCOM_eddy_matchip
# The small pickle with unique stations (u_stat_SOCCOM) is run through SOCCOM_eddy_matchup
#   which uses Veronica's eddy matchup function to obtain the lat, lon, and date of 
#   SOCCOM profiles and compare them to the Chelton database (surface altimetry identified eddies)
#   Values from the eddy_matchup function were re-saved onto u_stat_SOCCOM

u_stat_SOCCOM = pd.read_pickle('/Users/nicolausf/Desktop/Chambers_Argo/data/u_stat_SOCCOM.pkl')

# Set longitude bounds for Pacific Basin
neg_lon_min = -180
neg_lon_max = -75
lon_min = 150
lon_max = 179.99999999

# Cyclonic Vs. Anticyclonic Eddies
eddy_ID_cyclonic = u_stat_SOCCOM.loc[(u_stat_SOCCOM['eddy_type'] !=0) 
                                     & (u_stat_SOCCOM['eddy_type'] == -1)
                                     ]
eddy_ID_cyclonic['Lon [°E]'].loc[eddy_ID_cyclonic['Lon [°E]']>180] = eddy_ID_cyclonic['Lon [°E]'].loc[eddy_ID_cyclonic['Lon [°E]']>180]-360
eddy_ID_cyclonic = eddy_ID_cyclonic.loc[((eddy_ID_cyclonic['Lon [°E]'] >= neg_lon_min)
                                     & (eddy_ID_cyclonic['Lon [°E]'] <= neg_lon_max))
                                     | ((eddy_ID_cyclonic['Lon [°E]'] >= lon_min)
                                     & (eddy_ID_cyclonic['Lon [°E]'] <= lon_max))
                                     ]

eddy_ID_anticyclonic = u_stat_SOCCOM.loc[(u_stat_SOCCOM['eddy_type'] !=0) 
                                     & (u_stat_SOCCOM['eddy_type'] == 1)
                                     ]
eddy_ID_anticyclonic['Lon [°E]'].loc[eddy_ID_anticyclonic['Lon [°E]']>180] = eddy_ID_anticyclonic['Lon [°E]'].loc[eddy_ID_anticyclonic['Lon [°E]']>180]-360
eddy_ID_anticyclonic = eddy_ID_anticyclonic.loc[((eddy_ID_anticyclonic['Lon [°E]'] >= neg_lon_min)
                                     & (eddy_ID_anticyclonic['Lon [°E]'] <= neg_lon_max))
                                     | ((eddy_ID_anticyclonic['Lon [°E]'] >= lon_min)
                                     & (eddy_ID_anticyclonic['Lon [°E]'] <= lon_max))
                                     ]

eddy_ID_none = u_stat_SOCCOM.loc[(u_stat_SOCCOM['eddy_type']==0)]                              
eddy_ID_none['Lon [°E]'].loc[eddy_ID_none['Lon [°E]']>180] = eddy_ID_none['Lon [°E]'].loc[eddy_ID_none['Lon [°E]']>180]-360
eddy_ID_none = eddy_ID_none.loc[((eddy_ID_none['Lon [°E]'] >= neg_lon_min)
                                     & (eddy_ID_none['Lon [°E]'] <= neg_lon_max))
                                     | ((eddy_ID_none['Lon [°E]'] >= lon_min)
                                     & (eddy_ID_none['Lon [°E]'] <= lon_max))
                                     ]

# Print check
#print(len(eddy_ID_cyclonic['Station'].unique()))
#print(len(eddy_ID_anticyclonic['Station'].unique()))
#print(len(eddy_ID_none['Station'].unique()))
    

## Create Summer Subsets of Pacific Basin Data (Summer Months - January, February, March)
# Will be creating subsets for other seasons - seperate and perform as function

#Cyclonic Eddies Summer
ECS = eddy_ID_cyclonic.loc[(eddy_ID_cyclonic['mon/day/yr'] >= '01/01') 
                           & (eddy_ID_cyclonic['mon/day/yr'] <= '03/31')]
#Anticyclonic Eddies Summer
EAS = eddy_ID_anticyclonic.loc[(eddy_ID_anticyclonic['mon/day/yr'] >= '01/01') 
                               & (eddy_ID_anticyclonic['mon/day/yr'] <= '03/31')]
#Non-Eddies Summer
NES = eddy_ID_none.loc[(eddy_ID_none['mon/day/yr'] >= '01/01') 
                       & (eddy_ID_none['mon/day/yr'] <= '03/31')]








# ------------------ Load Kim and Orsi Fronts ------------#
# This data was downloaded from published work from Kim and Orsi 2014
# Had to be initially obtained from Dr. Chambers because no access by USF VPN

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


# ----- Load Subtropical Front from Orsi et al 1995 
# Could not use Kim and Orsi fronts for Subtropical Front because Kim and Orsi 
#   data did not include subtropical front
# Data also obtained from Dr. Chambers

# Subtropical Front
stf=pd.read_csv('Front_Txt_Files/stf.txt', header=None,sep='\s+', na_values='%',
                names=['lon','lat'])
stf_lon = stf['lon']
stf_lat = stf['lat']








# ----------------- Plot Fronts with ARGO Float Track and Individual Eddy -----------------------
# Code used to include data from Saildrone, including this moving forward is being discussed

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

# Create Circle Axes Coordinates
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

# Plot Fronts (uses circle axes from ARGO Float Track)
ax.set_boundary(circle, transform=ax.transAxes)
plt.plot(stf_lon, stf_lat, color='Red', transform=ccrs.PlateCarree(), label='Subtropical Front', linewidth = 1)
plt.plot(saf_lon, saf_lat, color='Orange', transform=ccrs.PlateCarree(), label='Subantarctic Front', linewidth = 1)
plt.plot(pf_lon, pf_lat, color='Yellow', transform=ccrs.PlateCarree(), label='Polar Front', linewidth = 1)
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
plt.title('Float ' + floatnum +' Track, Eddy and Fronts')
plt.savefig(output_path + floatnum + 'TrackFrontsEddyProfile' + '.png')








# ----------------- Plot Eddies/Non Eddies from SOCCOM Data -----------------------

# Set Dataframe (DF)
# DF options - EAS, ECS, NES
DF = NES  # change value for different plots
DF_str = 'NES' # change value to match above for proper labeling

# Set Figure and Map Characteristics
plt.figure(figsize=(9,5))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([150,-70,-70,-30], ccrs.PlateCarree(central_longitude=180))
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
gl = ax.gridlines()
gl.ylabels_left = True
gl.xlabels_bottom = True

# Plot Fronts (uses circle axes from ARGO Float Track)
#ax.set_boundary(circle, transform=ax.transAxes)
plt.plot(stf_lon, stf_lat, color='Red', transform=ccrs.PlateCarree(), 
         label='Subtropical Front')
plt.plot(saf_lon, saf_lat, color='Orange', transform=ccrs.PlateCarree(), #PLotting double for some reason???
         label='Subantarctic Front')
plt.plot(pf_lon, pf_lat, color='Yellow', transform=ccrs.PlateCarree(), 
         label='Polar Front')
plt.plot(saccf_lon, saccf_lat, color='Green', transform=ccrs.PlateCarree(), 
         label='Southern ACC Front')
plt.plot(sbdy_lon, sbdy_lat, color='Blue', transform=ccrs.PlateCarree(), 
         label='Southern Boundary')
#plt.legend()

# Plot DF
plt.scatter(DF['Lon [°E]'], DF['Lat [°N]'], 
            color='Black', transform=ccrs.PlateCarree(), s = 5, zorder = 1000) # change dot size
plt.title('Map of All Floats Matching ' + DF_str)
plt.savefig(output_path + DF_str + '_All_Map'  + '.png')







# ---------- Create Front Groups to Seperate EAS Data by Frontal Boundaries -----------
# Can be turned into a single function ****************

# Set range from front to still collect float data
lat_range = 2 #degrees
lat_str = str(lat_range)

np_EAS_lon = EAS['Lon [°E]'].values

front_groups_EAS = np.empty(len(EAS))
#loop through your float profiles
for n in range(len(EAS)):
    if np.isnan(np_EAS_lon[n]):
        continue
    #find index of closest longitude in front position
    closest_lon_index_stf = np.nanargmin(np.absolute(stf_lon - np_EAS_lon[n]))
    closest_lon_index_pf = np.nanargmin(np.absolute(pf_lon - np_EAS_lon[n]))
    closest_lon_index_saf = np.nanargmin(np.absolute(saf_lon - np_EAS_lon[n]))
    #print(closest_lon_index)
    #compare front latitude to float latitude
    lat_compare_stf = stf_lat[closest_lon_index_stf]
    lat_compare_pf = pf_lat[closest_lon_index_pf]
    lat_compare_saf = saf_lat[closest_lon_index_saf]
    print('I made it through defining the lat compare values for EAS Front Groups')
    if EAS['Lat [°N]'].values[n] > lat_compare_stf+lat_range:
        front_groups_EAS[n] = 1 # north of STF
    elif (EAS['Lat [°N]'].values[n] < lat_compare_stf+lat_range) & (EAS['Lat [°N]'].values[n] >
                                                                                    lat_compare_saf):
            front_groups_EAS[n] = 2 # between stf and saf 
    elif (EAS['Lat [°N]'].values[n] < lat_compare_saf) & (EAS['Lat [°N]'].values[n] >
                                                                                    lat_compare_pf-lat_range):
            front_groups_EAS[n] = 3 # between saf and pf 
    elif EAS['Lat [°N]'].values[n] < lat_compare_pf-lat_range:
        front_groups_EAS[n] = 4 # south of PF
    else:
        front_groups_EAS[n] = 0 



# ---------- Create Front Groups to Seperate ECS Data by Frontal Boundaries 
# Can be turned into a single function ****************

# Lat range is set in EAS Front Groups - want to use the same value for each

np_ECS_lon = ECS['Lon [°E]'].values

front_groups_ECS = np.empty(len(ECS))
#loop through your float profiles
for n in range(len(ECS)):
    if np.isnan(np_ECS_lon[n]):
        continue
    #find index of closest longitude in front position
    closest_lon_index_stf = np.nanargmin(np.absolute(stf_lon - np_ECS_lon[n]))
    closest_lon_index_pf = np.nanargmin(np.absolute(pf_lon - np_ECS_lon[n]))
    closest_lon_index_saf = np.nanargmin(np.absolute(saf_lon - np_ECS_lon[n]))
    #print(closest_lon_index)
    #compare front latitude to float latitude
    lat_compare_stf = stf_lat[closest_lon_index_stf]
    lat_compare_pf = pf_lat[closest_lon_index_pf]
    lat_compare_saf = saf_lat[closest_lon_index_saf]
    print('I made it through defining the lat compare values for ECS Front Groups')
    if ECS['Lat [°N]'].values[n] > lat_compare_stf+lat_range:
        front_groups_ECS[n] = 1 # north of STF
    elif (ECS['Lat [°N]'].values[n] < lat_compare_stf+lat_range) & (ECS['Lat [°N]'].values[n] >
                                                                                    lat_compare_saf):
            front_groups_ECS[n] = 2 # between stf and saf 
    elif (ECS['Lat [°N]'].values[n] < lat_compare_saf) & (ECS['Lat [°N]'].values[n] >
                                                                                    lat_compare_pf-lat_range):
            front_groups_ECS[n] = 3 # between saf and pf 
    elif ECS['Lat [°N]'].values[n] < lat_compare_pf-lat_range:
        front_groups_ECS[n] = 4 # south of PF
    else:
        front_groups_ECS[n] = 0 



# ---------- Create Front Groups to Seperate NES Data by Frontal Boundaries 
# Can be turned into a single function ****************

# Lat range is set in EAS Front Groups - want to use the same value for each

np_NES_lon = NES['Lon [°E]'].values

front_groups_NES = np.empty(len(NES))
#loop through your float profiles
for n in range(len(NES)):
    if np.isnan(np_NES_lon[n]):
        continue
    #find index of closest longitude in front position
    closest_lon_index_stf = np.nanargmin(np.absolute(stf_lon - np_NES_lon[n]))
    closest_lon_index_pf = np.nanargmin(np.absolute(pf_lon - np_NES_lon[n]))
    closest_lon_index_saf = np.nanargmin(np.absolute(saf_lon - np_NES_lon[n]))
    #print(closest_lon_index)
    #compare front latitude to float latitude
    lat_compare_stf = stf_lat[closest_lon_index_stf]
    lat_compare_pf = pf_lat[closest_lon_index_pf]
    lat_compare_saf = saf_lat[closest_lon_index_saf]
    print('I made it through defining the lat compare values for NES Front Groups')
    if NES['Lat [°N]'].values[n] > lat_compare_stf+lat_range:
        front_groups_NES[n] = 1 # north of STF
    elif (NES['Lat [°N]'].values[n] < lat_compare_stf+lat_range) & (NES['Lat [°N]'].values[n] >
                                                                                    lat_compare_saf):
            front_groups_NES[n] = 2 # between stf and saf 
    elif (NES['Lat [°N]'].values[n] < lat_compare_saf) & (NES['Lat [°N]'].values[n] >
                                                                                    lat_compare_pf-lat_range):
            front_groups_NES[n] = 3 # between saf and pf 
    elif NES['Lat [°N]'].values[n] < lat_compare_pf-lat_range:
        front_groups_NES[n] = 4 # south of PF
    else:
        front_groups_NES[n] = 0 







# ----------------------- Plot EAS Floats by Front Group  ----------------
# Can be turned into a single function ****************

# Set Front Group to Plot
front_group = 2 # This is currently also the front group being set for all of the following 
#   front group plots for ECS and NES 
front_group_str = str(front_group)

DF_fg = EAS.loc[front_groups_EAS ==2]

# Set Figure and Map Characteristics
plt.figure(figsize=(9,3))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([0,140,-70,-25], ccrs.PlateCarree(central_longitude=180))
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
gl = ax.gridlines()
gl.ylabels_left = True
gl.xlabels_bottom = True

# Plot Fronts (uses circle axes from ARGO Float Track)
#ax.set_boundary(circle, transform=ax.transAxes)
plt.plot(stf_lon, stf_lat, color='Red', transform=ccrs.PlateCarree(), 
         label='Subtropical Front')
plt.plot(saf_lon, saf_lat, color='Orange', transform=ccrs.PlateCarree(), 
         label='Subantarctic Front')
plt.plot(pf_lon, pf_lat, color='Yellow', transform=ccrs.PlateCarree(), 
         label='Polar Front')
#plt.plot(saccf_lon, saccf_lat, color='Green', transform=ccrs.PlateCarree(), 
         #label='Southern ACC Front')
#plt.plot(sbdy_lon, sbdy_lat, color='Blue', transform=ccrs.PlateCarree(), 
        # label='Southern Boundary')
#plt.legend()

# Plot ARGO Float Track (taken from ARGO Float Track)
plt.scatter(DF_fg['Lon [°E]'], DF_fg['Lat [°N]'], 
            color='Black', transform=ccrs.PlateCarree(), s = 5, zorder = 1000) # change dot size
plt.title('Map of Floats Matching EAS within Front Group '+ front_group_str)
plt.savefig(output_path +'EAS_Map_FG' + front_group_str + '.png')





# ------------------- Plot ECS Floats by Front Group  
# Can be turned into a single function ****************

# Set Front Group to Plot
#front_group = 2
#front_group_str = str(front_group)

DF_fg = ECS.loc[front_groups_ECS ==2]

# Set Figure and Map Characteristics
plt.figure(figsize=(9,3))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([0,140,-70,-25], ccrs.PlateCarree(central_longitude=180))
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
gl = ax.gridlines()
gl.ylabels_left = True
gl.xlabels_bottom = True

# Plot Fronts (uses circle axes from ARGO Float Track)
#ax.set_boundary(circle, transform=ax.transAxes)
plt.plot(stf_lon, stf_lat, color='Red', transform=ccrs.PlateCarree(), 
         label='Subtropical Front')
plt.plot(saf_lon, saf_lat, color='Orange', transform=ccrs.PlateCarree(), 
         label='Subantarctic Front')
plt.plot(pf_lon, pf_lat, color='Yellow', transform=ccrs.PlateCarree(), 
         label='Polar Front')
#plt.plot(saccf_lon, saccf_lat, color='Green', transform=ccrs.PlateCarree(), 
         #label='Southern ACC Front')
#plt.plot(sbdy_lon, sbdy_lat, color='Blue', transform=ccrs.PlateCarree(), 
        # label='Southern Boundary')
#plt.legend()

# Plot ARGO Float Track (taken from ARGO Float Track)
plt.scatter(DF_fg['Lon [°E]'], DF_fg['Lat [°N]'], 
            color='Black', transform=ccrs.PlateCarree(), s = 5, zorder = 1000) # change dot size
plt.title('Map of Floats Matching ECS within Front Group '+ front_group_str)
plt.savefig(output_path +'ECS_Map_FG' + front_group_str + '.png')





# -------------------- Plot NES Floats by Front Group  
# Can be turned into a single function ****************

# Set Front Group to Plot
#front_group = 2
#front_group_str = str(front_group)

DF_fg = NES.loc[front_groups_NES ==2]

# Set Figure and Map Characteristics
plt.figure(figsize=(9,3))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([0,140,-70,-25], ccrs.PlateCarree(central_longitude=180))
ax.coastlines()
ax.add_feature(cf.LAND)
ax.add_feature(cf.OCEAN)
gl = ax.gridlines()
gl.ylabels_left = True
gl.xlabels_bottom = True

# Plot Fronts (uses circle axes from ARGO Float Track)
#ax.set_boundary(circle, transform=ax.transAxes)
plt.plot(stf_lon, stf_lat, color='Red', transform=ccrs.PlateCarree(), 
         label='Subtropical Front')
plt.plot(saf_lon, saf_lat, color='Orange', transform=ccrs.PlateCarree(), 
         label='Subantarctic Front')
plt.plot(pf_lon, pf_lat, color='Yellow', transform=ccrs.PlateCarree(), 
         label='Polar Front')
#plt.plot(saccf_lon, saccf_lat, color='Green', transform=ccrs.PlateCarree(), 
         #label='Southern ACC Front')
#plt.plot(sbdy_lon, sbdy_lat, color='Blue', transform=ccrs.PlateCarree(), 
        # label='Southern Boundary')
#plt.legend()

# Plot ARGO Float Track (taken from ARGO Float Track)
plt.scatter(DF_fg['Lon [°E]'], DF_fg['Lat [°N]'], 
            color='Black', transform=ccrs.PlateCarree(), s = 5, zorder = 1000) # change dot size
plt.title('Map of Floats Matching NES within Front Group '+ front_group_str)
plt.savefig(output_path +'NES_Map_FG' + front_group_str + '.png')







# ------------------------ Organizing SOCCOM Fleet Data along same pressure -------------


#Create New Pressue Array
pnew = np.arange(25,2000,1) #this will make an array with pressures every 1 db from 0 to 2000 db, you could change the increment to e.g. 10

#EAS

# Create New Variable Arrays
tnew_eas = np.empty((len(EAS),len(pnew)))
snew_eas = np.empty((len(EAS),len(pnew)))
onew_eas = np.empty((len(EAS),len(pnew)))
nnew_eas = np.empty((len(EAS),len(pnew)))
dnew_eas = np.empty((len(EAS),len(pnew)))


for n in range(len(EAS)): #pandas values
    cruise_n = EAS['Cruise'].values[n]
    station_n = EAS['Station'].values[n]
    SOCCOM_prof = SOCCOM.loc[(SOCCOM['Cruise']==cruise_n) & (SOCCOM['Station']==station_n)]
    if np.all(np.isnan(SOCCOM_prof['Pressure[dbar]'])):
        continue
    if np.nanmin(SOCCOM_prof['Pressure[dbar]'])>= pnew[0]:
        continue
#    print('I made it to where I will look at all the EAS data and interpolate it to the same pressure')
    ftemp = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Temperature[°C]']) #, bounds_error = False, fill_value = 0)
    tnew_eas[n,:] = ftemp(pnew)
    
    fsal = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Salinity[pss]']) #, bounds_error = False, fill_value = 0)
    snew_eas[n,:] = fsal(pnew)
    
    fox = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Oxygen[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    onew_eas[n,:] = fox(pnew)
    
    fnit = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['Nitrate[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    nnew_eas[n,:] = fnit(pnew)

    fdic = interp.interp1d(SOCCOM_prof['Pressure[dbar]'],SOCCOM_prof['DIC_LIAR[µmol/kg]']) #, bounds_error = False, fill_value = 0)
    dnew_eas[n,:] = fdic(pnew)
    

### Next Steps (12/07/22)

# Re-do EAS pressure level interpolations to reflect profiles within the same front group
# Example from Lab Code with Chambers:

# ------ Front Group Arrays
#front_group = 1
#front_group_str = str(front_group)

# Create New Variable Arrays
#tnew_aw = np.empty((len(eddy_ID_anti_winter.loc[front_group_anti_winter == front_group]),len(pnew)))
#snew_aw = np.empty((len(eddy_ID_anti_winter.loc[front_group_anti_winter == front_group]),len(pnew)))
#onew_aw = np.empty((len(eddy_ID_anti_winter.loc[front_group_anti_winter == front_group]),len(pnew)))
#nnew_aw = np.empty((len(eddy_ID_anti_winter.loc[front_group_anti_winter == front_group]),len(pnew)))
#dnew_aw = np.empty((len(eddy_ID_anti_winter.loc[front_group_anti_winter == front_group]),len(pnew)))

#for n in range(len(eddy_ID_anti_winter.loc[front_group_anti_winter == front_group])): #pandas values
#    cruise_n = eddy_ID_anti_winter['Cruise'].loc[front_group_anti_winter == front_group].values[n]
#    station_n = eddy_ID_anti_winter['Station'].loc[front_group_anti_winter == front_group].values[n]
#    SOCCOM_prof = SOCCOM.loc[(SOCCOM['Cruise']==cruise_n) & (SOCCOM['Station']==station_n)]
#    #print(np.nanmin(SOCCOM_prof['Pressure[dbar]']))
#    if np.all(np.isnan(SOCCOM_prof['Pressure[dbar]'])):
#        continue
#    if np.nanmin(SOCCOM_prof['Pressure[dbar]'])>= pnew[0]:
#        continue
    

# Rpeat process for ECS and NES - set it up as a function using eddytype/non_eddy and 
#   front group as input variables
# Plot new variable profiles (can include as outputs of function mentioned above)

# Anomalies - Replicate anomalies calculated within lab code for Dr. Chambers (as function?)
# Obtain the mean/std from non_eddy profiles within a frontal group
# Substruct mean non_eddy profile from anti/cyc eddy profiles within same frontal group
# Examine anti/cyc profiles relative to bounds of non_eddy stds

# Replicate with single float organizing of fleet data (by eddy_type/non_eddy, frontal group)
# Run through eddy_matchup in new file (see code for Nancy - ArgoFloatPandasJan2021)
# Repeat data interpolation onto same pressure level
# Obtain anomalies and plots from within this float ALONE and compare eddies/non_eddies
# Compare eddies/non_eddies to entire fleet within same frontal group




