#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 13:35:38 2021

@author: nicolausf
"""

#import different modules

import re
import datetime
import shapely
#import cmocean
#import cmocean.cm as cmo
#os.environ["PROJ_LIB"] = "C:\\Utilities\\Python\\Anaconda\\Library\\share";
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.path as mpath
#from numpy import linspace
#from numpy import meshgrid
#import cartopy.geodesic 
import cartopy.crs as ccrs  # chose to use cartopy over basemap due to data format
import cartopy.feature as cf
#import xarray as xr






def BGC_plot(pd_Float_Dataframe, var, X_axis_variable, **kwargs):
#def BGC_plot(pd_Float_Dataframe, var, *args, **kwargs):
    """
    Inputs:
    pd_Float_Dataframe: required (data type***)- The name of the pandas data frame that you are 
        going to be pulling float data from
    var: required (data type***)- Name of the variable to create biogeochemical plot of.  
        Name must match the format that it appears in withint the Dataframes 
        selected as the pd_Float_Dataframe, and must include '' around it
    X_axis_variable: required (data type***)- Name of the variable you would like to use
        as the x-axis.  Examples include 'Station' or 'date'.  Must match a variable name
        exactly as seend in the pd_Float_Dataframe and must include '' around it
    
    
    station_min: optional (data type***) - value of the lowest station you want to observe (to set x limits)
    station_max: optional (data type***) - value of the highest station you want to observe (to set y limits)
    depth_max: optional (data type***)- maximum depth below 0, serves as bottom boundary of y-axis
    depth_min: optional (data type***)- minimum depth below 0, serves as top boundary of y-axis
    cmap: optional (string) -  needs '' around it
    clim_min: optional (data type***)- colorbar minimum limit
    clim_max: optional (data type***)- colorbar maximum limit
    MLD: optional (BOOL) - decide whether or not you want to include a line representing the mixed layer
        depth on your plot.
            
    """
    
    
### Mixed Layer Depth
## Process by which we determine this mixed layer depth is being discussed **********
    MLDgrad = []
    diurnal = 20

    for station in pd_Float_Dataframe['Station'].unique():
        pd_Float_Dataframe['DepSig'] = (
            pd_Float_Dataframe['Sigma_theta[kg/m^3]'].loc[(pd_Float_Dataframe['Station'] == station) & (
                pd_Float_Dataframe['Depth[m]'] > diurnal)].diff()
            )/(
            pd_Float_Dataframe['Depth[m]'].loc[(pd_Float_Dataframe['Station']==station) & (
                pd_Float_Dataframe['Depth[m]'] > diurnal)].diff()
            )
        MLDgrad.append([
            station,
            pd_Float_Dataframe['date'].loc[(pd_Float_Dataframe['Station']==station) & (
                pd_Float_Dataframe['Depth[m]'] > diurnal) & (
                (pd_Float_Dataframe['DepSig'])==(pd_Float_Dataframe['DepSig'].max()))].min(),
            pd_Float_Dataframe['Depth[m]'].loc[(pd_Float_Dataframe['Station']==station) & (
                pd_Float_Dataframe['Depth[m]'] > diurnal) & (
                (pd_Float_Dataframe['DepSig'])==(pd_Float_Dataframe['DepSig'].max()))].min(),
            pd_Float_Dataframe['Lon [°E]'].loc[(pd_Float_Dataframe['Station']==station) & (
                pd_Float_Dataframe['Depth[m]'] > diurnal)].mean(),
            pd_Float_Dataframe['Lat [°N]'].loc[(pd_Float_Dataframe['Station']==station) & (
                pd_Float_Dataframe['Depth[m]'] > diurnal)].mean()
            ])


    MLDgrad = pd.DataFrame(data=MLDgrad, columns=['Station', 'date', 'MLD','Lon [°E]','Lat [°N]'])  



### Creating BGC Plot
    BGC = pd_Float_Dataframe[pd.notna(pd_Float_Dataframe[var])]
    fig = plt.figure(num=None, figsize=(8,3), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_axes([0.1, 0.1, .8, .8])
    sc = ax.tricontourf(BGC[X_axis_variable], BGC['Depth[m]'], BGC[var], levels = 16)
# Potential help to creating a changing color bar in tricontourf
#https://stackoverflow.com/questions/21952100/setting-the-limits-on-a-colorbar-of-a-contour-plot

    

### Invert y axis
    ax.invert_yaxis()
    #plt.gca().invert_yaxis()



### Colorbar Settings
    cb=plt.colorbar(sc)
    cb.set_label(var) 



### Titles and Labels on Plot
    ax.set_title(var)
    ax.set_ylabel('Depth [m]')
    ax.set_xlabel(X_axis_variable)



### Establishing kwargs

    for key, value in kwargs.items():
        if re.match('MLD', key):
            if value == False:
                continue
            if value == True:
                ax.plot(MLDgrad['Station'], MLDgrad['MLD'], c='red', label='Mixed Layer Depth')
        if re.match('clim_min', key):
            #cb.set_clim(vmin = value)
            #plt.clim(vmin = value)
            #cb.set_lim(vmin = value)
            sc.set_clim(vmin = value)
            #sc = ax.tricontourf(vmin = value)
            #sc.set_under(value, 'r')
        if re.match('clim_max', key):
            #cb.set_clim(vmax = value)
            #plt.clim(vmax = value)            
            #cb.set_lim(vmin = value)
            sc.set_clim(vmax = value)
            #sc = ax.tricontourf(vmax = value)
            #sc.set_over(value, 'r')
        if re.match('map', key):
            if value == True:
                #plt.axes((left, bottom, width, height), facecolor='w')
                ax2 = plt.axes((.4, .1, .3, .5), facecolor='w', projection=ccrs.PlateCarree(central_longitude=180))
                ax2.set_extent([120,300,-80,-30], ccrs.PlateCarree())
                ax2.coastlines()
                ax2.add_feature(cf.LAND)
                ax2.add_feature(cf.OCEAN)
                gl = ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
                gl.top_labels = False
                ## Plot Argo Float Lon and Lat Data
                plt.plot(BGC['Lon [°E]'], BGC['Lat [°N]'], transform=ccrs.PlateCarree(), marker='o', markerfacecolor='yellow', markevery=[0], zorder=200)
                plt.plot(BGC['Lon [°E]'], BGC['Lat [°N]'], transform=ccrs.PlateCarree(), marker='o', markerfacecolor='red', markevery=[-1], zorder=100)
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
                ## Plot Fronts
                plt.plot(stf['lon'], stf['lat'], color='Red', transform=ccrs.PlateCarree(), linewidth=1, zorder=50)
                plt.plot(saf['lon'], saf['lat'], color='Orange', transform=ccrs.PlateCarree(), linewidth=1, zorder=60)
                plt.plot(pf['lon'], pf['lat'], color='Yellow', transform=ccrs.PlateCarree())
                plt.plot(saccf['lon'], saccf['lat'], color='Green', transform=ccrs.PlateCarree())
                plt.plot(sbdy['lon'], sbdy['lat'], color='Blue', transform=ccrs.PlateCarree())
                #ax2.set_title(var)
                #ax2.set_ylabel('Longitude')
                #ax2.set_xlabel('Latitude')
        if re.match('station_min', key):
            ax.set_xlim(left = value)
        if re.match('station_max', key):
            ax.set_xlim(right = value)
        if re.match('depth_max', key):
            ax.set_ylim(bottom = value)
        if re.match('depth_min', key):
            ax.set_ylim(top = value)  
        if re.match('cmap', key):
            sc.set_cmap(value)
        if re.match('eddy_station', key):
            if value == True:
                plt.axvline(97, color = 'black')
                plt.axvline(107, color = 'black')
        if re.match('legend', key):
            if value == True:
                plt.legend()
   
 


  
### Legend
    #plt.legend()
    

    

    return plt.show()
