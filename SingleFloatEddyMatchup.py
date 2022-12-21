#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 10:02:34 2022

@author: nicolausf
"""

# ************************** Load in Data **************************

import pandas as pd
import eddy_matchup

#------------------- Import Single Argo Data with Eddy ---------------------
#  Note:  This is not the only location in the code where data is imported
##### Import ARGO Float File #####

# Currently set up to pull data from a locally downloaded source
# In this code, all current files are being downloaded from the local desktop of 
# the laptop

# Location where you want figures to save:
output_path = 'generated2/'

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

# Make proper date format
ARGO['date'] = pd.to_datetime(ARGO['mon/day/yr'] +' '+ ARGO['hh:mm'])


# ------------------------ Matchup of All Eddies Encountered by Float X -------------

#ARGO_Subset = ARGO.loc[(ARGO['Station'] >= 97) & (ARGO['Station'] <= 121)]


eddy = eddy_matchup.match(
    ARGO['Lon [°E]'],
    ARGO['Lat [°N]'],
    ARGO['date'],
    latmin=ARGO['Lat [°N]'].min(),
    latmax=ARGO['Lat [°N]'].max(),
    radiusrange=1.5) # Was 1.2 Change threshold value on this **************

# Store eddy data
ARGO['eddy_type'] = eddy.eddy_type
ARGO['eddy_ID'] = eddy.eddy_ID
ARGO['eddy_lat'] = eddy.eddy_lat
ARGO['eddy_lon'] = eddy.eddy_lon
ARGO['eddy_time'] = eddy.eddy_time
ARGO['eddy_amplitude'] = eddy.eddy_amplitude
ARGO['eddy_vmax'] = eddy.eddy_vmax
ARGO['eddy_rad_to_vmax'] = eddy.eddy_rad_to_vmax
ARGO['eddy_age'] = eddy.eddy_age
ARGO['eddy_dist_to_ctr'] = eddy.eddy_dist_to_ctr


ARGO.to_pickle('data/ARGO.pkl')
