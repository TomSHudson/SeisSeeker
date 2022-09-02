#!/usr/bin/python
#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Script to perform earthquake location using array processing methods.

# Created by Tom Hudson, 10th August 2022

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import pandas as pd
import numpy as np
import matplotlib
import os, sys
import obspy
from scipy.signal import find_peaks
from numba import jit, objmode, prange, set_num_threads
import gc 
# import multiprocessing as mp
import time 
from SeisSeeker.processing import lookup_table_manager 



#----------------------------------------------- Define main functions -----------------------------------------------
class CustomError(Exception):
    pass



def locate_events_from_P_and_S_array_arrivals(LUT_dict):
    """Function to locate events from P and S phase arrivals."""

    # 0. Create 2D LUTs:
    # (for P and S travel-times, and inclination angle)
    dx = 100.
    depth_range_m = 3000
    oneD_vel_model_z_df = pd.DataFrame({'depth': np.arange(0,3000,dx), 'vp': 3500*np.ones(int(3000/dx)), 'vs': 2000*np.ones(int(3000/dx))})
    array_centre_xz = [2000, 0]
    extent_x_m = 4000
    # And get travel times:
    trav_times_grid_P, trav_times_grid_S, theta_grid_P, theta_grid_S, vel_model_x_labels, vel_model_z_labels = lookup_table_manager.create_2D_LUT(oneD_vel_model_z_df, array_centre_xz, extent_x_m=extent_x_m, dxz=dxz)



    # 1. Find effective radius within LUT grid by minimising:
    # delta_tp_ts - (tp_tt - ts_tt)
    # To create PDF of uncertainty in grids of LUT for location

    # 2. Use inclination angle from slowness (of P and/or S) to search from LUT for possible cells:
    # To create PDF for location of cells vertically in LUT
    # v = v_app * sin(inc_angle_from_vert)
    # Therefore, theta = arcsin( v / v_app )

    # 3. Combine radius PDF and inc-angle PDF to get best result location within 2D LUT

    # 4. Convert 2D LUT result into 3D cartesian location by combining with bazi.







#----------------------------------------------- End: Define main functions -----------------------------------------------



