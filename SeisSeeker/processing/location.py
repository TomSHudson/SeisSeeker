#!/usr/bin/python
#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Script to perform earthquake location using array processing methods.

# Created by Tom Hudson, 10th August 2022

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import os, sys
import obspy
from scipy.signal import find_peaks
from numba import jit, objmode, prange, set_num_threads
import gc 
# import multiprocessing as mp
import time 



#----------------------------------------------- Define main functions -----------------------------------------------
class CustomError(Exception):
    pass



def locate_events_from_P_and_S_array_arrivals(events_df, LUTs_dict, array_latlon, receiver_vp, receiver_vs, verbosity=0):
    """Function to locate events from P and S phase arrivals."""
    # Append rows to events_df to save locations:
    events_df["x_km"] = ""
    events_df["y_km"] = ""
    events_df["z_km"] = ""
    events_df["lat"] = ""
    events_df["lon"] = ""

    # Calculate various objects only once, for efficiency:
    delta_tp_ts_grid = LUTs_dict['trav_times_grid_S'] - LUTs_dict['trav_times_grid_P']

    # Loop over events, processing to find locations:
    count=0
    for index, row in events_df.iterrows():
        print("Locating event", count+1, '/', len(events_df['bazi1']))
        # 1. Find effective radius within LUT grid by minimising:
        # delta_tp_ts - (ts_tt - tp_tt)
        # To create PDF of uncertainty in grids of LUT for location
        # Calc. delta_tp_ts:
        delta_tp_ts = obspy.UTCDateTime(row['t2']) - obspy.UTCDateTime(row['t1'])
        # Find minimum:
        abs_min_arr = np.abs(delta_tp_ts - delta_tp_ts_grid)
        tp_ts_res_pdf = 1 - (abs_min_arr / np.max(abs_min_arr))

        # 2. Use inclination angle from slowness (of P and/or S) to search from LUT for possible cells:
        if row['slow1'] > 0 and row['slow2'] > 0:
            # To create PDF for location of cells vertically in LUT
            # 2.i. For P:
            # Calculate inclination angle:
            # v = v_app * sin(inc_angle_from_vert) ?!
            # Therefore, theta = arcsin( v / v_app ) ?!
            v_app_P = 1. / row['slow1']
            inc_angle_P = np.rad2deg(np.arcsin( v_app_P / receiver_vp ))
            # And calculate inc. angle pdf:
            abs_min_arr = np.abs(LUTs_dict['theta_grid_P'] - inc_angle_P)
            P_inc_angle_res_pdf = 1 - (abs_min_arr / np.max(abs_min_arr))
            # 2.ii. For S:
            # Calculate inclination angle:
            # v = v_app * sin(inc_angle_from_vert) ?!
            # Therefore, theta = arcsin( v / v_app ) ?!
            v_app_S = 1. / row['slow2']
            inc_angle_S = np.rad2deg(np.arcsin( v_app_S / receiver_vs ))
            # And calculate inc. angle pdf:
            abs_min_arr = np.abs(LUTs_dict['theta_grid_S'] - inc_angle_S)
            S_inc_angle_res_pdf = 1 - (abs_min_arr / np.max(abs_min_arr))
            # 2.iii. Stack P and S inc. angle PDFs:
            PS_stack_inc_angle_res_pdf = (P_inc_angle_res_pdf + S_inc_angle_res_pdf) / 2.

            # 3. Combine radius PDF and inc-angle PDF to get best result location within 2D LUT:
            # Stack pdfs:
            stacked_pdf = (tp_ts_res_pdf + PS_stack_inc_angle_res_pdf) / 2.
            # And get best result:
            max_xz_idxs = np.argwhere(stacked_pdf == np.max(stacked_pdf))
            x_idx_curr = max_xz_idxs[0][0]
            z_idx_curr = max_xz_idxs[0][1]
            event_x_coord_km = LUTs_dict['vel_model_x_labels'][x_idx_curr]
            event_z_coord_km = LUTs_dict['vel_model_z_labels'][z_idx_curr]
        else:
            event_x_coord_km = np.nan
            event_z_coord_km = np.nan

        # Plot workings, if specified:
        if verbosity > 1:
            fig, axes = plt.subplots(nrows=4, figsize=(4,16))
            Z, X = np.meshgrid(LUTs_dict['vel_model_z_labels'], LUTs_dict['vel_model_x_labels'])
            # Plot P-S pdf:
            im = axes[0].pcolormesh(X, Z, tp_ts_res_pdf, cmap="Greys_r", vmin=0, vmax=1)
            plt.colorbar(im, ax=axes[0], label="PDF, $t_{P-S}$")
            # for i in range(len(min_xz_idxs)):
            #     x_idx_curr = min_xz_idxs[i][0]
            #     z_idx_curr = min_xz_idxs[i][1]
            #     axes[0].scatter(LUTs_dict['vel_model_x_labels'][x_idx_curr], LUTs_dict['vel_model_z_labels'][z_idx_curr], c='k')
            # And plot P inc. angle pdf:
            im2 = axes[1].pcolormesh(X, Z, P_inc_angle_res_pdf, cmap="Greys_r", vmin=0, vmax=1)
            plt.colorbar(im2, ax=axes[1], label="PDF, $\\theta_P$")
            # And plot S inc. angle pdf:
            im3 = axes[2].pcolormesh(X, Z, S_inc_angle_res_pdf, cmap="Greys_r", vmin=0, vmax=1)
            plt.colorbar(im3, ax=axes[2], label="PDF, $\\theta_S$")
            # And plot stacked pdf:
            im4 = axes[3].pcolormesh(X, Z, stacked_pdf, cmap="Greys_r", vmin=0, vmax=1)
            plt.colorbar(im4, ax=axes[3], label="stacked PDF")
            # And plot minimum idx on stacked pdf:
            axes[3].scatter(event_x_coord_km, event_z_coord_km, c='g')
            for i in range(len(axes)):
                axes[i].invert_yaxis()
                axes[i].set_xlabel("X (m)")
                axes[i].set_ylabel("Z (m)")
            plt.show()

        # 4. Convert 2D LUT result into 3D cartesian location by combining with bazi.:
        # (and append to events_df)
        r_hor_km = event_x_coord_km / 1000
        mean_bazi = (row['bazi1'] + row['bazi2']) / 2
        events_df.iloc[count, events_df.columns.get_loc('x_km')] = r_hor_km * np.sin( np.deg2rad( mean_bazi ) )
        events_df.iloc[count, events_df.columns.get_loc('y_km')] = r_hor_km * np.cos( np.deg2rad( mean_bazi ) )
        events_df.iloc[count, events_df.columns.get_loc('z_km')] = event_z_coord_km / 1000
        # Calculate event lat and lons relative to array centre:
        events_df.iloc[count, events_df.columns.get_loc('lat')] = array_latlon[0] + obspy.geodetics.base.kilometers2degrees(events_df.iloc[count, events_df.columns.get_loc('y_km')])
        events_df.iloc[count, events_df.columns.get_loc('lon')] = array_latlon[1] + obspy.geodetics.base.kilometers2degrees(events_df.iloc[count, events_df.columns.get_loc('x_km')])

        # Update event count:
        count+=1

    return events_df









#----------------------------------------------- End: Define main functions -----------------------------------------------



