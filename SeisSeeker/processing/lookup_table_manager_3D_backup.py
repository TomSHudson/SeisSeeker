#!/Users/eart0504/opt/anaconda3/bin/python
#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Script to calculate travel times lookup tables for various seismic phases.

# Input variables:

# Output variables:

# Created by Tom Hudson, 17th August 2022

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import skfmm 

#----------------------------------------------- Define constants and parameters -----------------------------------------------

#----------------------------------------------- End: Define constants and parameters -----------------------------------------------


#----------------------------------------------- Define main functions -----------------------------------------------
def read_1D_vel_model_for_fmm(oneD_vel_model_z_df, extent_xy_m=[3000,3000], dxyz=[1.,1.,1.]):
    """Function to create specific ice velocity model.
    Inputs:
    oneD_vel_model_z_df - Pandas DataFrame of shape (z_extent, 3) with the columns corresponding to: depth, vp, vs
    extent_xy_m - The extent of the velocity model in xy
    Returns:
    vel_model_arr_P - 3D Velocity grid for P wave
    vel_model_arr_S - 3D Velocity grid for S wave
    ix, iy, iz - Grids containing index labels
    """
    dx = dxyz[0] # Grid spacing in x dir
    dy = dxyz[1] # Grid spacing in x dir
    dz = dxyz[2] # Grid spacing in z dir
    vel_model_x_labels = np.arange(0., extent_xy_m[0]+dx, dx)
    vel_model_y_labels = np.arange(0., extent_xy_m[1]+dy, dy)
    vel_model_z_labels = oneD_vel_model_z_df['depth'].values
    # Create the index grids:
    iy, ix, iz = np.meshgrid(vel_model_y_labels, vel_model_x_labels, vel_model_z_labels)
    # Create the P and S wave model:
    vel_model_arr_P = np.zeros(np.shape(ix), dtype=float)
    vel_model_arr_S = np.zeros(np.shape(ix), dtype=float)
    for i in range(vel_model_arr_P.shape[0]):
        for j in range(vel_model_arr_P.shape[1]):
            vel_model_arr_P[i,j,:] = oneD_vel_model_z_df['vp']
            vel_model_arr_S[i,j,:] = oneD_vel_model_z_df['vs']
    return vel_model_arr_P, vel_model_arr_S, vel_model_x_labels, vel_model_y_labels, vel_model_z_labels

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def create_LUT(oneD_vel_model_z_df, array_centre_xyz, extent_xy_m=[3000.,3000.], dxyz=[1.,1.,1.]):
    """Get theoretical P and S travel times for all nodes in a lookup table, relative to the centre of the array 
    (using Eikonal method).
    Inputs:
    oneD_vel_model_z_df - Pandas DataFrame of shape (z_extent, 3) with the columns corresponding to: depth, vp, vs.
    station_xyz_coords_df - Pandas DataFrame containing stations with x_m, y_m, z_m coordinates.
    src_depth_xyz - The source depth in x,y,z in metres.
    """
    # Get velocity model:
    dx = dxyz[0] # Grid spacing in x dir
    dy = dxyz[1] # Grid spacing in x dir
    dz = dxyz[2] # Grid spacing in z dir
    vel_model_arr_P, vel_model_arr_S, vel_model_x_labels, vel_model_y_labels, vel_model_z_labels = read_1D_vel_model_for_fmm(oneD_vel_model_z_df, 
                                                                                                                            extent_xy_m=extent_xy_m, 
                                                                                                                            dxyz=dxyz)
    
    # Calculate travel times for all points in the grid:
    # Setup array for masking/showing where array centre is is:
    phi = -np.ones(vel_model_arr_P.shape)
    val, x_idx = find_nearest(vel_model_x_labels, array_centre_xyz[0])
    val, y_idx = find_nearest(vel_model_y_labels, array_centre_xyz[1])
    val, z_idx = find_nearest(vel_model_z_labels, array_centre_xyz[2])
    # Set array_centre_xyz location:
    phi[x_idx, y_idx, z_idx] = 1. 

    # Calculate travel times array to all the points in the grid:
    trav_times_grid_P = skfmm.travel_time(phi,vel_model_arr_P,dx=dxyz)
    trav_times_grid_S = skfmm.travel_time(phi,vel_model_arr_S,dx=dxyz)
    
    return trav_times_grid_P, trav_times_grid_S






#----------------------------------------------- End: Define main functions -----------------------------------------------


#----------------------------------------------- Run script -----------------------------------------------
if __name__ == "__main__":
    # Example of usage:
    # (For a 2200 x 1 x 2100 m grid with receivers in a line from the surface to 2 km then along horizontally for 1 km)
    # Create velocity model:
    oneD_vel_model_z_df = pd.DataFrame({'depth': np.arange(0,3000,1.), 'vp': 3500*np.ones(2100), 'vs': 2000*np.ones(2100)})
    station_xs = np.concatenate((1100*np.ones(len(np.arange(1,2000))), np.arange(100,1100)[::-1]))
    station_ys = np.zeros(len(station_xs))
    station_zs = np.concatenate((np.arange(1,2000), 2000*np.ones(1000)))
    station_xyz_coords_df = pd.DataFrame({'x_m': station_xs, 'y_m': station_ys, 'z_m': station_zs})
    src_xyz = [600., 0., 1500.]
    # And get travel times:
    trav_times_P, trav_times_S = calc_travel_times_fmm(oneD_vel_model_z_df, station_xyz_coords_df, src_xyz, extent_xy_m=[2200.,1.])

    # And plot result:
    plt.figure()
    plt.scatter(np.arange(len(trav_times_P)), trav_times_P)
    plt.scatter(np.arange(len(trav_times_S)), trav_times_S)
    plt.show()

    print("Finished")


















    
    









