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
import skfmm # For fast-marching travel-time lookup tables
# import ttcrpy.rgrid as ttcrpy_rgrid # For ray-tracing based incidence angles
import gc 
import pykonal # For ray-tracing based incidence angles

#----------------------------------------------- Define constants and parameters -----------------------------------------------

#----------------------------------------------- End: Define constants and parameters -----------------------------------------------


#----------------------------------------------- Define main functions -----------------------------------------------
def read_1D_vel_model_to_3D_model_for_fmm(oneD_vel_model_z_df, extent_xy_m=[3000,3000], dxyz=[1.,1.,1.]):
    """Function to create specific velocity model.
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


def vec_norm(x):
    """Function to calculate norm of a vector"""
    return np.sqrt(x.dot(x))


def read_1D_vel_model_to_2D_model_for_fmm(oneD_vel_model_z_df, extent_x_m=3000, dxz=[1.,1.]):
    """Function to create specific velocity model in 2D (x, z).
    Inputs:
    oneD_vel_model_z_df - Pandas DataFrame of shape (z_extent, 3) with the columns corresponding to: depth, vp, vs
    extent_xy_m - The extent of the velocity model in xy
    Returns:
    vel_model_arr_P - 3D Velocity grid for P wave
    vel_model_arr_S - 3D Velocity grid for S wave
    """
    dx = dxz[0] # Grid spacing in x dir
    dz = dxz[1] # Grid spacing in z dir
    vel_model_x_labels = np.arange(0., extent_x_m+dx, dx)
    vel_model_z_labels = oneD_vel_model_z_df['depth'].values
    # Create the index grids:
    iz, ix = np.meshgrid(vel_model_z_labels, vel_model_x_labels)
    # Create the P and S wave model:
    vel_model_arr_P = np.zeros(np.shape(ix), dtype=float)
    vel_model_arr_S = np.zeros(np.shape(iz), dtype=float)
    for i in range(vel_model_arr_P.shape[0]):
        vel_model_arr_P[i,:] = oneD_vel_model_z_df['vp']
        vel_model_arr_S[i,:] = oneD_vel_model_z_df['vs']
    return vel_model_arr_P, vel_model_arr_S, vel_model_x_labels, vel_model_z_labels


def ray_tracer(v_model, dxyz, src_loc=[0,0,0], rx_loc=[0,0,0]):
    """Function to perform ray-tracing, using pykonal package.
    All units are in km. src_loc, rx_loc are in km x,y,z."""
    # Define the solver:
    solver = pykonal.solver.PointSourceSolver(coord_sys="cartesian") 
    # Define the computational domain:
    solver.vv.min_coords = 0, 0, 0
    solver.vv.node_intervals = dxyz[0], dxyz[1], dxyz[2]
    solver.velocity.npts = v_model.shape[0], v_model.shape[1], v_model.shape[2]
    solver.vv.values = v_model
    # Define the source location:
    solver.src_loc = np.array(src_loc, dtype=float)
    # Solve the eikonal equation for the travel-time field:
    solver.solve()
    # And perform the ray tracing:
    ray = solver.trace_ray(np.array(rx_loc, dtype=float))
    # And tidy:
    del solver
    gc.collect()
    return ray


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def create_2D_LUT(oneD_vel_model_z_df, array_centre_xz, extent_x_m=3000., dxz=[1.,1.], n_threads=1):
    """Get theoretical P and S travel times for all nodes in a lookup table, relative to the centre of the array 
    (using Eikonal method).
    Inputs:
    oneD_vel_model_z_df - Pandas DataFrame of shape (z_extent, 3) with the columns corresponding to: depth, vp, vs.
    array_centre_xz - The location of the array centre within the proposed lookup table grid, in metres.
    """
    # Get 2D velocity model:
    dx = dxz[0] # Grid spacing in x dir
    dz = dxz[1] # Grid spacing in z dir
    vel_model_arr_P, vel_model_arr_S, vel_model_x_labels, vel_model_z_labels = read_1D_vel_model_to_2D_model_for_fmm(oneD_vel_model_z_df, 
                                                                                                                            extent_x_m=extent_x_m, 
                                                                                                                             dxz=dxz)

    # Create travel-time lookup table grids:
    # (from fast marching method)
    # Calculate travel times for all points in the grid:
    # Setup array for masking/showing where array centre is is:
    phi = -np.ones(vel_model_arr_P.shape)
    val, x_idx = find_nearest(vel_model_x_labels, array_centre_xz[0])
    val, z_idx = find_nearest(vel_model_z_labels, array_centre_xz[1])
    # Set array_centre_xyz location:
    phi[x_idx, z_idx] = 1. 
    # Calculate travel times array to all the points in the grid:
    trav_times_grid_P = skfmm.travel_time(phi,vel_model_arr_P,dx=dxz)
    trav_times_grid_S = skfmm.travel_time(phi,vel_model_arr_S,dx=dxz)

    # And create incidence angle lookup table grids:
    # (from ray tracing)
    # Create grid:
    # x_node_labels = vel_model_x_labels
    # z_node_labels = vel_model_z_labels
    # rgrid = ttcrpy_rgrid.Grid2d(x_node_labels, z_node_labels, cell_slowness=False, n_threads=n_threads)
    # Specify velocity model for ray tracing:
    vel_model_arr_P_3D = vel_model_arr_P.reshape(1, vel_model_arr_P.shape[0], vel_model_arr_P.shape[1])
    vel_model_arr_S_3D = vel_model_arr_S.reshape(1, vel_model_arr_S.shape[0], vel_model_arr_S.shape[1])
    # Calculate rays for each point in the grid:
    theta_grid_P = np.zeros(trav_times_grid_P.shape)
    theta_grid_S = np.zeros(trav_times_grid_P.shape)
    # Loop over x:
    count = 0
    for i in range(theta_grid_P.shape[0]):
        # Loop over Z:
        for j in range(theta_grid_P.shape[1]):
            # Print progress:
            if count % 100 == 0:
                print("Processing for ray", count, "/", theta_grid_P.shape[0]*theta_grid_P.shape[1])
            count += 1

            # Calculate ray-tracing and incidence angle for P wave:
            # Calculate rays for current event:
            # (Note that coords are converted into 3D and km)
            node_coords = np.array([0, vel_model_x_labels[i], vel_model_z_labels[j]], dtype=float) / 1000
            node_coords = node_coords + 0.001 # Add 1 metre, so that node coords are non-zero
            receiver_coords = np.array([0, array_centre_xz[0], array_centre_xz[1]], dtype=float) / 1000
            # tt, rays = rgrid.raytrace(event_coords, receiver_coords, 1./vel_model_arr_P, return_rays=True)
            # Perform ray tracing:
            dxyz = np.array([dx, dx, dz], dtype=float) / 1000
            ray = ray_tracer(vel_model_arr_P_3D / 1000, dxyz, src_loc=node_coords, rx_loc=receiver_coords)
            # Calculate incidence angle at array:
            ray_vec_at_receiver = - np.array([ ray[-1,1] - ray[-2,1], ray[-1,2] - ray[-2,2] ]) # Note minus sign, as defining as vector out from array
            vert_vec = np.array([ 0,  1])
            theta_curr = np.arccos( ( ray_vec_at_receiver.dot(vert_vec) ) / ( vec_norm(ray_vec_at_receiver) * vec_norm(vert_vec) ) ) # cos(theta) = a.b / |a| |b|
            theta_curr = np.rad2deg(theta_curr)
            theta_grid_P[i,j] = theta_curr
            # And clear up:
            del ray 
            gc.collect()

            # Calculate ray-tracing and incidence angle for S wave:
            # Calculate rays for current event:
            # (Uses some parameters as for P wave)
            # Perform ray tracing:
            ray = ray_tracer(vel_model_arr_S_3D / 1000, dxyz, src_loc=node_coords, rx_loc=receiver_coords)
            # Calculate incidence angle at array:
            ray_vec_at_receiver = - np.array([ ray[-1,1] - ray[-2,1], ray[-1,2] - ray[-2,2] ]) # Note minus sign, as defining as vector out from array
            vert_vec = np.array([ 0,  1])
            theta_curr = np.arccos( ( ray_vec_at_receiver.dot(vert_vec) ) / ( vec_norm(ray_vec_at_receiver) * vec_norm(vert_vec) ) ) # cos(theta) = a.b / |a| |b|
            theta_curr = np.rad2deg(theta_curr)
            theta_grid_S[i,j] = theta_curr
            # And clear up:
            del ray 
            gc.collect()


            
    
    return trav_times_grid_P, trav_times_grid_S, theta_grid_P, theta_grid_S, vel_model_x_labels, vel_model_z_labels






#----------------------------------------------- End: Define main functions -----------------------------------------------



















    
    









