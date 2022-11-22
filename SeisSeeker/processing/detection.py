#!/usr/bin/python
#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Script to perform earthquake detection using array processing methods.

# Created by Tom Hudson, 10th August 2022

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import os, sys
import obspy
from scipy.signal import find_peaks
from numba import jit, objmode, prange, set_num_threads
import gc 
# import multiprocessing as mp
import time 
import glob 
import pickle
from SeisSeeker.processing import lookup_table_manager, location 



#----------------------------------------------- Define main functions -----------------------------------------------
class CustomError(Exception):
    pass


def flatten_list(l):
    return [item for sublist in l for item in sublist]


@jit(nopython=True, parallel=True)#, nogil=True)
def _fast_freq_domain_array_proc(data, max_sl, fs, target_freqs, xx, yy, n_stations, n_t_samp, remove_autocorr):
    """Function to perform array processing fast due to being designed to 
    be wrapped using Numba.
    Returns:
    Pfreq_all
    """
    # Define grid of slownesses:
    # number of pixes in x and y
    # (Determines number of phase shifts to perform)
    nux = 51 #101
    nuy = 51 #101
    ux = np.linspace(-max_sl,max_sl,nux)
    uy = np.linspace(-max_sl,max_sl,nuy)
    dux=ux[1]-ux[0]
    duy=uy[1]-uy[0]

    # To speed things up, we precompute a library of time shifts,
    #  so we don't have to do it for each loop of frequency:
    tlib = np.zeros((n_stations,nux,nuy), dtype=np.complex128)#, dtype=np.float64)
    for ix in range(0,nux):
            for iy in range(0,nuy):
                tlib[:,ix,iy] = xx*ux[ix] + yy*uy[iy]
    # In this case, we assume station locations xx and yy are already relative to some convenient midpoint
    # Rather than shift one station to a single other station, we'll shift *all* stations back to that midpoint

    # Create data stores:
    Pfreq_all = np.zeros((data.shape[0],len(target_freqs),nux,nuy), dtype=np.complex128) # Explicitly create Pxx_all, as otherwise prange won't work correctly.

    # Then loop over windows:
    for win_idx in prange(data.shape[0]):
        # if(win_idx % 10 == 0):
        #     print("Processing for window", win_idx+1, "/", data.shape[0])
        
        # Calculate spectra:
        # Construct data structure:
        nfft = (2.0**np.ceil(np.log2(n_t_samp)))
        nfft = np.array(nfft, dtype=np.int64)
        Pxx_all = np.zeros((np.int((nfft/2)+1), n_stations), dtype=np.complex128) # Power spectra
        dt = 1. / fs 
        df = 1.0/(2.0*nfft*dt)
        xf = np.linspace(0.0, 1.0/(2.0*dt), np.int((nfft/2)+1))
        # Calculate power spectra for all stations:
        for sta_idx in range(n_stations):
            # Calculate spectra for current station:
            ###Pxx_all[:,sta_idx] = np.fft.rfft(data[win_idx,sta_idx,:], n=nfft) # (Use real fft, as input data is real) # DOESN'T WORK WITH NUMBA!
            with objmode(Pxx_curr='complex128[:]'):
                Pxx_curr = np.fft.rfft(data[win_idx,sta_idx,:], n=nfft)
            Pxx_all[:,sta_idx] = Pxx_curr

        # Loop over all freqs, performing phase shifts:
        Pfreq=np.zeros((len(target_freqs),nux,nuy),dtype=np.complex128)
        counter_grid = 0
        for ii in range(len(target_freqs)):
            # Find closest current freq.:
            target_f = target_freqs[ii]
            curr_f_idx = (np.abs(xf - target_f)).argmin()

            # Construct a matrix of each station-station correlation *before* any phase shifts
            Rxx=np.zeros((n_stations,n_stations),dtype=np.complex128)
            for i1 in range(0,n_stations):
                for i2 in range(0,n_stations):
                    # Remove autocorrelations:
                    if remove_autocorr:
                        if not i1 == i2:
                            Rxx[i1,i2] = np.conj(Pxx_all[curr_f_idx,i1]) * Pxx_all[curr_f_idx,i2]
                        else:
                            Rxx[i1,i2] = 0

            # And loop over phase shifts, calculating cross-correlation power:
            for ix in range(0,nux):
                for iy in range(0,nuy):
                    timeshifts = tlib[:,ix,iy] # Calculate the "steering vector" (a vector in frequency space, based on phase-shift)
                    a = np.exp(-1j*2*np.pi*target_f*timeshifts)
                    aconj = np.conj(a)
                    # "a" is a "steering vector." It contains info on all the phase delays needed to 
                    #  push our stations to the middle point.
                    # Since each element of Rxx contains cross-spectra of two stations, we need two timeshifts
                    #  to push both to the centerpoint. This can also be seen as projecting Rxx onto a new basis.
                    Pfreq[ii,ix,iy]=np.dot(np.dot(aconj,Rxx),a)     

        # And remove any data where stations don't exist:
        ###np.nan_to_num(Pfreq, copy=False, nan=0.0) # NOT SUPPORTED BY NUMBA SO DO OUTSIDE NUMBA
                    
        # And append output to datastore:
        Pfreq_all[win_idx,:,:,:] = Pfreq

    return Pfreq_all 


# @jit(nopython=True, parallel=True)#, nogil=True)
def _phase_associator_core_worker(peaks_Z, peaks_hor, bazis_Z, bazis_hor, bazi_tol, t_Z_secs_after_start, t_hor_secs_after_start, max_phase_sep_s):
    """Function to do the heavy lifting of the phase association."""
    # Specify data stores:
    Z_hor_phase_pair_idxs = []

    # Loop over phases, seeing if they meet phase association criteria:
    for i in range(len(peaks_Z)):
        if i % 1000 == 0:
            print(i,"/",len(peaks_Z))
        curr_peak_Z_idx = peaks_Z[i]
        for j in range(len(peaks_hor)):
            curr_peak_hor_idx = peaks_hor[j]
            # i. Check if phase arrivals are within specified time limits:
            curr_t_phase_diff = t_hor_secs_after_start[curr_peak_hor_idx] - t_Z_secs_after_start[curr_peak_Z_idx]
            if curr_t_phase_diff > 0:
                if curr_t_phase_diff <= max_phase_sep_s:
                    # ii. Check if bazis for Z and horizontals current pick match:
                    if np.abs( bazis_Z[i] - bazis_hor[j] ) < bazi_tol:
                        match = True
                    # And deal with if they are close to North:
                    elif np.abs( bazis_Z[i] - bazis_hor[j] ) > (360. - bazi_tol):
                        match = True
                    else:
                        match = False

                    # And associate phases and create event data if a match is found:
                    if match:
                        # Append pair idxs to data store:
                        Z_hor_phase_pair_idxs.append([curr_peak_Z_idx, curr_peak_hor_idx])
    
    return Z_hor_phase_pair_idxs


def _phase_associator(t_series_df_Z, t_series_df_hor, peaks_Z, peaks_hor, bazi_tol, filt_phase_assoc_by_max_power, max_phase_sep_s, min_event_sep_s, verbosity=0):
    """
    Function to perform phase association for numba implementation.
    """
    # Setup events datastores:
    events_df = pd.DataFrame()
    # Find back-azimuths associated with phase picks:
    bazis_Z = t_series_df_Z['back_azi'].values[peaks_Z]
    bazis_hor = t_series_df_hor['back_azi'].values[peaks_hor]

    #-------------------------------------------------------------------
    # Perform core phae association:
    # Prep. data for numba format:
    if verbosity > 1:
        print("Pre-processing time-series")
    t_Z_secs_after_start = []
    for index, row in t_series_df_Z.iterrows():
        t_Z_secs_after_start.append(obspy.UTCDateTime(row['t']) - obspy.UTCDateTime(t_series_df_Z['t'][0]))
    t_hor_secs_after_start = []
    for index, row in t_series_df_hor.iterrows():
        t_hor_secs_after_start.append(obspy.UTCDateTime(row['t']) - obspy.UTCDateTime(t_series_df_hor['t'][0]))
    # Run function:
    if verbosity > 1:
        print("Performing phase association")
    Z_hor_phase_pair_idxs = _phase_associator_core_worker(peaks_Z, peaks_hor, bazis_Z, bazis_hor, bazi_tol, t_Z_secs_after_start, t_hor_secs_after_start, max_phase_sep_s)
    # Organise outputs into useful form:
    if verbosity > 1:
        print("Writing events")
    for event_idx in range(len(Z_hor_phase_pair_idxs)):
        curr_peak_Z_idx = Z_hor_phase_pair_idxs[event_idx][0]
        curr_peak_hor_idx = Z_hor_phase_pair_idxs[event_idx][1]
        curr_event_df = pd.DataFrame({'t1': [t_series_df_Z['t'][curr_peak_Z_idx]], 't2': [t_series_df_hor['t'][curr_peak_hor_idx]], 
                                            'pow1': [t_series_df_Z['power'][curr_peak_Z_idx]], 'pow2': [t_series_df_hor['power'][curr_peak_hor_idx]], 
                                            'slow1': [t_series_df_Z['slowness'][curr_peak_Z_idx]], 'slow2': [t_series_df_hor['slowness'][curr_peak_hor_idx]], 
                                            'bazi1': [t_series_df_Z['back_azi'][curr_peak_Z_idx]], 'bazi2': [t_series_df_hor['back_azi'][curr_peak_hor_idx]]})
        events_df = events_df.append(curr_event_df)
    # And tidy:
    del t_Z_secs_after_start, t_hor_secs_after_start, Z_hor_phase_pair_idxs
    gc.collect()
    #-------------------------------------------------------------------

    # And filter events to only output events with max. power within the max. phase window, 
    # if speficifed by user:
    if filt_phase_assoc_by_max_power:
        # Only process if found some events:
        if len(events_df) > 0:
            # Calculate max power of P and S for each potential event:
            # events_overall_powers = events_df['pow1'].values + events_df['pow2'].values
            # Define datastores:
            filt_events_df = pd.DataFrame()
            # And loop over events, selecting only max. power events:
            tmp_count = 0
            for index, row in events_df.iterrows():
                tmp_count+=1
                if tmp_count == 1:
                    tmp_df = pd.DataFrame()
                    tmp_df = tmp_df.append(row)
                else:
                    # Append event if phase within minimum event separation:
                    if obspy.UTCDateTime(row['t1']) - obspy.UTCDateTime(tmp_df['t1'].iloc[0]) < min_event_sep_s:
                        # Append event to compare:
                        tmp_df = tmp_df.append(row)
                    else:
                        # Find best event from previous events:
                        combined_pows_tmp = tmp_df['pow1'].values + tmp_df['pow2'].values
                        max_power_idx = np.argmax(combined_pows_tmp)
                        filt_events_df = filt_events_df.append(tmp_df.iloc[max_power_idx])

                        # And start acrewing new events:
                        tmp_df = pd.DataFrame()
                        tmp_df = tmp_df.append(row)
            # And calculate highest power event for final window:
            combined_pows_tmp = tmp_df['pow1'].values + tmp_df['pow2'].values
            max_power_idx = np.argmax(combined_pows_tmp)
            filt_events_df = filt_events_df.append(tmp_df.iloc[max_power_idx])

            # And sort indices:
            filt_events_df.reset_index(drop=True, inplace=True)

            # And remove duplicate S pick associations:
            # (using same max. power method)
            # Append summed powers, for sorting:
            sum_pows = filt_events_df['pow1'].values + filt_events_df['pow2'].values
            sum_pows_df = pd.DataFrame({'sum_pows': sum_pows})
            filt_events_df = filt_events_df.join(sum_pows_df)
            # Remove t2 duplicates, keep highest summed power:
            filt_events_df = filt_events_df.sort_values('sum_pows').drop_duplicates(subset='t2', keep='last')
            # And remove sum_pows column:
            filt_events_df = filt_events_df.drop(columns=['sum_pows'])

            # And output df:
            events_df = filt_events_df.copy()
            del filt_events_df
            gc.collect()

    return events_df


def _submit_parallel_fast_freq_domain_array_proc(procnum, return_dict_Pfreq_all, data_curr_run, max_sl, fs, target_freqs, xx, yy, n_stations, n_t_samp, remove_autocorr):
    """Function to submit parallel runs of _fast_freq_domain_array_proc() function."""
    # Run function:
    Pfreq_all_curr_run = _fast_freq_domain_array_proc(data_curr_run, max_sl, fs, target_freqs, xx, yy, n_stations, n_t_samp, remove_autocorr)
    # And return data
    return_dict_Pfreq_all[procnum] = Pfreq_all_curr_run


def _calc_time_shift_from_array_cent(slow, bazi, x_arr_cent, y_arr_cent, x_rec, y_rec):
    """Calculates time shift of signal at receiver from array centre for stacking data.
    Note: All distances and velocities use km unless otherwise specified.
    
    Parameters
    ----------
    slow : float
        Slowness of arrival, in s/km.
    
    bazi : float
        Back azimuth of arrival in degrees from .

    x_arr_cent : float
        Location of centre of array in km East.

    y_arr_cent : float
        Location of centre of array in km North.

    x_rec :
        Location of receiver in km East.

    y_rec
        Location of receiver in km North.
    """
    # Calculate time shift:
    # (formulated on 16th Nov 2022)
    # Calculate angle from array centre to current receiver:
    x0 = x_arr_cent
    y0 = y_arr_cent
    x1 = x_rec
    y1 = y_rec
    station_to_array_centre_angle_from_N = (np.pi/2) -  np.arctan2((y1-y0) , (x1-x0)) 
    if station_to_array_centre_angle_from_N < 0:
        station_to_array_centre_angle_from_N += 2*np.pi
    # And calculate relative apparent velocity for current receiver:
    rel_vel_curr = -1 * (1. / slow) * np.cos(station_to_array_centre_angle_from_N - np.deg2rad(bazi)) # (Note: minus convention for arrival at receivers of similar back-azimuths to ray arrive first).
    # Calculate distance from centre to receiver:
    rec_dist_curr = np.sqrt( (x1-x0)**2 + (y1-y0)**2 )
    # And calculate time shift:
    time_shift_curr = rec_dist_curr / rel_vel_curr # t = d/v

    return time_shift_curr


def _create_stacked_data_st(st, Z_all, N_all, E_all):
    """Function to create stacked data st."""

    composite_st = obspy.Stream()
    # For Z stacked:
    tr = st[0].copy()
    tr.stats.station = "STACK"
    tr.stats.channel = st[0].stats.channel[0:2]+"Z"
    tr.data = np.sum(Z_all, axis=1)
    composite_st.append(tr)
    # For Z mean:
    tr = st[0].copy()
    tr.stats.station = "MEAN"
    tr.stats.channel = st[0].stats.channel[0:2]+"Z"
    tr.data = np.mean(Z_all, axis=1)
    composite_st.append(tr)
    # For Z stdev:
    tr = st[0].copy()
    tr.stats.station = "STDEV"
    tr.stats.channel = st[0].stats.channel[0:2]+"Z"
    tr.data = np.std(Z_all, axis=1)
    composite_st.append(tr)
    # For N stacked:
    tr = st[0].copy()
    tr.stats.station = "STACK"
    tr.stats.channel = st[0].stats.channel[0:2]+"N"
    tr.data = np.sum(N_all, axis=1)
    composite_st.append(tr)
    # For N mean:
    tr = st[0].copy()
    tr.stats.station = "MEAN"
    tr.stats.channel = st[0].stats.channel[0:2]+"N"
    tr.data = np.mean(N_all, axis=1)
    composite_st.append(tr)
    # For N stdev:
    tr = st[0].copy()
    tr.stats.station = "STDEV"
    tr.stats.channel = st[0].stats.channel[0:2]+"N"
    tr.data = np.std(N_all, axis=1)
    composite_st.append(tr)
    # For E stacked:
    tr = st[0].copy()
    tr.stats.station = "STACK"
    tr.stats.channel = st[0].stats.channel[0:2]+"E"
    tr.data = np.sum(E_all, axis=1)
    composite_st.append(tr)
    # For E mean:
    tr = st[0].copy()
    tr.stats.station = "MEAN"
    tr.stats.channel = st[0].stats.channel[0:2]+"E"
    tr.data = np.mean(E_all, axis=1)
    composite_st.append(tr)
    # For E stdev:
    tr = st[0].copy()
    tr.stats.station = "STDEV"
    tr.stats.channel = st[0].stats.channel[0:2]+"E"
    tr.data = np.std(E_all, axis=1)
    composite_st.append(tr)
    del tr
    gc.collect()

    return composite_st


class setup_detection:
    """
    Class to create detection object, for running array detection algorithm.

    Parameters
    ----------
    archivedir : str
        Path to data archive. Data archive must be of specific format:
        <archivedir>/YEAR/JULDAY/YEARJULDAY_*STATION_COMP.*

    outdir : str
        Path to directory to save outputs to.

    stations_fname : str
        Path to csv file containing station/receiver locations. Headers need 
        to be of format: Latitude  Longitude  Elevation  Name.

    starttime : obspy UTCDateTime object
        Start time of data window to process for.

    endtime : obspy UTCDateTime object
       End time of data window to process for.

    channels_to_use : list of strs (optional, default = ["??Z"])
        List of channels to use for the processing (e.g. HHZ, ??Z or similar).

    Attributes
    ----------

    freqmin : float
        If specified, lower frequency of bandpass filter, in Hz. Default is None.

    freqmax : float
        If specified, upper frequency of bandpass filter, in Hz. Default is None.

    num_freqs : int
        Number of discrete frequencies to use between <freqmin> and <freqmax> 
                in analysis. Default is 100. Note: Reducing this value increases 
                efficiency linearly, but costs in terms of absolute power. However, 
                SNR of power time-series remains approximately constant (to a point).
                Can affect slowness and bazi results, especially within noise, though.

    max_sl : float
        Maximum slowness to analyse for, in s / km. Default is 1.0 s / km.

    win_len_s : float
        The window length for each frequency domain step. Units are seconds.
        Default value is 0.1 s.
    
    remove_autocorr : bool
        If True, then will remove autocorrelations. Default is True.

    norm_pre_stacking : bool
        If True, normallises data before stacking. Similar to performing spectral 
        whitening. Default is False

    mad_window_length_s : float
        Length of time-window, in seconds, to calculate background Median Absolute 
        Deviation (MAD) for triggering events. Default is 3600 s.

    mad_multiplier : int
        The trigger level above the background MAD level to trigger a phase detection.
        Default = 8.

    min_event_sep_s : float
        Minimum separation between event detections, in seconds. Default = 1 s.

    bazi_tol : float 
        The back-azimuth tolerance to associate incoming phases with the same event.
        Various phases have to fulfil the criteria of having the same back-azimuth 
        for all phase arrivals, +/- <bazi_tol>. Units are degrees. Defaul is 20 degrees.

    max_phase_sep_s : float
        The maximum time separation between individual phases. Units are seconds. Default 
        is 2.5 s.

    filt_phase_assoc_by_max_power : bool
        If True, filters event phase association within a given window by max. power. I.e. 
        if True, will only pick highest amplitude phase arrivals on both vertical and 
        horizontal components, within the time window <max_phase_sep_s>.

    Methods
    -------

    """
    
    def __init__(self, archivedir, outdir, stations_fname, starttime, endtime, preload_fname=None, channels_to_use=["??Z"]):
        """Initiate the class object.

        Parameters
        ----------
        archivedir : str
            Path to data archive. Data archive must be of specific format:
            <archivedir>/YEAR/JULDAY/YEARJULDAY_*STATION_COMP.*

        outdir : str
            Path to directory to save outputs to.

        stations_fname : str
            Path to csv file containing station/receiver locations. Headers need 
            to be of format: Latitude  Longitude  Elevation  Name.

        starttime : obspy UTCDateTime object
            Start time of data window to process for.

        endtime : obspy UTCDateTime object
            End time of data window to process for.

        preload_fname : str
            Path to previously created detection class object. Optional. Default is 
            None, which means it doesn't load an existing file.

        channels_to_use : list of strs (optional, default = ["??Z"])
            List of channels to use for the processing (e.g. HHZ, ??Z or similar).

        """
        # Initialise input params:
        self.archivedir = archivedir
        self.outdir = outdir
        self.stations_fname = stations_fname
        self.starttime = starttime
        self.endtime = endtime
        self.channels_to_use = channels_to_use

        # Setup outdir:
        os.makedirs(outdir, exist_ok=True)

        # Setup station information:
        self.stations_df = pd.read_csv(self.stations_fname)

        # And define attributes:
        # For array processing:
        self.freqmin = None
        self.freqmax = None
        self.num_freqs = 100
        self.max_sl = 1.0
        self.win_len_s = 0.1
        self.remove_autocorr = True
        self.norm_pre_stacking = False
        # self.nproc = 1
        self.out_fnames_array_proc = []
        # For detection:
        self.mad_window_length_s = 3600
        self.mad_multiplier = 8
        self.min_event_sep_s = 1.
        self.bazi_tol = 20.
        self.max_phase_sep_s = 2.5
        self.filt_phase_assoc_by_max_power = True
        # For location:
        self.receiver_vp = None
        self.receiver_vs = None
        self.array_latlon = None

        # And load existing detection instance, if specified:
        if preload_fname:
            self.load(preload_fname)


    def _setup_array_receiver_coords(self):
        """Function to setup station receiver coords in correct format for 
        array processing."""
        ref_lat = np.mean(self.stations_df['Latitude'])
        ref_lon = np.mean(self.stations_df['Longitude'])
        # Calculate receiver locations in km grid format:
        xs = []
        ys = []
        for index, row in self.stations_df.iterrows():
            lon2 = row['Longitude']
            lat2 = row['Latitude']
            # Calc. inter-station distance:
            r, a, b = obspy.geodetics.base.gps2dist_azimuth(ref_lat, ref_lon, lat2, lon2)
            xs.append(r * np.sin(np.deg2rad(a)) / 1000)
            ys.append(r * np.cos(np.deg2rad(a)) / 1000)
        self.stations_df['x (km)'] = xs
        self.stations_df['y (km)'] = ys
        self.stations_df['z (km)'] = self.stations_df['Elevation'].values / 1000.
        self.stations_df['ref_lat'] = np.ones(len(xs)) * ref_lat
        self.stations_df['ref_lon'] = np.ones(len(xs)) * ref_lon

        # And calculate receiver locations in terms of array centre:
        array_centre = np.array([np.mean(self.stations_df['x (km)']), np.mean(self.stations_df['y (km)']), np.mean(self.stations_df['z (km)'])])
        self.stations_df['x_array_coords_km'] = self.stations_df['x (km)'].values - array_centre[0]
        self.stations_df['y_array_coords_km'] = self.stations_df['y (km)'].values - array_centre[1]
        self.stations_df['z_array_coords_km'] = self.stations_df['z (km)'].values - array_centre[2]


    def find_min_max_array_sensitivity(self, vel_assumed=3.0):
        """Function to find array min and max sensitivities.
        Parameters
        ----------
        vel_assumed : float (default = 3.0)
            Approx. velocity of seismic waves incident at the array, in km/s.
        """
        # And find min and max distance (and hence frequency) for the array:
        inter_station_dists = []
        # Loop over first set of stations:
        for index, row in self.stations_df.iterrows():
            lon1 = row['Longitude']
            lat1 = row['Latitude']
            # Loop over stations again:
            for index, row in self.stations_df.iterrows():
                lon2 = row['Longitude']
                lat2 = row['Latitude']
                # Calc. inter-station distance:
                r, a, b = obspy.geodetics.base.gps2dist_azimuth(lat1, lon1, lat2, lon2)
                inter_station_dists.append(np.abs(r) / 1000)
        inter_station_dists = np.array(inter_station_dists)
        inter_station_dists = inter_station_dists[inter_station_dists != 0]
        print("="*60)
        print("Min. inter-station distance:", np.min(inter_station_dists), "km")
        print("Therefore, optimal sensitive higher freq.:", vel_assumed / np.min(inter_station_dists), "Hz")
        print("Max. inter-station distance:", np.max(inter_station_dists), "km")
        print("Therefore, optimal sensitive lower freq.:", vel_assumed / np.max(inter_station_dists), "Hz")
        print("="*60)


    def _load_day_of_data(self, year, julday, hour=None):
        """Function to load a day of data."""
        # Load in data:
        mseed_dir = os.path.join(self.archivedir, str(year), str(julday).zfill(3))
        st = obspy.Stream()
        for index, row in self.stations_df.iterrows():
            station = row['Name']
            for channel in self.channels_to_use:
                try:
                    if hour:
                        st_tmp = obspy.read(os.path.join(mseed_dir, ''.join((str(year), str(julday).zfill(3), "_", str(hour).zfill(2), "*", station, "*", channel, "*"))))
                    else:
                        st_tmp = obspy.read(os.path.join(mseed_dir, ''.join((str(year), str(julday).zfill(3), "_*", station, "*", channel, "*"))))
                    for tr in st_tmp:
                        st.append(tr)
                except:
                    print("No data for "+station+", channel = "+channel+". Skipping this data.")
                    continue
        # Merge data:
        st.detrend('demean')
        st.merge(method=1, fill_value=0.)
        # And apply filter:
        if self.freqmin:
            if self.freqmax:
                st.filter('bandpass', freqmin=self.freqmin, freqmax=self.freqmax)
        # And trim data, if some lies outside start and end times:
        if self.starttime > st[0].stats.starttime:
            st.trim(starttime=self.starttime)
        if self.endtime < st[0].stats.endtime:
            st.trim(endtime=self.endtime)
        return st 

    
    def _convert_st_to_np_data(self, st):
        """Function to convert data to numpy format for processing."""
        self.n_win = int((st[0].stats.endtime - st[0].stats.starttime) / self.win_len_s)
        self.fs = st[0].stats.sampling_rate
        self.n_t_samp = int(self.win_len_s * self.fs) # num samples in time
        station_labels = self.stations_df['Name'].values
        self.n_stations = len(station_labels) # num stations
        data = np.zeros((self.n_win, self.n_stations, self.n_t_samp))
        for i in range(self.n_win):
            for j in range(self.n_stations):
                station = station_labels[j]
                win_start_idx = i * self.n_t_samp
                win_end_idx = (i * self.n_t_samp) + self.n_t_samp
                try:
                    data[i,j,:] = st.select(station=station, channel=self.channel_curr)[0].data[win_start_idx:win_end_idx]
                except IndexError:
                    # Deal with if a particular station has no data for given window:
                    data[i,j,:] = 0.
        return data 

    
    def _stack_results(self, Pfreq_all):
        """Function to perform stacking of the results."""
        Psum_all = np.zeros((Pfreq_all.shape[0], Pfreq_all.shape[2], Pfreq_all.shape[3]), dtype=complex)
        # Loop over time windows:
        for i in range(Pfreq_all.shape[0]):
            if self.norm_pre_stacking:
                Pfreq_norm_curr = Pfreq_all[i,:,:,:] / np.sum(np.abs(Pfreq_all[i,:,:,:]), axis=0)
                Psum_all[i,:,:] = np.sum(Pfreq_norm_curr,axis=0)
            else:
                Psum_all[i,:,:] = np.sum(Pfreq_all[i,:,:,:],axis=0)
        return Psum_all

    def _find_time_series(self, Psum_all):
        """Function to calculate beamforming time-series outputs, given 
        a raw beamforming result.
        Psum_all - Sum of Pxx for each time window, and for all slownesses.
                    Shape is n_win x slowness WE x slowness SN
        Returns time-series of coherency (power), slowness and back-azimuth.
        """
        # Calcualte ux, uy:
        ux = np.linspace(-self.max_sl,self.max_sl,Psum_all.shape[1])
        uy = np.linspace(-self.max_sl,self.max_sl,Psum_all.shape[2])
        dux=ux[1]-ux[0]
        duy=uy[1]-uy[0]
        # Create time-series:
        n_win_curr = Psum_all.shape[0]
        t_series = np.arange(self.win_len_s/2,(n_win_curr*self.win_len_s) + (self.win_len_s/2), self.win_len_s)
        if len(t_series) > n_win_curr:
            t_series = t_series[0:n_win_curr]
        # And find power, slowness and back-azimuth:
        powers = np.zeros(n_win_curr)
        slownesses = np.zeros(n_win_curr)
        back_azis = np.zeros(n_win_curr)
        # Loop over windows in time:
        for i in range(n_win_curr):
            # Calculate max. power:
            powers[i] = np.max(np.abs(Psum_all[i,:,:]))
            # Calculate slowness:
            x_idx = np.where(Psum_all[i,:,:] == Psum_all[i,:,:].max())[0][0]
            y_idx = np.where(Psum_all[i,:,:] == Psum_all[i,:,:].max())[1][0]
            x_sl = ux[x_idx] - dux/2
            y_sl = uy[y_idx] - duy/2
            slownesses[i] = np.sqrt(x_sl**2 + y_sl**2)
            # And calculate back-azimuth:
            back_azis[i] = np.rad2deg( np.arctan2( x_sl, y_sl ) )
            if back_azis[i] < 0:
                back_azis[i] += 360
            
        return t_series, powers, slownesses, back_azis
    
    def run_array_proc(self):
        """Function to run core array processing.
        Performed in frequency domain. Involves applying phase (equiv. to time) shift 
        for each frequency, over a range of specified slownesses.
        Function inspured by work of D. Bowden (see Bowden et al. (2020))."""
        # Prep. stations df:
        self._setup_array_receiver_coords()

        # Loop over years:
        for year in range(self.starttime.year, self.endtime.year+1):
            # Loop over days:
            for julday in range(1,367):
                # Do some filtering for first and last years:
                if year == self.starttime.year:
                    if julday < self.starttime.julday:
                        continue # Ignore day, as out of range
                if year == self.endtime.year:
                    if julday > self.endtime.julday:
                        continue # Ignore day, as out of range
                
                # And process data:

                # Loop over channels:
                for self.channel_curr in self.channels_to_use:
                    print("="*60)
                    print("Processing data for year "+str(year)+", day "+str(julday).zfill(3)+", channel "+self.channel_curr)

                    # And process for individual hours:
                    # (to reduce memory usage)
                    for hour in range(24):
                        print("Processing for hour", str(hour).zfill(2))
                        if self.starttime > obspy.UTCDateTime(year=year, julday=julday, hour=hour) + 3600:
                            continue
                        if self.endtime < obspy.UTCDateTime(year=year, julday=julday, hour=hour):
                            continue

                        # Create datastore:
                        out_df = pd.DataFrame({'t': [], 'power': [], 'slowness': [], 'back_azi': []})

                        # Load data:
                        st = self._load_day_of_data(year, julday, hour=hour)
                        # starttime_this_day = obspy.UTCDateTime(year=year, julday=julday)
                        try:
                            starttime_this_st = st[0].stats.starttime
                        except IndexError:
                            # And skip if no data:
                            print("Skipping hour as no data")
                            del st 
                            gc.collect()
                            continue
                        
                        # And loop over minutes (to save on memory issues):
                        for minute in range(60):
                            # Check whether specified window is greater than a minute in duration:
                            if self.endtime - self.starttime > 60:
                                # Check time within specified run window:
                                if self.starttime > obspy.UTCDateTime(year=year, julday=julday, hour=hour, minute=minute) + 60:
                                    continue
                                if self.endtime < obspy.UTCDateTime(year=year, julday=julday, hour=hour, minute=minute):
                                    continue
                            elif self.starttime.minute != minute:
                                continue
                                
                            # Trim data:
                            st_trimmed = st.copy()
                            if self.endtime - self.starttime > 60:
                                st_trimmed.trim(starttime=obspy.UTCDateTime(year=year, julday=julday, hour=hour, minute=minute), endtime=obspy.UTCDateTime(year=year, julday=julday, hour=hour, minute=minute)+60)
                            else:
                                st_trimmed.trim(starttime=self.starttime, endtime=self.endtime)

                            # Convert data to np:
                            data = self._convert_st_to_np_data(st_trimmed)
                            del st_trimmed 
                            gc.collect()

                            # Run heavy array processing algorithm:
                            # Specify various variables needed:
                            # Make a linear spacing of frequencies. One might use periods, logspacing, etc.:
                            # (Note: linspace much less noisy than logspace)
                            target_freqs = np.linspace(self.freqmin,self.freqmax,self.num_freqs) #np.logspace(self.freqmin,self.freqmax,self.num_freqs) 
                            # Station locations:
                            xx = self.stations_df['x_array_coords_km'].values
                            yy = self.stations_df['y_array_coords_km'].values
                            # And run:
                            tic = time.time()
                            print("Performing run for",data.shape[0],"windows")
                            Pfreq_all = _fast_freq_domain_array_proc(data, self.max_sl, self.fs, target_freqs, xx, yy, 
                                                                            self.n_stations, self.n_t_samp, self.remove_autocorr)
                            toc = time.time()
                            print(toc-tic)
                            # And tidy:
                            del data 
                            gc.collect()

                            # And remove any data where stations don't exist:
                            # for i in np.arange(len(Pfreq_all)):
                                # Pfreq_all[i] = np.nan_to_num(Pfreq_all[i], copy=False, nan=0.0)
                            Pfreq_all = np.nan_to_num(Pfreq_all, copy=False, nan=0.0)

                            # Sum/stack/normallise data:
                            Psum_all = self._stack_results(Pfreq_all)
                            del Pfreq_all
                            gc.collect()

                            # Calculate time-series outputs (for detection) from data:
                            t_series, powers, slownesses, back_azis = self._find_time_series(Psum_all)
                            
                            # And append to data out:
                            t_series_out = []
                            if self.endtime - self.starttime > 60:
                                for t_serie in t_series:
                                    t_series_out.append( str(starttime_this_st + (minute*60) + t_serie) )
                            else:
                                for t_serie in t_series:
                                    t_series_out.append( str(starttime_this_st + t_serie) )
                            tmp_df = pd.DataFrame({'t': t_series_out, 'power': powers, 'slowness': slownesses, 'back_azi': back_azis})
                            out_df = out_df.append(tmp_df)
                            out_df.reset_index(drop=True, inplace=True)

                            # And save data out:
                            out_fname = os.path.join(self.outdir, ''.join(("detection_t_series_", str(year).zfill(4), str(julday).zfill(3), "_", 
                                                        str(starttime_this_st.hour).zfill(2), "00", "_ch", self.channel_curr[-1], ".csv")))
                            out_df.to_csv(out_fname, index=False)

                            # And append fname to history:
                            self.out_fnames_array_proc.append(out_fname)

                            # And clear memory:
                            del Psum_all, t_series, powers, slownesses, back_azis
                            gc.collect()
                            
        return None

    def _calculate_mad(self, x, scale=1.4826):
        """
        Calculates the Median Absolute Deviation (MAD) of the input array x.
        Outputs an array of scaled mean absolute deviation values for the input array, x,
        scaled to provide an estimation of the standard deviation of the distribution.
        """
        # Calculate median and mad values:
        mad = np.median(np.abs(x - np.median(x)))
        return scale * mad
    
    
    def detect_events(self, verbosity=0):
        """Function to detect events, based on the power time-series generated 
        by run_array_proc(). Note: Currently, only Median Absolute Deviation 
        triggering is implemented.
        Key attributes used are:
        - mad_window_length_s
        - mad_multiplier
        - min_event_sep_s
        """
        print("Note: <mad_window_length_s> not yet implemented.")
        # Create datastore:
        events_df_all = pd.DataFrame()
        # Loop over array proc outdir data:
        for fname in glob.glob(os.path.join(self.outdir, "detection_t_series_*_chZ.csv")):
            f_uid = fname[-20:-8]
            # Check if in list to process:
            if fname in self.out_fnames_array_proc:
                # And load in data:
                # Read in vertical data:
                t_series_df_Z = pd.read_csv(fname)
                # And read in horizontals:
                try:
                    fname_N = os.path.join(self.outdir, ''.join(( "detection_t_series_", f_uid, "_chN.csv" )))
                    t_series_df_N = pd.read_csv(fname_N)
                except FileNotFoundError:
                    fname_N = os.path.join(self.outdir, ''.join(( "detection_t_series_", f_uid, "_ch1.csv" )))
                    t_series_df_N = pd.read_csv(fname_N)
                try:
                    fname_E = os.path.join(self.outdir, ''.join(( "detection_t_series_", f_uid, "_chE.csv" )))
                    t_series_df_E = pd.read_csv(fname_E)
                except FileNotFoundError:
                    fname_E = os.path.join(self.outdir, ''.join(( "detection_t_series_", f_uid, "_ch2.csv" )))
                    t_series_df_E = pd.read_csv(fname_E)
            else:
                continue # Skip file, as not previously been processed.
            # And check to see that t-series exists within file:
            if len(t_series_df_Z) == 0:
                continue 
            if len(t_series_df_N) == 0:
                continue 
            if len(t_series_df_E) == 0:
                continue 

            # Combine horizontals:
            # (Using RMS of N and E signals for slowness and average for BAZI)
            t_series_df_hor = t_series_df_N.copy()
            t_series_df_hor["power"] = np.sqrt(t_series_df_N["power"].values**2 + t_series_df_E["power"].values**2)
            NE_Pxx_max = np.max(np.concatenate((t_series_df_N["power"].values, t_series_df_E["power"].values)))
            N_weighting = t_series_df_N["power"].values / NE_Pxx_max
            E_weighting = t_series_df_E["power"].values / NE_Pxx_max
            t_series_df_hor["slowness"] = np.sqrt(np.average(np.vstack((t_series_df_N['slowness']**2, t_series_df_E['slowness']**2)), 
                                                        axis=0, weights=np.vstack((N_weighting, 
                                                        E_weighting)))) # Weighted mean (weighted by power)
            t_series_df_hor["back_azi"] = np.average(np.vstack((t_series_df_N['back_azi'], t_series_df_E['back_azi'])), 
                                                        axis=0, weights=np.vstack((N_weighting, 
                                                        E_weighting))) # Weighted mean (weighted by power)
            print("(Weighted horizontal slowness and back-azi using power)")
            del N_weighting, E_weighting, t_series_df_N, t_series_df_E
            gc.collect()

            # Calculate pick thresholds:
            mad_pick_threshold_Z = np.median(t_series_df_Z['power'].values) + (self.mad_multiplier * self._calculate_mad(t_series_df_Z['power']))
            mad_pick_threshold_hor = np.median(t_series_df_hor['power'].values) + (self.mad_multiplier * self._calculate_mad(t_series_df_hor['power']))
            
            # Get phase picks:
            min_pick_dist = int(self.min_event_sep_s / (obspy.UTCDateTime(t_series_df_Z['t'][1]) - obspy.UTCDateTime(t_series_df_Z['t'][0])))
            peaks_Z, _ = find_peaks(t_series_df_Z['power'].values, height=mad_pick_threshold_Z, distance=min_pick_dist)
            peaks_hor, _ = find_peaks(t_series_df_hor['power'].values, height=mad_pick_threshold_hor, distance=min_pick_dist)

            # Phase assoicate by BAZI threshold and max. power:
            events_df = _phase_associator(t_series_df_Z, t_series_df_hor, peaks_Z, peaks_hor, 
                                                self.bazi_tol, self.filt_phase_assoc_by_max_power, self.max_phase_sep_s, self.min_event_sep_s, verbosity=verbosity)

            # Append to datastore:
            events_df_all = events_df_all.append(events_df)

            # Plot detected, phase-associated picks:
            if verbosity > 1:
                print("="*40)
                print("Event phase associations:")            
                print(events_df)
                print("="*40)
                plt.figure(figsize=(8,6))
                plt.plot(t_series_df_Z['t'], t_series_df_Z['power'], label="Vertical power")
                plt.plot(t_series_df_hor['t'], t_series_df_hor['power'], label="Horizontal power")
                plt.scatter(events_df['t1'], np.ones(len(events_df))*np.max(t_series_df_Z['power']), c='r', label="P phase picks")
                plt.scatter(events_df['t2'], np.ones(len(events_df))*np.max(t_series_df_Z['power']), c='b', label="S phase picks")
                plt.legend()
                plt.xlabel("Time")
                plt.ylabel("Power (arb. units)")
                # plt.gca().yaxis.set_major_locator(MaxNLocator(5)) 
                plt.gca().xaxis.set_major_locator(plt.MaxNLocator(3))
                plt.show()
        
        return events_df_all
    
    
    def create_location_LUTs(self, oneD_vel_model_z_df, extent_x_m=4000, dxz=[100,100], array_centre_xz=[0, 0]):
        """Function to create lookup tables used for location. Lookup tables created are:
        - P travel-times
        - S travel-times.
        - P inclination angles.
        - S inclination angles.
        Note: Array centre is typically defined as the 

        Parameters
        ----------
        oneD_vel_model_z_df : pandas DataFrame
            Pandas DataFrame containing 1D velocity model, with the following columns:
            depth | vp | vs
            All units are SI units (i.e. metres, metres/second). Depths are positive down.

        extent_x_m : float
            Extent of lookup table horizontal. Units are metres. Default is 4000 m.

        array_centre_xz : list
            Path to csv file containing station/receiver locations. Headers need 
            to be of format: Latitude  Longitude  Elevation  Name. No need to change unless 
            the user has a particularly good reason. Takes a list of length two, containing 
            the array centre in x and z, in metres from the LUT grid origin.

        dxz : list
            List of two floats, defining the spatial spacing of the nodes in the LUTs in x 
            and z. Units are metres. Default is [100, 100].

        Returns
        -------
        Returns a pickled LUT dict to <outdir>/LUT.
        """
        # Assign LUT creation parameters to class attributes:
        self.oneD_vel_model_z_df = oneD_vel_model_z_df
        self.extent_x_m = extent_x_m
        self.dxz = dxz
        self.array_centre_xz = array_centre_xz

        # Create 2D LUTs:
        # (for P and S travel-times, and inclination angle)
        # And get travel times:
        trav_times_grid_P, trav_times_grid_S, theta_grid_P, theta_grid_S, vel_model_x_labels, vel_model_z_labels = lookup_table_manager.create_2D_LUT(oneD_vel_model_z_df, 
                                                                                                                            array_centre_xz, extent_x_m=extent_x_m, dxz=dxz)
        # And save outputs:
        LUT_outdir = os.path.join(self.outdir, "LUT")
        os.makedirs(LUT_outdir, exist_ok=True)
        LUTs_dict = {}
        LUTs_dict['trav_times_grid_P'] = trav_times_grid_P
        LUTs_dict['trav_times_grid_S'] = trav_times_grid_S
        LUTs_dict['theta_grid_P'] = theta_grid_P
        LUTs_dict['theta_grid_S'] = theta_grid_S
        LUTs_dict['vel_model_x_labels'] = vel_model_x_labels
        LUTs_dict['vel_model_z_labels'] = vel_model_z_labels
        LUTs_dict['oneD_vel_model_z_df'] = self.oneD_vel_model_z_df
        LUTs_dict['extent_x_m'] = self.extent_x_m
        LUTs_dict['dxz'] = self.dxz
        LUTs_dict['array_centre_xz'] = self.array_centre_xz
        self.LUTs_fname = os.path.join(LUT_outdir, 'LUTs.pkl')
        pickle.dump( LUTs_dict, open( self.LUTs_fname, "wb" ) )
        print("Saved LUT to:", self.LUTs_fname)
        self.LUTs_dict = LUTs_dict 
        return LUTs_dict

    def load_location_LUTs(self, LUTs_fname=None):
        """Function to load lookup tables used for location.
        Parameters
        ----------
        LUTs_fname : str
            Path to LUT file to load. Optional. Default is to use the attribute 
            <LUTs_fname>. 
        """
        # Assign attribute, if not already specified correctly:
        if LUTs_fname:
            self.LUTs_fname = LUTs_fname
        # And load LUTs:
        LUTs_dict = pickle.load( open( self.LUTs_fname, "rb" ) )
        # And assign relevent attributes:
        self.oneD_vel_model_z_df = LUTs_dict['oneD_vel_model_z_df']
        self.extent_x_m = LUTs_dict['extent_x_m']
        self.dxz = LUTs_dict['dxz']
        self.array_centre_xz = LUTs_dict['array_centre_xz']
        self.LUTs_dict = LUTs_dict 
        return LUTs_dict

    
    def locate_events(self, events_df, verbosity=0):
        """Function to locate events using LUT."""
        # Perform tests to check that various required attributes are specified:
        exit_bool = False
        if not self.LUTs_dict:
            print("Warning: LUTs_dict class attribute not specified. Exiting.")
            exit_bool = True
        if not self.array_latlon:
            print("Warning: array_latlon class attribute not specified. Exiting.")
            exit_bool = True
        if not self.receiver_vp:
            print("Warning: receiver_vp class attribute not specified. Exiting.")
            exit_bool = True
        if not self.receiver_vs:
            print("Warning: receiver_vs class attribute not specified. Exiting.")
            exit_bool = True

        # Locate events:
        if not exit_bool:
            events_df = location.locate_events_from_P_and_S_array_arrivals(events_df, self.LUTs_dict, self.array_latlon, 
                                                                            self.receiver_vp, self.receiver_vs, 
                                                                            verbosity=verbosity)
        
        return events_df
    

    def save(self, out_fname=None):
        """Function to save class object to file.
        Parameters
        ----------
        out_fname : str
            Path to save class object to. Optional. If not specified, then save to 
            <outdir>/detect_obj.pkl.
        """
        # Save class to file:
        if not out_fname:
            out_fname = os.path.join(self.outdir, "detect_obj.pkl")
        f = open(out_fname, 'wb')
        pickle.dump(self.__dict__, f)
        f.close()
        print("Saved detection instance to:", out_fname)
    
    def load(self, preload_fname):
        """try load self.name.txt"""
        f = open(preload_fname, 'rb')
        self.__dict__ = pickle.load(f) 
        f.close()
        print("Loaded detection instance from:", preload_fname)

    
    def get_composite_array_st_from_bazi_slowness(self, arrival_time, bazis_1_2, slows_1_2, t_before_s=10, t_after_s=10, st_out_fname='out.m', return_streams=False):
        """Function to find array stacked stream from back-azimuth and slowness. Returns average amplitude 
        time-series seismogram of stacked array data, for all three componets.
        Parameters
        ----------
        arrival_time : obspy UTCDateTime object
            Arrival time at array. Will output window around this time, as defined by 
            <t_before_s> and <t_after_s>.

        bazis_1_2 : list of 2 floats
            Back-azimuth of event phase arrivals in vertical and horizontal, in degrees from North.

        slows_1_2 : list of 2 floats
            Slowness of event phase arrivals in vertical and horizontal, in seconds/km.

        t_before_s : float
            Time, in seconds, before arrival time to include in window. Optional. Default 
            is 10 s.

        t_after_s : float
            Time, in seconds, after arrival time to include in window. Optional. Default 
            is 10 s.   
        
        st_out_fname : str
            Filename of to save mseed data stream to. Optional. Default is out.m.

        return_streams : bool
            If True, returns st and composite_st. Optional. Default = False.  
        """
        # Load in raw mseed data:
        st = self._load_day_of_data(arrival_time.year, arrival_time.julday, hour=arrival_time.hour)
        # And trim data:
        st.trim(starttime=arrival_time-t_before_s, endtime=arrival_time+t_after_s)

        # Upsample data soas to provide best time shift later:
        st.interpolate(sampling_rate=10*st[0].stats.sampling_rate)

        # Find and perform time shifts for all receivers:
        # Get array centre coords (in km):
        # _setup_array_receiver_coords
        try:
            self.stations_df['x_array_coords_km'].values
        except KeyError:
            self._setup_array_receiver_coords()
        xx = self.stations_df['x_array_coords_km'].values
        yy = self.stations_df['y_array_coords_km'].values
        x_arr_cent = np.mean(xx)
        y_arr_cent = np.mean(yy)
        # Loop over stations:
        for i in range(len(st)):
            # Find time shift:
            # Get current station location:
            curr_station = st[i].stats.station
            x_rec = self.stations_df.loc[self.stations_df['Name'] == curr_station]['x_array_coords_km'].values[0]
            y_rec = self.stations_df.loc[self.stations_df['Name'] == curr_station]['y_array_coords_km'].values[0]
            # Select either vertical or horizontal slowness and back-azimuth, depending on component:
            comp = st[i].stats.channel[-1]
            if comp == 'Z':
                bazi = bazis_1_2[0]
                slow = slows_1_2[0]
            elif comp == 'N' or comp == 'E' or comp == '1' or comp == '2':
                bazi = bazis_1_2[1]
                slow = slows_1_2[1]
            # Calculate arrival time shift relative to centre of array:
            time_shift_curr_s = _calc_time_shift_from_array_cent(slow, bazi, x_arr_cent, y_arr_cent, x_rec, y_rec)
            
            # And perform time shift on data:
            n_samp_to_shift = round(time_shift_curr_s * st[i].stats.sampling_rate)
            st[i].data = np.roll(st[i].data, n_samp_to_shift)
        
        # And find stacked, mean and stdev of data:
        n_stat = len(st.select(channel="??Z"))
        # Get unique channels:
        chan_labels_tmp = []
        chan_labels_unique = []
        for tr in st:
            chan_labels_tmp.append(tr.stats.channel)
        chan_labels_unique = list(set(chan_labels_tmp))
        # Create datastores to save to:
        Z_all = np.zeros((len(st[0].data), n_stat))
        N_all = np.zeros((len(st[0].data), n_stat))
        E_all = np.zeros((len(st[0].data), n_stat))

        # Loop over st:
        for i in range(n_stat):
            Z_all[:,i] = st.select(channel="??Z")[i].data
            try: 
                N_all[:,i] = st.select(channel="??N")[i].data
                E_all[:,i] = st.select(channel="??E")[i].data
            except IndexError:
                # And write if uses 1 and 2 labels rather than N and E:
                N_all[:,i] = st.select(channel="??1")[i].data
                E_all[:,i] = st.select(channel="??2")[i].data            

        # And create stacked data stream:
        composite_st = _create_stacked_data_st(st, Z_all, N_all, E_all)

        # And decimate data back down to original sampling rate:
        st.decimate(10, no_filter=True)
        composite_st.decimate(10, no_filter=True)

        # And save data:
        if st_out_fname:
            st.write(st_out_fname, format="MSEED")
            composite_st_out_fname = st_out_fname.split('.')[0] + "_composite.m"
            composite_st.write(composite_st_out_fname, format="MSEED")

        if return_streams:
            return st, composite_st
        else:
            del st, composite_st
            gc.collect()













#----------------------------------------------- End: Define main functions -----------------------------------------------



