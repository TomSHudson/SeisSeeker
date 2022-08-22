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


def _submit_parallel_fast_freq_domain_array_proc(procnum, return_dict_Pfreq_all, data_curr_run, max_sl, fs, target_freqs, xx, yy, n_stations, n_t_samp, remove_autocorr):
    """Function to submit parallel runs of _fast_freq_domain_array_proc() function."""
    # Run function:
    Pfreq_all_curr_run = _fast_freq_domain_array_proc(data_curr_run, max_sl, fs, target_freqs, xx, yy, n_stations, n_t_samp, remove_autocorr)
    # And return data
    return_dict_Pfreq_all[procnum] = Pfreq_all_curr_run


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
    
    def __init__(self, archivedir, outdir, stations_fname, starttime, endtime, channels_to_use=["??Z"]):
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


    def _load_day_of_data(self, year, julday):
        """Function to load a day of data."""
        # Load in data:
        mseed_dir = os.path.join(self.archivedir, str(year), str(julday).zfill(3))
        st = obspy.Stream()
        for index, row in self.stations_df.iterrows():
            station = row['Name']
            for channel in self.channels_to_use:
                try:
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
        init_numba_compile_switch = True
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

                    # Create datastore:
                    out_df = pd.DataFrame({'t': [], 'power': [], 'slowness': [], 'back_azi': []})

                    # Load data:
                    st = self._load_day_of_data(year, julday)
                    starttime_this_day = st[0].stats.starttime

                    # Convert data to np:
                    data = self._convert_st_to_np_data(st)
                    del st 
                    gc.collect()

                    # Run heavy array processing algorithm:
                    # Specify various variables needed:
                    # Make a linear spacing of frequencies. One might use periods, logspacing, etc.:
                    # (Note: linspace much less noisy than logspace)
                    target_freqs = np.linspace(self.freqmin,self.freqmax,self.num_freqs) #np.logspace(self.freqmin,self.freqmax,self.num_freqs) 
                    # Station locations:
                    xx = self.stations_df['x_array_coords_km'].values
                    yy = self.stations_df['y_array_coords_km'].values
                    # Run initial numba jit compile:
                    if init_numba_compile_switch:
                        print("Performing initial compile")
                        init_data = data[0,:,:].copy()
                        init_data = init_data.reshape((1, data[0,:,:].shape[0], data[0,:,:].shape[1]))
                        Pfreq_all = _fast_freq_domain_array_proc(init_data, self.max_sl, self.fs, target_freqs, xx, yy, 
                                                                    self.n_stations, self.n_t_samp, self.remove_autocorr)
                        init_numba_compile_switch = False
                    # And run:
                    # if self.nproc == 1:
                    tic = time.time()
                    print("Performing run for",data.shape[0],"windows")
                    Pfreq_all = _fast_freq_domain_array_proc(data, self.max_sl, self.fs, target_freqs, xx, yy, 
                                                                    self.n_stations, self.n_t_samp, self.remove_autocorr)
                    toc = time.time()
                    print(toc-tic)
                    # else:
                    #     # pool = mp.Pool(mp.cpu_count())
                    #     # print("Using", mp.cpu_count(), "CPUs.")
                    #     # Pfreq_all = [pool.apply(_fast_freq_domain_array_proc, args=(win.reshape((1, win.shape[0], win.shape[1])), self.max_sl, self.fs, target_freqs, xx, yy, self.n_stations, self.n_t_samp, self.remove_autocorr)) for win in data] # Loop over windows of data
                    #     # pool.close()

                    #     print("Using", self.nproc, "CPUs.")
                    #     # Start jobs:
                    #     manager = mp.Manager()
                    #     return_dict_Pfreq_all = manager.dict() # for returning data
                    #     jobs = []
                    #     n_samp_per_proc = int(data.shape[0] / self.nproc)
                    #     for procnum in np.arange(self.nproc):
                    #         win_start_idx = int(procnum * n_samp_per_proc)
                    #         if procnum + 1 < self.nproc:
                    #             win_end_idx = int((procnum + 1) * n_samp_per_proc)
                    #         else:
                    #             win_end_idx = -1
                    #         p = mp.Process(target=_submit_parallel_fast_freq_domain_array_proc, args=(procnum, return_dict_Pfreq_all, 
                    #                         data[win_start_idx:win_end_idx, :, :], self.max_sl, self.fs, target_freqs, xx, yy, self.n_stations, self.n_t_samp, 
                    #                         self.remove_autocorr))
                    #         jobs.append(p)
                    #         p.start() # Start process
                    #     # And join processes together:
                    #     for p in jobs:
                    #         p.join()
                    #     # And get data:
                    #     for procnum in np.arange(self.nproc):
                    #         # Append, as in chronological order already:
                    #         Pfreq_all = return_dict_Pfreq_all[procnum]
                    #     Pfreq_all = flatten_list(Pfreq_all)
                    #     del return_dict_Pfreq_all
                    #     gc.collect()

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
                    for t_serie in t_series:
                        t_series_out.append( str(starttime_this_day + t_serie) )
                    tmp_df = pd.DataFrame({'t': t_series_out, 'power': powers, 'slowness': slownesses, 'back_azi': back_azis})
                    out_df = out_df.append(tmp_df)

                    # And save data out:
                    out_fname = os.path.join(self.outdir, ''.join(("detection_t_series_", str(year).zfill(4), str(julday).zfill(3), "_", 
                                                str(starttime_this_day.hour).zfill(2), "00", "_ch", self.channel_curr[-1], ".csv")))
                    out_df.to_csv(out_fname, index=False)
                    self.out_fnames_array_proc.append(out_fname)

                    # And clear memory:
                    del Psum_all, t_series, powers, slownesses, back_azis, out_df
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
    
    def _phase_associator(self, t_series_df_Z, t_series_df_hor, peaks_Z, peaks_hor):
        """
        Function to perform phase association.
        """
        # Setup events datastore:
        events_df = pd.DataFrame()
        # Find back-azimuths associated with phase picks:
        bazis_Z = t_series_df_Z['back_azi'].values[peaks_Z]
        bazis_hor = t_series_df_hor['back_azi'].values[peaks_hor]
        # Loop over phases, seeing if they meet phase association criteria:
        for i in range(len(peaks_Z)):
            curr_peak_Z_idx = peaks_Z[i]
            for j in range(len(peaks_hor)):
                curr_peak_hor_idx = peaks_hor[j]
                # i. Check if bazis for Z and horizontals current pick match:
                if np.abs( bazis_Z[i] - bazis_hor[j] ) < self.bazi_tol:
                    match = True
                # And deal with if they are close to North:
                elif np.abs( bazis_Z[i] - bazis_hor[j] ) > (360. - self.bazi_tol):
                    match = True
                else:
                    match = False
                # ii. Check if phase arrivals are within specified limits:
                if match == True:
                    if obspy.UTCDateTime(t_series_df_hor['t'][curr_peak_hor_idx]) - obspy.UTCDateTime(t_series_df_Z['t'][curr_peak_Z_idx]):
                        match = True
                    else:
                        match = False 

                # And associate phases and create event data if a match is found:
                if match:
                    # Write event:
                    curr_event_df = pd.DataFrame({'t1': [t_series_df_Z['t'][curr_peak_Z_idx]], 't2': [t_series_df_hor['t'][curr_peak_hor_idx]], 
                                                'pow1': [t_series_df_Z['power'][curr_peak_Z_idx]], 'pow2': [t_series_df_hor['power'][curr_peak_hor_idx]], 
                                                'slow1': [t_series_df_Z['slowness'][curr_peak_Z_idx]], 'slow2': [t_series_df_hor['slowness'][curr_peak_hor_idx]], 
                                                'bazi1': [t_series_df_Z['back_azi'][curr_peak_Z_idx]], 'bazi2': [t_series_df_hor['back_azi'][curr_peak_hor_idx]]})
                    events_df = events_df.append(curr_event_df)

        # And filter events to only output events with max. power within the max. phase window, 
        # if speficifed by user:
        if self.filt_phase_assoc_by_max_power:
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
                    if obspy.UTCDateTime(row['t1']) - obspy.UTCDateTime(tmp_df['t1'].iloc[0]) < self.max_phase_sep_s:
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
            # And output df:
            events_df = filt_events_df.copy()
            del filt_events_df
            gc.collect()


        return events_df
    
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

            # Combine horizontals:
            t_series_df_hor = t_series_df_N.copy()
            t_series_df_hor["power"] = np.sqrt(t_series_df_N["power"].values**2 + t_series_df_E["power"].values**2)
            NE_Pxx_max = np.max(np.concatenate((t_series_df_N["power"].values, t_series_df_E["power"].values)))
            N_weighting = t_series_df_N["power"].values / NE_Pxx_max
            E_weighting = t_series_df_E["power"].values / NE_Pxx_max
            t_series_df_hor["slowness"] = np.average(np.vstack((t_series_df_N['slowness'], t_series_df_E['slowness'])), 
                                                        axis=0, weights=np.vstack((N_weighting, 
                                                        E_weighting))) # Weighted mean (weighted by power)
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
            events_df = self._phase_associator(t_series_df_Z, t_series_df_hor, peaks_Z, peaks_hor)
            # Plot detected, phase-associated picks:
            if verbosity > 1:
                print("="*40)
                print("Event phase associations:")            
                print(events_df)
                print("="*40)
                plt.figure()
                plt.plot(t_series_df_Z['t'], t_series_df_Z['power'])
                plt.plot(t_series_df_hor['t'], t_series_df_hor['power'])
                plt.scatter(events_df['t1'], np.ones(len(events_df))*np.max(t_series_df_Z['power']), c='r')
                plt.scatter(events_df['t2'], np.ones(len(events_df))*np.max(t_series_df_Z['power']), c='b')
                plt.show()

            return events_df


#----------------------------------------------- End: Define main functions -----------------------------------------------


