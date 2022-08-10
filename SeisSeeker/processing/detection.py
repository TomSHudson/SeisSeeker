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
import os, sys
import obspy
from scipy.signal import find_peaks
from numba import jit, prange, set_num_threads
import gc 


#----------------------------------------------- Define main functions -----------------------------------------------
class CustomError(Exception):
    pass

#@jit(nopython=True)
def _fast_freq_domain_array_proc(data, max_sl, fs, target_freqs, xx, yy, n_stations, n_t_samp, remove_autocorr):
    """Function to perform array processing fast due to being designed to 
    be wrapped using Numba.
    Returns:
    Pfreq_all
    """
    # Create data stores:
    Pfreq_all = []

    # Define grid of slownesses:
    # number of pixes in x and y
    # (Determines number of phase shifts to perform)
    nux = 101
    nuy = 101
    ux = np.linspace(-max_sl,max_sl,nux)
    uy = np.linspace(-max_sl,max_sl,nuy)
    dux=ux[1]-ux[0]
    duy=uy[1]-uy[0]

    # To speed things up, we precompute a library of time shifts,
    #  so we don't have to do it for each loop of frequency:
    tlib=np.zeros([n_stations,nux,nuy],dtype=float)
    for ix in range(0,nux):
            for iy in range(0,nuy):
                tlib[:,ix,iy] = xx*ux[ix] + yy*uy[iy]
    # In this case, we assume station locations xx and yy are already relative to some convenient midpoint
    # Rather than shift one station to a single other station, we'll shift *all* stations back to that midpoint

    # Then loop over windows:
    for win_idx in range(data.shape[0]):
        if(win_idx % 10 == 0):
            print("Processing for window", win_idx+1, "/", data.shape[0])
        
        # Calculate spectra:
        # Construct data structure:
        nfft = (2.0**np.ceil(np.log2(n_t_samp))).astype(int)
        Pxx_all = np.zeros((np.int((nfft/2)+1), n_stations), dtype=complex) # Power spectra
        dt = 1. / fs 
        df = 1.0/(2.0*nfft*dt)
        xf = np.linspace(0.0, 1.0/(2.0*dt), np.int((nfft/2)+1))
        # Calculate power spectra for all stations:
        for sta_idx in range(n_stations):
            # Calculate spectra for current station:
            Pxx_all[:,sta_idx] = np.fft.rfft(data[win_idx,sta_idx,:],n=nfft) # (Use real fft, as input data is real)

        # Loop over all freqs, performing phase shifts:
        Pfreq=np.zeros([len(target_freqs),nux,nuy],dtype=complex)
        counter_grid = 0
        for ii in range(len(target_freqs)):
            # Find closest current freq.:
            target_f = target_freqs[ii]
            curr_f_idx = (np.abs(xf - target_f)).argmin()

            # Construct a matrix of each station-station correlation *before* any phase shifts
            Rxx=np.zeros([n_stations,n_stations],dtype=complex)
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
                    timeshifts = tlib[:,ix,iy]
                    # Calculate the "steering vector" (a vector in frequency space, based on phase-shift)
                    a = np.exp(-1j*2*np.pi*target_f*timeshifts)
                    aconj = np.conj(a)
                    # "a" is a "steering vector." It contains info on all the phase delays needed to 
                    #  push our stations to the middle point.
                    # Since each element of Rxx contains cross-spectra of two stations, we need two timeshifts
                    #  to push both to the centerpoint. This can also be seen as projecting Rxx onto a new basis.
                    Pfreq[ii,ix,iy]=np.dot(np.dot(aconj,Rxx),a)     
        # And remove any data where stations don't exist:
        Pfreq = np.nan_to_num(Pfreq, copy=False, nan=0.0)
                    
        # And append output to datastore:
        Pfreq_all.append(Pfreq)

    return Pfreq_all 


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
                in analysis. Default is 100.

    max_sl : float
        Maximum slowness to analyse for, in s / km. Default is 1.0 s / km.

    win_len_s : float
        The window length for each frequency domain step. Units are seconds.
        Default value is 0.1 s.

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

        # Setup station information:
        self.stations_df = pd.read_csv(self.stations_fname)

        # And define attributes:
        self.freqmin = None
        self.freqmax = None
        self.num_freqs = 100
        self.max_sl = 1.0
        self.win_len_s = 0.1
        self.remove_autocorr = True


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
                print("="*60)
                print("Processing data for year "+str(year)+", day "+str(julday).zfill(3))

                # Loop over channels:
                for self.channel_curr in self.channels_to_use:

                    # Load data:
                    st = self._load_day_of_data(year, julday)

                    # Convert data to np:
                    data = self._convert_st_to_np_data(st)
                    del st 
                    gc.collect()

                    # Run heavy array processing algorithm:
                    # Specify various variables needed:
                    # Make a linear spacing of frequencies. One might use periods, logspacing, etc.:
                    target_freqs = np.linspace(self.freqmin,self.freqmax,self.num_freqs) #np.logspace(f_min,f_max,num_freqs) #np.linspace(f_min,f_max,num_freqs)
                    # Station locations:
                    xx = self.stations_df['x_array_coords_km'].values
                    yy = self.stations_df['y_array_coords_km'].values
                    # And run
                    Pfreq_all = _fast_freq_domain_array_proc(data, self.max_sl, self.fs, target_freqs, xx, yy, self.n_stations, self.n_t_samp, self.remove_autocorr)


                    # Sum/stack/normallise data:


                    # Calculate detection outputs from data:

        return Pfreq_all

















#----------------------------------------------- End: Define main functions -----------------------------------------------



