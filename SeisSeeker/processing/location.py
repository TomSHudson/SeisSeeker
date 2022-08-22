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



#----------------------------------------------- Define main functions -----------------------------------------------
class CustomError(Exception):
    pass







#----------------------------------------------- End: Define main functions -----------------------------------------------



