"""
Calculator class

"""

import ccdproc
import configparser
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import time
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import ascii, fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.table import hstack, Table
from astropy.time import Time
from astropy.visualization import LinearStretch, MinMaxInterval
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astroquery.gaia import Gaia
from photutils import make_source_mask
from photutils.aperture import aperture_photometry, CircularAnnulus, CircularAperture, RectangularAperture, SkyCircularAnnulus, SkyCircularAperture
from photutils.background import Background2D, MedianBackground
from reproject import reproject_interp
from scipy.optimize import curve_fit

class Calculator:

	@staticmethod
	def calculate_airmass(location, time, ra, dec):

		aa = AltAz(location=location, obstime=time)

		coord = SkyCoord(str(ra), str(dec), unit="deg")
		coord_transform = coord.transform_to(aa)

		altitude = coord_transform.alt.degree
		zenith_distance = 90 - altitude
		airmass = 1 / (math.cos(math.radians(zenith_distance)) + 0.50572*((6.07995 + 90 - zenith_distance)**-1.6364)) # Kasten and Young (1994) method

		return airmass, zenith_distance, altitude