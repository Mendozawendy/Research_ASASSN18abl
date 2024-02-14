"""
Reader class

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

class Reader:

	@staticmethod
	def config_camera(config, verbose=False):

		name = config["Camera"]["name"]

		print("[READER][config_camera]: Reading camera information")

		params = {}

		if name == "PL16803":

			dx = 4096
			dy = 4096
			gain = 0.72
			inverse_gain = 1.39
			read_noise = 11.4

		elif name == "ST8300":

			dx = 3352
			dy = 2532
			gain = 2.48
			inverse_gain = 0.403
			read_noise = 28.5

		params["dx"] = dx
		params["dy"] = dy
		params["gain"] = gain
		params["inverse_gain"] = inverse_gain
		params["read_noise"] = read_noise

		if verbose == True:

			print("[PIPELINE][config_camera]:", name)
			print("                           Camera:", name)
			print("                           Dimensions:", dx, "x", dy)
			print("                           Gain:", gain, "ADU/e")
			print("                           Inverse gain:", inverse_gain, "e/ADU")
			print("                           Read noise:", read_noise, "ADU")

		return params

	@staticmethod
	def get_location(config):

		latitude = config["Location"]["latitude"]
		longitude = config["Location"]["longitude"]
		height = config["Location"]["height"]

		location = EarthLocation(lat=latitude, lon=longitude, height=height)

		return location

	@staticmethod
	def read_frame(config, object_frame):

		sigma = config["Preprocessing"]["sigma"]
		sigma = float(sigma)

		params = {}

		object_name = object_frame[:-4]

		frame = fits.open(object_frame)
		frame_data = frame[0].data
		frame_header = frame[0].header

		wcs = WCS(frame_header)

		dateobs = frame_header["DATE-OBS"]
		jd = frame_header["JD"]
		obs_time = Time(dateobs)

		ra = frame_header["CRVAL1"]
		dec = frame_header["CRVAL2"]
		x = frame_header["CRPIX1"]
		y = frame_header["CRPIX2"]

		exptime = frame_header["EXPTIME"]

		mean, median, std = sigma_clipped_stats(frame_data, sigma=sigma)

		params["dateobs"] = dateobs
		params["jd"] = jd
		params["ra"] = ra
		params["dec"] = dec
		params["x"] = x
		params["y"] = y
		params["exptime"] = exptime
		params["mean"] = mean
		params["median"] = median
		params["std"] = std

		return params