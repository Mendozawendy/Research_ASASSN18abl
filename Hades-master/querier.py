"""
Querier class

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

class Querier:

	@staticmethod
	def submit_query(config, survey):

		ra = config["Photometry"]["alpha"]
		ra = str(ra)

		dec = config["Photometry"]["delta"]
		dec = str(dec)

		radius = config["Photometry"]["query_radius"]

		if survey == "gaia":

			print("[QUERIER][submit_query]: Querying Gaia DR2 at", (ra, dec))

			search_coord = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame="fk5")
			search_radius = u.Quantity(radius, u.deg)

			Gaia.ROW_LIMIT = -1

			search = Gaia.cone_search_async(search_coord, search_radius)

			table = search.get_results()

		return table