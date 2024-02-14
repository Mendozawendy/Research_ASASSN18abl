"""
Photometer class

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

from plotter import *
plotter = Plotter()

class Photometer:

	@staticmethod
	def make_lightcurve(source_id, time_list, plot=True):

		dat_list = []
		mag_list = []
		err_list = []

		for item in glob.glob("*.dat"):

			if item == "stack-table.dat":

				pass

			else:

				dat_list.append(item)

		dat_list = sorted(dat_list)

		for item in dat_list:

			table = ascii.read(item)

			for line in table:

				if line[3] == source_id:

					mag = line[115]
					mag = float(mag)
					mag_list.append(mag)

					err = line[110]
					err = float(err)
					err_list.append(err)

		time_array = np.asarray(time_list)
		mag_array = np.asarray(mag_list)
		err_array = np.asarray(err_list)

		return time_array, mag_array, err_array

	@staticmethod
	def measure_target(object_frame, alpha, delta, radius, mask, boxes, mean, std, exptime):

		print("[PHOTOMETER][measure_target]: Running photometry at position", (alpha, delta))

		object_name = object_frame[:-4]
		frame = fits.open(object_frame)
		frame_data = frame[0].data
		frame_header = frame[0].header

		wcs = WCS(frame_header)
		sky_position = SkyCoord(alpha, delta, unit="deg")
		pix_position = skycoord_to_pixel(sky_position, wcs=wcs)

		xp = pix_position[0]
		yp = pix_position[1]

		pix_coord_tuple = (xp, yp)

		aperture = CircularAperture(pix_coord_tuple, r=radius)
		annulus = CircularAnnulus(pix_coord_tuple, r_in=radius+5, r_out=radius+8)
		apers = [aperture, annulus]

		phot_table = aperture_photometry(frame_data, apers, mask=mask, wcs=wcs)

		aperture_area = aperture.area
		annulus_area = annulus.area

		bkg_mean = phot_table["aperture_sum_1"] / annulus_area
		phot_table["annulus_mean"] = bkg_mean

		bkg_sum = bkg_mean * aperture_area
		phot_table["aperture_bkg_sum"] = bkg_sum

		final_sum = phot_table["aperture_sum_0"] - bkg_sum
		phot_table["res_aperture_sum"] = final_sum

		if phot_table["res_aperture_sum"] <= mean:

			flux_flag = True
			flux = mean

		else:

			flux_flag = False
			flux = final_sum

		flux_error = math.sqrt(flux + (aperture_area * (1 + (math.pi * aperture_area) / (2 * annulus_area))*(std**2)))
		inst_mag = -2.5 * math.log10(flux)
		corr_inst_mag = inst_mag + (2.5 * math.log10(exptime))
		inst_mag_error = (2.5 * flux_error) / (math.log(10) * flux)
		log_inst_mag_error = math.log10(inst_mag_error)

		phot_table["flux"] = flux
		phot_table["flux_flag"] = flux_flag
		phot_table["flux_error"] = flux_error
		phot_table["inst_mag"] = inst_mag
		phot_table["corr_inst_mag"] = corr_inst_mag
		phot_table["inst_mag_error"] = inst_mag_error
		phot_table["log_inst_mag_error"] = log_inst_mag_error

		for item in phot_table:

			flux = item["flux"]
			flux_error = item["flux_error"]
			inst_mag = item["inst_mag"]
			corr_inst_mag = item["corr_inst_mag"]
			inst_mag_error = item["inst_mag_error"]
			log_inst_mag_error = item["log_inst_mag_error"]

		return flux, flux_error, inst_mag, corr_inst_mag, inst_mag_error, log_inst_mag_error

	@staticmethod
	def make_table(object_frame, query_table, radius, mask, boxes, mean, std, exptime, plot=True):

		params = {}

		object_name = object_frame[:-4]
		frame = fits.open(object_frame)
		frame_data = frame[0].data
		frame_header = frame[0].header

		wcs = WCS(frame_header)

		sky_coord_list = []
		for item in query_table:

			ra = item["ra"]
			dec = item["dec"]
			coord_tuple = (ra, dec)
			sky_coord_list.append(coord_tuple)

		sky_positions = SkyCoord(sky_coord_list, unit="deg")
		pix_positions = skycoord_to_pixel(sky_positions, wcs=wcs)

		pix_coord_list = []
		for px in range(len(pix_positions[0])):

			xp = pix_positions[0][px]
			yp = pix_positions[1][px]

			coord_tuple = (xp, yp)
			pix_coord_list.append(coord_tuple)

		apertures = CircularAperture(pix_coord_list, r=radius)
		annuli = CircularAnnulus(pix_coord_list, r_in=radius+5, r_out=radius+8)

		apers = [apertures, annuli]

		phot_table = aperture_photometry(frame_data, apers, mask=mask, wcs=wcs)

		aperture_area = apertures.area
		annulus_area = annuli.area

		bkg_mean = phot_table["aperture_sum_1"] / annulus_area
		phot_table["annulus_mean"] = bkg_mean

		bkg_sum = bkg_mean * aperture_area
		phot_table["aperture_bkg_sum"] = bkg_sum

		final_sum = phot_table["aperture_sum_0"] - bkg_sum
		phot_table["res_aperture_sum"] = final_sum

		source_table = hstack([query_table, phot_table])

		flux_list = []
		flux_flag_list = []
		flux_error_list = []

		inst_mag_list = []
		corr_inst_mag_list = []
		inst_mag_error_list = []
		log_inst_mag_error_list = []

		for item in source_table:

			if item["res_aperture_sum"] <= mean:

				flux_flag_list.append(True)
				flux = mean

			else:

				flux_flag_list.append(False)
				flux = item["res_aperture_sum"]

			flux_list.append(flux)

			flux_error = math.sqrt(flux + (aperture_area * (1 + (math.pi * aperture_area) / (2 * annulus_area))*(std**2)))
			flux_error_list.append(flux_error)

			inst_mag = -2.5 * math.log10(flux)
			inst_mag_list.append(inst_mag)

			corr_inst_mag = inst_mag + (2.5 * math.log10(exptime))
			corr_inst_mag_list.append(corr_inst_mag)

			inst_mag_error = (2.5 * flux_error) / (math.log(10) * flux)
			inst_mag_error_list.append(inst_mag_error)

			log_inst_mag_error = math.log10(inst_mag_error)
			log_inst_mag_error_list.append(log_inst_mag_error)

		source_table["flux"] = flux_list
		source_table["flux_flag"] = flux_flag_list
		source_table["flux_error"] = flux_error_list
		source_table["inst_mag"] = inst_mag_list
		source_table["corr_inst_mag"] = corr_inst_mag_list
		source_table["inst_mag_error"] = inst_mag_error_list
		source_table["log_inst_mag_error"] = log_inst_mag_error_list

		# --- Calculate colors and delta magnitudes
		color_list = []
		color_flag_list = []
		delta_mag_list = []

		for item in source_table:

			if item["phot_bp_n_obs"] == 0 or item["phot_rp_n_obs"] == 0:

				color_flag_list.append(True)
				
				color = 0

			else:

				color_flag_list.append(False)

				b = item["phot_bp_mean_mag"]
				b = b.unmasked.value

				r = item["phot_rp_mean_mag"]
				r = r.unmasked.value

				color = b - r

			color_list.append(color)

			inst_mag = item["corr_inst_mag"]

			cat_mag = item["phot_g_mean_mag"]
			cat_mag = cat_mag.unmasked.value

			delta_mag = inst_mag - cat_mag
			delta_mag_list.append(delta_mag)

		source_table["color"] = color_list
		source_table["color_flag"] = color_flag_list
		source_table["delta_mag"] = delta_mag_list

		# --- Calculate transform and zero point
		color_list = []
		delta_mag_list = []

		test_error_list = []

		for item in source_table:

			if item["flux_flag"] == True or item["color_flag"] == True:

				pass

			else:

				error = item["log_inst_mag_error"]
				test_error_list.append(error)

				color = item["color"]
				color_list.append(color)

				delta_mag = item["delta_mag"]
				delta_mag_list.append(delta_mag)

		sigma_clip = 1.5

		# --- Original lists
		yfit, slope, intercept, delta_slope, delta_intercept = Photometer.unweighted_fit(color_list, delta_mag_list)

		# --- Sigma clip 1
		color_list_1, delta_mag_list_1 = Photometer.sigma_clip(color_list, delta_mag_list, sigma=sigma_clip)
		yfit_1, slope_1, intercept_1, delta_slope_1, delta_intercept_1 = Photometer.unweighted_fit(color_list_1, delta_mag_list_1)

		# --- Sigma clip 2
		color_list_2, delta_mag_list_2 = Photometer.sigma_clip(color_list_1, delta_mag_list_1, sigma=sigma_clip)
		yfit_2, slope_2, intercept_2, delta_slope_2, delta_intercept_2 = Photometer.unweighted_fit(color_list_2, delta_mag_list_2)

		# --- Sigma clip 3
		color_list_3, delta_mag_list_3 = Photometer.sigma_clip(color_list_2, delta_mag_list_2, sigma=sigma_clip)
		yfit_3, slope_3, intercept_3, delta_slope_3, delta_intercept_3 = Photometer.unweighted_fit(color_list_3, delta_mag_list_3)

		# --- Sigma clip 4
		color_list_4, delta_mag_list_4 = Photometer.sigma_clip(color_list_3, delta_mag_list_3, sigma=sigma_clip)
		yfit_4, slope_4, intercept_4, delta_slope_4, delta_intercept_4 = Photometer.unweighted_fit(color_list_4, delta_mag_list_4)

		# --- Sigma clip 5
		color_list_5, delta_mag_list_5 = Photometer.sigma_clip(color_list_4, delta_mag_list_4, sigma=sigma_clip)
		yfit_5, slope_5, intercept_5, delta_slope_5, delta_intercept_5 = Photometer.unweighted_fit(color_list_5, delta_mag_list_5)

		"""
		print("Original:", slope, "pm", delta_slope)
		print("         ", intercept, "pm", delta_intercept)
		print("1st clip:", slope_1, "pm", delta_slope_1)
		print("         ", intercept_1, "pm", delta_intercept_1)
		print("2nd clip:", slope_2, "pm", delta_slope_2)
		print("         ", intercept_2, "pm", delta_intercept_2)
		print("3rd clip:", slope_3, "pm", delta_slope_3)
		print("         ", intercept_3, "pm", delta_intercept_3)
		print("4th clip:", slope_4, "pm", delta_slope_4)
		print("         ", intercept_4, "pm", delta_intercept_4)
		print("5th clip:", slope_5, "pm", delta_slope_5)
		print("         ", intercept_5, "pm", delta_intercept_5)
		"""

		"""
		reduced_mag_list = []

		for item in source_table:

			inst_mag = item["corr_inst_mag"]
			color = item["color"]

			reduced_mag = inst_mag - (transform * color) - eff_zp
			reduced_mag_list.append(reduced_mag)

		source_table["reduced_mag"] = reduced_mag_list
		"""

		params["transform"] = slope
		params["delta_transform"] = delta_slope
		params["eff_zp"] = intercept
		params["delta_eff_zp"] = delta_intercept

		source_table.write(object_name + "-table.dat", format="ascii.fixed_width", overwrite=True)

		if plot == True:

			plotter.plot_histogram(object_name, corr_inst_mag_list)
			plotter.plot_histogram("test", delta_mag_list)
			plotter.plot_field(object_name, frame_data, apertures, boxes)

			plotter.plot_magerr(object_name, corr_inst_mag_list, log_inst_mag_error_list)
			plotter.plot_magerr("test", delta_mag_list, test_error_list)

			plotter.plot_colormag(object_name, color_list, delta_mag_list, yfit)

			"""
			plotter.plot_colormag("Clip 1", color_list_1, delta_mag_list_1, yfit_1)
			plotter.plot_colormag("Clip 2", color_list_2, delta_mag_list_2, yfit_2)
			plotter.plot_colormag("Clip 3", color_list_3, delta_mag_list_3, yfit_3)
			plotter.plot_colormag("Clip 4", color_list_4, delta_mag_list_4, yfit_4)
			plotter.plot_colormag("Clip 5", color_list_5, delta_mag_list_5, yfit_5)
			"""

		return source_table, params

	@staticmethod
	def sigma_clip(x_list, y_list, sigma):

		#print("[PHOTOMETER][sigma_clip]: Clipping arrays")

		x_array = np.asarray(x_list)
		y_array = np.asarray(y_list)

		mean_x = np.mean(x_array)
		mean_y = np.mean(y_array)

		std_x = np.std(x_array)
		std_y = np.std(y_array)

		new_x_list = []
		new_y_list = []

		for i in range(len(x_list)):

			if y_list[i] < (mean_y - (sigma * std_y)):

				pass

			else:

				new_x_list.append(x_list[i])
				new_y_list.append(y_list[i])

		return new_x_list, new_y_list


	@staticmethod
	def unweighted_fit(x_list, y_list):

		#print("[PHOTOMETER][unweighted_fit]: Calculating unweighted linear fit")

		x_array = np.asarray(x_list)
		y_array = np.asarray(y_list)

		def f(x, m, b):
			y = (m*x) + b
			return y

		popt, pcov = curve_fit(f, x_array, y_array)

		yfit = f(x_array, *popt)

		slope = popt[0]
		delta_slope = math.sqrt(pcov[0][0])

		intercept = popt[1]
		delta_intercept = math.sqrt(pcov[1][1])

		return yfit, slope, intercept, delta_slope, delta_intercept