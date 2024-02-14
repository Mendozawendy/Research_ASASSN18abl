"""
Reducer class

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

class Reducer:

	@staticmethod
	def align_frames(config):

		obj_dir = config["Test"]["obj_dir"]

		align_list = []

		os.chdir(obj_dir)

		for item in glob.glob("wcs*.fit"):

			align_list.append(item)

		align_list = sorted(align_list)

		print("[REDUCER][align_frames]: Saving reference frame", align_list[0])

		reference_frame = fits.open(align_list[0])
		reference_data = reference_frame[0].data
		reference_header = reference_frame[0].header

		reference_hdu = fits.PrimaryHDU(reference_data, header=reference_header)
		reference_hdu.writeto(obj_dir + "/a-" + str(align_list[0]), overwrite=True)

		for i in range(1, len(align_list)):

			if os.path.isfile("a-" + align_list[i]):

				print("[REDUCER][align_frames]: Skipping align on", align_list[i])

			else:

				print("[REDUCER][align_frames]: Aligning", align_list[i], "to reference frame")

				target_frame = fits.open(align_list[i])
				target_data = target_frame[0].data
				target_header = target_frame[0].header

				target_hdu = fits.PrimaryHDU(target_data, header=target_header)
				array, footprint = reproject_interp(target_hdu, reference_header)

				aligned_hdu = fits.PrimaryHDU(array, header=target_header)
				aligned_hdu.writeto(obj_dir + "/a-" + str(align_list[i]))

		return align_list

	@staticmethod
	def extract_sources(object_frame, inverse_gain):

		print("[REDUCER][extract_sources]: Extracting sources on", object_frame)

		object_name = object_frame[:-4]

		Reducer.make_sextractor_conv()
		Reducer.make_sextractor_nnw()
		Reducer.make_sextractor_param()
		Reducer.make_sextractor_sex(object_name, inverse_gain)

		subprocess.run(["source-extractor", object_frame])
		subprocess.run(["rm", "default.conv"])
		subprocess.run(["rm", "default.nnw"])

		seeing_pix_list = []
		seeing_sky_list = []
		growth_radius_list = []

		catalog = open(object_name + ".cat")

		for line in catalog:
			line = line.split()

			if line[0] == "#":

				pass

			else:

				if line[1] == "0.0000000" or line[2] == "+0.0000000":

					pass

				else:

					seeing_pix = float(line[8])
					seeing_pix_list.append(seeing_pix)

					seeing_sky = float(line[9]) * 3600
					seeing_sky_list.append(seeing_sky)

					growth_radius = float(line[7])
					growth_radius_list.append(growth_radius)

		mean_seeing_pix = np.mean(seeing_pix_list)
		mean_seeing_sky = np.mean(seeing_sky_list)
		mean_growth_radius = np.mean(growth_radius_list)

		return mean_seeing_pix, mean_seeing_sky, mean_growth_radius

	@staticmethod
	def make_dark(config, frame_set):
		"""Combine a series of dark frames into a master dark"""

		if frame_set == "flat":
			dark_dir = config["Test"]["dark_flat_dir"]

		elif frame_set == "object":
			dark_dir = config["Test"]["dark_obj_dir"]

		method = config["Preprocessing"]["combine_method"]
		dtype = config["Preprocessing"]["dtype"]
		mem_limit = config["Preprocessing"]["mem_limit"]
		mem_limit = float(mem_limit)

		dark_path = dark_dir + "/master-dark.fit"

		if os.path.isfile(dark_path):
			
			print("[REDUCER][make_dark]: Reading extant master dark")
			master_dark = fits.open(dark_path)

		else:

			print("[REDUCER][make_dark]: Combining darks by", method)
			os.chdir(dark_dir)
			dark_list = []

			for item in glob.glob("*.fit"):

				dark_list.append(item)

			master_dark = ccdproc.combine(dark_list, method=method, unit="adu", mem_limit=mem_limit, dtype=dtype)

			ccdproc.fits_ccddata_writer(master_dark, dark_path, overwrite=True)

		return master_dark

	@staticmethod
	def make_flat(config):
		"""Combine a series of flat frames into a normalized flatfield"""

		flat_dir = config["Test"]["flat_dir"]
		dark_dir = config["Test"]["dark_flat_dir"]

		method = config["Preprocessing"]["combine_method"]
		dtype = config["Preprocessing"]["dtype"]
		mem_limit = config["Preprocessing"]["mem_limit"]
		mem_limit = float(mem_limit)

		flat_path = flat_dir + "/flatfield.fit"
		dark_path = dark_dir + "/master-dark.fit"

		if os.path.isfile(flat_path):

			print("[REDUCER][make_flat]: Reading extant flatfield")
			flatfield = fits.open(flat_path)

		else:

			print("[REDUCER][make_flat]: Combining flats by", method)
			master_dark = ccdproc.fits_ccddata_reader(dark_path)

			os.chdir(flat_dir)
			flat_list = []

			for item in glob.glob("*.fit"):
				flat_list.append(item)

			combined_flat = ccdproc.combine(flat_list, method=method, unit="adu", mem_limit=mem_limit, dtype=dtype)

			flat_exposure = combined_flat.header["exposure"]*u.second
			dark_exposure = master_dark.header["exposure"]*u.second
			master_flat = ccdproc.subtract_dark(combined_flat, master_dark, data_exposure=flat_exposure, dark_exposure=dark_exposure)

			master_flat_data = np.asarray(master_flat)
			flatfield_data = master_flat_data / np.mean(master_flat_data)
			flatfield = ccdproc.CCDData(flatfield_data, unit="adu")

			ccdproc.fits_ccddata_writer(flatfield, flat_path)

		return flatfield

	@staticmethod
	def make_mask(config, object_frame):

		camera_name = config["Camera"]["name"]

		print("[REDUCER][make_mask]: Constructing mask for", object_frame)

		object_name = object_frame[:-4]

		frame = fits.open(object_frame)
		frame_data = frame[0].data
		frame_header = frame[0].header

		mask = np.zeros(frame_data.shape, dtype=bool)

		edge = 100

		if camera_name == "PL16803":

			dx = frame_data.shape[0]
			dy = frame_data.shape[1]

			# --- Bad pixel mask
			mask_y1 = 860
			mask_y2 = 4089
			mask_delta_y = mask_y2 - mask_y1
			center_y = mask_delta_y // 2

			mask_x1 = 1200
			mask_x2 = 1210
			mask_delta_x = mask_x2 - mask_x1
			center_x = mask_delta_x // 2

			mask[mask_y1:mask_y2, mask_x1:mask_x2] = True

		elif camera_name == "ST8300":

			dx = frame_data.shape[1]
			dy = frame_data.shape[0]

		# --- Left edge mask
		eml_center_x = edge // 2
		eml_center_y = dy // 2
		mask[0:dy, 0:edge] = True

		# --- Top edge mask
		emt_center_x = dx // 2
		emt_center_y = edge // 2
		mask[dy-edge:dy, 0:dx] = True

		# --- Right edge mask
		emr_center_x = edge // 2
		emr_center_y = dy // 2
		mask[0:dy, dx-edge:dx] = True

		# --- Bottom edge mask
		emb_center_x = dx // 2
		emb_center_y = edge // 2
		mask[0:edge, 0:dx] = True

		boxes = {}

		#mask_box = RectangularAperture((center_x + mask_x1, center_y + mask_y1), mask_delta_x, mask_delta_y, theta=0.)
		eml_box = RectangularAperture((eml_center_x, eml_center_y), edge, dy, theta=0.)
		emt_box = RectangularAperture((emt_center_x, emt_center_y + dy - edge), dx, edge, theta=0.)
		emr_box = RectangularAperture((emr_center_x + dx - edge, emr_center_y), edge, dy, theta=0.)
		emb_box = RectangularAperture((emb_center_x, emb_center_y), dx, edge, theta=0.)

		#boxes["mask_box"] = mask_box
		boxes["eml_box"] = eml_box
		boxes["emt_box"] = emt_box
		boxes["emr_box"] = emr_box
		boxes["emb_box"] = emb_box

		return mask, boxes

	@staticmethod
	def make_sextractor_conv():

		conv_norm_1 = "CONV NORM\n"
		conv_norm_2 = "# 3x3 ``all-ground'' convolution mask with FWHM = 2 pixels.\n"
		conv_norm_3 = "1 2 1\n"
		conv_norm_4 = "2 4 2\n"
		conv_norm_5 = "1 2 1"

		conv_norm_lines = [conv_norm_1, conv_norm_2, conv_norm_3, conv_norm_4, conv_norm_5]

		default_conv = open("default.conv", "w")

		for line in conv_norm_lines:

			default_conv.write(line)

		default_conv.close()

		return

	@staticmethod
	def make_sextractor_nnw():

		nnw_1 = "NNW\n"
		nnw_2 = "# Neural Network Weights for the SExtractor star/galaxy classifier (V1.3)\n"
		nnw_3 = "# inputs:  9 for profile parameters + 1 for seeing.\n"
		nnw_4 = "# outputs: ``Stellarity index'' (0.0 to 1.0)\n"
		nnw_5 = "# Seeing FWHM range: from 0.025 to 5.5'' (images must have 1.5 < FWHM < 5 pixels)\n"
		nnw_6 = "# Optimized for Moffat profiles with 2<= beta <= 4.\n\n"
		nnw_7 = " 3 10 10 1\n\n"
		nnw_8 = "-1.56604e+00 -2.48265e+00 -1.44564e+00 -1.24675e+00 -9.44913e-01 -5.22453e-01  4.61342e-02  8.31957e-01  2.15505e+00  2.64769e-01\n"
		nnw_9 = " 3.03477e+00  2.69561e+00  3.16188e+00  3.34497e+00  3.51885e+00  3.65570e+00  3.74856e+00  3.84541e+00  4.22811e+00  3.27734e+00\n\n"
		nnw_10 = "-3.22480e-01 -2.12804e+00  6.50750e-01 -1.11242e+00 -1.40683e+00 -1.55944e+00 -1.84558e+00 -1.18946e-01  5.52395e-01 -4.36564e-01 -5.30052e+00\n"
		nnw_11 = " 4.62594e-01 -3.29127e+00  1.10950e+00 -6.01857e-01  1.29492e-01  1.42290e+00  2.90741e+00  2.44058e+00 -9.19118e-01  8.42851e-01 -4.69824e+00\n"
		nnw_12 = "-2.57424e+00  8.96469e-01  8.34775e-01  2.18845e+00  2.46526e+00  8.60878e-02 -6.88080e-01 -1.33623e-02  9.30403e-02  1.64942e+00 -1.01231e+00\n"
		nnw_13 = " 4.81041e+00  1.53747e+00 -1.12216e+00 -3.16008e+00 -1.67404e+00 -1.75767e+00 -1.29310e+00  5.59549e-01  8.08468e-01 -1.01592e-02 -7.54052e+00\n"
		nnw_14 = " 1.01933e+01 -2.09484e+01 -1.07426e+00  9.87912e-01  6.05210e-01 -6.04535e-02 -5.87826e-01 -7.94117e-01 -4.89190e-01 -8.12710e-02 -2.07067e+01\n"
		nnw_15 = "-5.31793e+00  7.94240e+00 -4.64165e+00 -4.37436e+00 -1.55417e+00  7.54368e-01  1.09608e+00  1.45967e+00  1.62946e+00 -1.01301e+00  1.13514e-01\n"
		nnw_16 = " 2.20336e-01  1.70056e+00 -5.20105e-01 -4.28330e-01  1.57258e-03 -3.36502e-01 -8.18568e-02 -7.16163e+00  8.23195e+00 -1.71561e-02 -1.13749e+01\n"
		nnw_17 = " 3.75075e+00  7.25399e+00 -1.75325e+00 -2.68814e+00 -3.71128e+00 -4.62933e+00 -2.13747e+00 -1.89186e-01  1.29122e+00 -7.49380e-01  6.71712e-01\n"
		nnw_18 = "-8.41923e-01  4.64997e+00  5.65808e-01 -3.08277e-01 -1.01687e+00  1.73127e-01 -8.92130e-01  1.89044e+00 -2.75543e-01 -7.72828e-01  5.36745e-01\n"
		nnw_19 = "-3.65598e+00  7.56997e+00 -3.76373e+00 -1.74542e+00 -1.37540e-01 -5.55400e-01 -1.59195e-01  1.27910e-01  1.91906e+00  1.42119e+00 -4.35502e+00\n\n"
		nnw_20 = "-1.70059e+00 -3.65695e+00  1.22367e+00 -5.74367e-01 -3.29571e+00  2.46316e+00  5.22353e+00  2.42038e+00  1.22919e+00 -9.22250e-01 -2.32028e+00\n\n\n"
		nnw_21 = " 0.00000e+00\n"
		nnw_22 = " 1.00000e+00"

		default_nnw_lines = [nnw_1, nnw_2, nnw_3, nnw_4, nnw_5, nnw_6, nnw_7, nnw_8, nnw_9, nnw_10, nnw_11, nnw_12, nnw_13, nnw_14, nnw_15, nnw_16, nnw_17, nnw_18, nnw_19, nnw_20, nnw_21, nnw_22]

		default_nnw = open("default.nnw", "w")

		for line in default_nnw_lines:

		    default_nnw.write(line)

		default_nnw.close()

		return

	@staticmethod
	def make_sextractor_sex(object_name, inverse_gain):

		# Default configuration file for SExtractor 2.12.4
		# EB 2010-10-10

		catalog_name = ["CATALOG_NAME", object_name + ".cat"]
		catalog_type = ["CATALOG_TYPE", "ASCII_HEAD"]
		parameters_name = ["PARAMETERS_NAME", "default.param"]

		detect_type = ["DETECT_TYPE", "CCD"] # CCD (linear) or PHOTO (with gamma correction)
		detect_minarea = ["DETECT_MINAREA", "3"] # min. # of pixels above threshold
		detect_thresh = ["DETECT_THRESH", "5.0"] # <sigmas> or <threshold>,<ZP> in mag.arcsec-2
		analysis_thresh = ["ANALYSIS_THRESH", "5.0"] # <sigmas> or <threshold>,<ZP> in mag.arcsec-2
		detection_filter = ["FILTER", "Y"] # apply filter for detection (Y or N)?
		detection_filter_name = ["FILTER_NAME", "default.conv"] # name of the file containing the filter
		deblend_nthresh = ["DEBLEND_NTHRESH", "32"] # Number of deblending sub-thresholds
		deblend_mincont = ["DEBLEND_MINCONT", "0.005"] # Minimum contrast parameter for deblending
		clean = ["CLEAN", "Y"] # Clean spurious detections? (Y or N)?
		clean_param = ["CLEAN_PARAM", "1.0"] # Cleaning efficiency
		
		weight_type = ["WEIGHT_TYPE", "NONE"] # type of WEIGHTing: NONE, BACKGROUND, MAP_RMS, MAP_VAR or MAP_WEIGHT
		weight_image = ["WEIGHT_IMAGE", "weight.fits"] # weight-map filename
		
		flag_image = ["FLAG_IMAGE", "flag.fits"] # filename for an input FLAG-image
		flag_type = ["FLAG_TYPE", "OR"] # flag pixel combination: OR, AND, MIN, MAX or MOST
		
		phot_apertures = ["PHOT_APERTURES", "10"] # MAG_APER aperture diameter(s) in pixels
		phot_autoparams = ["PHOT_AUTOPARAMS", "2.5", "3.5"] # MAG_AUTO parameters: <Kron_fact>,<min_radius>
		phot_petroparams = ["PHOT_PETROPARAMS", "2.0", "3.5"] # MAG_PETRO parameters: <Petrosian_fact>,<min_radius>
		phot_autoapers = ["PHOT_AUTOAPERS", "0.0", "0.0"] # <estimation>,<measurement> minimum apertures for MAG_AUTO and MAG_PETRO
		satur_level = ["SATUR_LEVEL", "65535.0"] # level (in ADUs) at which arises saturation
		satur_key = ["SATUR_KEY", "SATURATE"] # keyword for saturation level (in ADUs)
		mag_zeropoint = ["MAG_ZEROPOINT", "0.0"] # magnitude zero-point
		mag_gamma = ["MAG_GAMMA", "4.0"] # gamma of emulsion (for photographic scans)
		gain = ["GAIN", str(inverse_gain)] # keyword for detector gain in e-/ADU
		gain_key = ["GAIN_KEY", "GAIN"] # keyword for detector gain in e-/ADU
		pixel_scale = ["PIXEL_SCALE", "0"] # size of pixel in arcsec (0=use FITS WCS info)
		
		seeing_fwhm = ["SEEING_FWHM", "3.2"] # stellar FWHM in arcsec
		starnnw_name = ["STARNNW_NAME", "default.nnw"] # Neural-Network_Weight table filename
		
		back_type = ["BACK_TYPE", "AUTO"] # AUTO or MANUAL
		back_value = ["BACK_VALUE", "0.0"] # Default background value in MANUAL mode
		back_size = ["BACK_SIZE", "64"] # Background mesh: <size> or <width>,<height>
		back_filtersize = ["BACK_FILTERSIZE", "3"] # Background filter: <size> or <width>,<height>
		
		checkimage_type = ["CHECKIMAGE_TYPE", "NONE"] # can be NONE, BACKGROUND, BACKGROUND_RMS, MINIBACKGROUND, MINIBACK_RMS, -BACKGROUND, FILTERED, OBJECTS, -OBJECTS, SEGMENTATION, or APERTURES
		checkimage_name = ["CHECKIMAGE_NAME", "check.fits"] # Filename for the check-image
		
		memory_objstack = ["MEMORY_OBJSTACK", "3000"] # number of objects in stack
		memory_pixstack = ["MEMORY_PIXSTACK", "300000"] # number of pixels in stack
		memory_bufsize = ["MEMORY_BUFSIZE", "1024"] # number of lines in buffer
		
		assoc_name = ["ASSOC_NAME", "sky.list"] # name of the ASCII file to ASSOCiate
		assoc_data = ["ASSOC_DATA", "2", "3", "4"] # columns of the data to replicate (0=all)
		assoc_params = ["ASSOC_PARAMS", "2", "3", "4"] # columns of xpos,ypos[,mag]
		assoc_radius = ["ASSOC_RADIUS", "2.0"] # cross-matching radius (pixels)
		assoc_type = ["ASSOC_TYPE", "NEAREST"] # ASSOCiation method: FIRST, NEAREST, MEAN, MAG_MEAN, SUM, MAG_SUM, MIN or MAX
		assocselec_type = ["ASSOCSELEC_TYPE", "MATCHED"] # ASSOC selection type: ALL, MATCHED or -MATCHED
		
		verbose_type = ["VERBOSE_TYPE", "NORMAL"] # can be QUIET, NORMAL or FULL
		header_suffix = ["HEADER_SUFFIX", ".head"] # Filename extension for additional headers
		write_xml = ["WRITE_XML", "N"]# Write XML file (Y/N)?
		xml_name = ["XML_NAME", "sex.xml"] # Filename for XML output
		xsl_url = ["XSL_URL", "file:///usr/local/share/sextractor/sextractor.xsl"] # Filename for XSL style-sheet

		config_parameters = [catalog_name, catalog_type, parameters_name, detect_type, detect_minarea, detect_thresh,
						analysis_thresh, detection_filter, detection_filter_name, deblend_nthresh, deblend_mincont, 
						clean, clean_param, weight_type, weight_image, flag_image, flag_type, phot_apertures, 
						phot_autoparams, phot_petroparams, phot_autoapers, satur_level, satur_key, mag_zeropoint,
						mag_gamma, gain, gain_key, pixel_scale, seeing_fwhm, starnnw_name, back_type, back_value,
						back_size, back_filtersize, checkimage_type, checkimage_name, memory_objstack, memory_pixstack,
						memory_bufsize, assoc_name, assoc_data, assoc_params, assoc_radius, assoc_type, assocselec_type,
						verbose_type, header_suffix, write_xml, xml_name, xsl_url]

		default_sex = open("default.sex", "w")

		for param in config_parameters:

			for item in param:

				default_sex.write(item + " ")

			default_sex.write("\n")

		default_sex.close()

		return

	@staticmethod
	def make_sextractor_param():

		param_number = "NUMBER"
		param_alphapeak_j2000 = "ALPHAPEAK_J2000"
		param_deltapeak_j2000 = "DELTAPEAK_J2000"
		param_xpeak_image = "XPEAK_IMAGE"
		param_ypeak_image = "YPEAK_IMAGE"
		param_flux_growth = "FLUX_GROWTH"
		param_fluxerr_best = "FLUXERR_BEST"
		param_flux_growthstep = "FLUX_GROWTHSTEP"
		param_fwhm_image = "FWHM_IMAGE"
		param_fwhm_world = "FWHM_WORLD"
		
		default_parameters = [param_number, param_alphapeak_j2000, param_deltapeak_j2000, param_xpeak_image, param_ypeak_image, param_flux_growth, param_fluxerr_best, param_flux_growthstep, param_fwhm_image, param_fwhm_world]
		
		default_param = open("default.param", "w")

		for param in default_parameters:

			default_param.write(param + "\n")

		default_param.close()

		return

	@staticmethod
	def make_stack(config):

		obj_dir = config["Test"]["obj_dir"]

		method = config["Preprocessing"]["combine_method"]
		dtype = config["Preprocessing"]["dtype"]
		mem_limit = config["Preprocessing"]["mem_limit"]
		mem_limit = float(mem_limit)

		stack_path = obj_dir + "/stack.fit"

		if os.path.isfile(stack_path):

			print("[REDUCER][make_stack]: Reading extant stack")
			stack = ccdproc.fits_ccddata_reader("stack.fit")

		else:

			print("[REDUCER][make_stack]: Combining frames by", method)

			os.chdir(obj_dir)
			stack_list = []

			for item in glob.glob("a-wcs-reduced-*.fit"):
				stack_list.append(item)

			stack_list = sorted(stack_list)

			stack = ccdproc.combine(stack_list, method=method, unit="adu", mem_limit=mem_limit, dtype=dtype)

			ccdproc.fits_ccddata_writer(stack, stack_path)

		return stack

	@staticmethod
	def reduce_objects(config):

		flat_dir = config["Test"]["flat_dir"]
		dark_dir = config["Test"]["dark_obj_dir"]
		obj_dir = config["Test"]["obj_dir"]

		sigma = config["Preprocessing"]["sigma"]
		sigma = float(sigma)
		npixels = config["Preprocessing"]["npixels"]
		npixels = int(npixels)
		dilate_size = config["Preprocessing"]["dilate_size"]
		dilate_size = int(dilate_size)

		flat_path = flat_dir + "/flatfield.fit"
		dark_path = dark_dir + "/master-dark.fit"

		obj_list = []

		os.chdir(obj_dir)

		for item in glob.glob("*.fit"):

			if "reduced" in item:
				pass
			elif "wcs-reduced" in item:
				pass
			elif "a-wcs-reduced" in item:
				pass
			elif "stack" in item:
				pass
			else:
				obj_list.append(item)

		obj_list = sorted(obj_list)

		for obj in obj_list:

			if os.path.isfile("reduced-" + obj):

				print("[REDUCER][reduce_objects]: Skipping reduction on", obj)

			else:

				print("[REDUCER][reduce_objects]: Reducing", obj)
				obj_frame = fits.open(obj)
				obj_frame_data = obj_frame[0].data
				obj_frame_header = obj_frame[0].header

				master_dark = fits.open(dark_path)
				master_dark_data = master_dark[0].data
				master_dark_header = master_dark[0].header
				reduced_obj_frame_data = obj_frame_data - master_dark_data

				flatfield = fits.open(flat_path)
				flatfield_data = flatfield[0].data
				flatfield_header = flatfield[0].header
				reduced_obj_frame_data /= flatfield_data

				sigma_clip = SigmaClip(sigma=sigma)
				bkg_estimator = MedianBackground()
				mask = make_source_mask(reduced_obj_frame_data, nsigma=sigma, npixels=npixels, dilate_size=dilate_size)
				bkg = Background2D(obj_frame_data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
				reduced_obj_frame_data -= bkg.background
				reduced_obj_frame_data += bkg.background_median

				obj_hdu = fits.PrimaryHDU(reduced_obj_frame_data, header=obj_frame_header)
				obj_hdu.writeto(obj_dir + "/reduced-" + obj)

		return obj_list

	@staticmethod
	def solve_plate(config, search=False):

		obj_dir = config["Test"]["obj_dir"]
		ra = config["Photometry"]["alpha"]
		dec = config["Photometry"]["delta"]
		radius = config["Photometry"]["solve_radius"]

		obj_list = []

		os.chdir(obj_dir)

		for item in glob.glob("reduced*.fit"):

			obj_list.append(item)

		obj_list = sorted(obj_list)

		for obj in obj_list:

			if os.path.isfile("wcs-" + obj):

				print("[REDUCER][solve_plate]: Skipping plate solve on", obj)

			else:

				file_name = obj[:-4]
				file_axy = file_name + ".axy"
				file_corr = file_name + ".corr"
				file_match = file_name + ".match"
				file_new = file_name + ".new"
				file_rdls = file_name + ".rdls"
				file_solved = file_name + ".solved"
				file_wcs = file_name + ".wcs"
				file_xyls = file_name + "-indx.xyls"

				if search == False:

					print("[REDUCER][solve_plate]: Running unconstrained astrometry on", obj)
					subprocess.run(["solve-field", "--no-plots", obj])

				else:

					print("[REDUCER][solve_plate]: Running constrained", (ra, dec, radius), "astrometry on", obj)
					subprocess.run(["solve-field", "--no-plots", obj, "--ra", ra, "--dec", dec, "--radius", radius])

				subprocess.run(["rm", file_axy])
				subprocess.run(["rm", file_corr])
				subprocess.run(["rm", file_match])
				subprocess.run(["rm", file_rdls])
				subprocess.run(["rm", file_solved])
				subprocess.run(["rm", file_wcs])
				subprocess.run(["rm", file_xyls])
				subprocess.run(["mv", file_new, "wcs-" + str(file_name) + ".fit"])

		return obj_list