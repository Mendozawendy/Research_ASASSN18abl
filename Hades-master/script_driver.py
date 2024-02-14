#from beta_refactor import *
from calculator import *
from photometer import *
from plotter import *
from querier import *
from reader import *
from reducer import *

# --- Start clock
os.system("clear")
print("Running workflow")
start_time = time.time()

# --- Create class instances
calculator = Calculator()
photometer = Photometer()
plotter = Plotter()
querier = Querier()
reader = Reader()
reducer = Reducer()

config = configparser.ConfigParser()
config.read("config.ini")

log_dir = config["Test"]["log_dir"]
#log = open(log_dir + "/log.txt", "w")

# --- Camera information
camera_params = reader.config_camera(config, True)
inverse_gain = camera_params["inverse_gain"]


# --- Location information
location = reader.get_location(config)

# --- Target information
target_alpha = config["Photometry"]["alpha"]
target_delta = config["Photometry"]["delta"]

t_alpha = config["Photometry"]["t_alpha"]
t_delta = config["Photometry"]["t_delta"]

c1_alpha = config["Photometry"]["c1_alpha"]
c1_delta = config["Photometry"]["c1_delta"]

c2_alpha = config["Photometry"]["c2_alpha"]
c2_delta = config["Photometry"]["c2_delta"]

c3_alpha = config["Photometry"]["c3_alpha"]
c3_delta = config["Photometry"]["c3_delta"]

# --- Reduce frames
#reducer.make_dark(config, "flat")
#reducer.make_dark(config, "object")
#reducer.make_flat(config)
reducer.reduce_objects(config)
reducer.solve_plate(config, search=True)
#align_list = reducer.align_frames(config)
#reducer.make_stack(config)

#
# --- STACK SUBROUTINE
#

# --- Read frame
stack_frame = "/Users/wendymendoza/Desktop/Yan/2023-03-16/stacked_5.fits"

stack_params = reader.read_frame(config, stack_frame)

stack_dateobs = stack_params["dateobs"]
stack_jd = stack_params["jd"]
stack_ra = stack_params["ra"]
stack_dec = stack_params["dec"]
stack_x = stack_params["x"]
stack_y = stack_params["y"]
stack_exptime = stack_params["exptime"]
stack_mean = stack_params["mean"]
stack_median = stack_params["median"]
stack_std = stack_params["std"]

# --- Calculate air mass
stack_obstime = Time(stack_dateobs)
stack_airmass, stack_zenith_distance, stack_altitude = calculator.calculate_airmass(location, stack_obstime, stack_ra, stack_dec)

# --- Calculate seeing and growth radius
stack_seeing_pix, stack_seeing_sky, stack_growth_radius = reducer.extract_sources(stack_frame, inverse_gain)

# --- Build mask
mask, boxes = reducer.make_mask(config, stack_frame)

# --- Submit query
stack_table = querier.submit_query(config, "gaia")

# --- Photometry
phot_table, phot_params = photometer.make_table(stack_frame, stack_table, stack_growth_radius, mask, boxes, stack_mean, stack_std, stack_exptime, plot=True)

stack_transform = phot_params["transform"]
stack_delta_transform = phot_params["delta_transform"]
stack_eff_zp = phot_params["eff_zp"]
stack_delta_eff_zp = phot_params["delta_eff_zp"]

log.write("Date: " + str(stack_dateobs) + "\n")
log.write("JD: " + str(stack_jd) + "\n")
log.write("RA: " + str(stack_ra) + " deg" + "\n")
log.write("Dec: " + str(stack_dec) + " deg" + "\n")
log.write("x: " + str(stack_x) + " px" + "\n")
log.write("y: " + str(stack_y) + " px" + "\n")
log.write("Exptime: " + str(stack_exptime) + " s" + "\n")
log.write("Bkg mean: " + str(stack_mean) + " ADU" + "\n")
log.write("Bkg median: " + str(stack_median) + " ADU" + "\n")
log.write("Bkg std dev: " + str(stack_std) + " ADU" + "\n")
log.write("Altitude: " + str(stack_altitude) + " deg" + "\n")
log.write("Zenith distance: " + str(stack_zenith_distance) + " deg" + "\n")
log.write("Air mass: " + str(stack_airmass) + "\n")
log.write("Mean seeing (pix): " + str(stack_seeing_pix) + " px" + "\n")
log.write("Mean seeing (sky): " + str(stack_seeing_sky) + " arcsec" + "\n")
log.write("Mean growth radius: " + str(stack_growth_radius) + " px" + "\n")
log.write("Transform: " + str(stack_transform) + " +/- " + str(stack_delta_transform) + "\n")
log.write("ZP': " + str(stack_eff_zp) + " +/- " + str(stack_delta_eff_zp) + "\n")



log.close()

end_time = time.time()
total_time = end_time - start_time
print("Workflow ended in", "%.1f" % total_time, "seconds")