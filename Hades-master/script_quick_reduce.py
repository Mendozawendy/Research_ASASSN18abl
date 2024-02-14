from reducer import *

os.system("clear")
print("Running quick_reduce")
start_time = time.time()

reducer = Reducer() 

config = configparser.ConfigParser()
config.read("config.ini")

log_dir = config["Test"]["log_dir"]
log = open(log_dir + "/log.txt", "w")

reducer.make_dark(config, "flat")
reducer.make_dark(config, "object")
reducer.make_flat(config)
reducer.reduce_objects(config) 