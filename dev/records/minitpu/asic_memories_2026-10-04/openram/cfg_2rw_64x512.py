# VMEM mid as 8 x (512 x 64 b) OpenRAM 2RW banks: the fallback if one 4096 x 64 macro does not build in time.
# Routers off (pure-Python escape/supply routers do not finish at this size on this host; stated in the record):
# the area is the macro core (width x height) without power ring or escape routes. DRC/LVS off: no NCSU PDK here.
word_size = 64
num_words = 512
num_rw_ports = 2
num_r_ports = 0
num_w_ports = 0
tech_name = "freepdk45"
nominal_corner_only = True
process_corners = ["TT"]
supply_voltages = [1.0]
temperatures = [25]
check_lvsdrc = False
analytical_delay = True
use_nix = False
route_supplies = False
perimeter_pins = False
num_threads = 8
output_path = "/work/shared/users/phd/sk3463/scratch/asicmem_openram/w512/out"
output_name = "sram_2rw_64x512_freepdk45"
