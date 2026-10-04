# MiniTPU VMEM `mid` (vpu_word_array GEOM["mid"]: 64 b x 4096 words) as one OpenRAM 2RW macro, FreePDK45.
# Routers off (pure-Python escape/supply routers do not finish at this size on this host; stated in the record):
# the area is the macro core (width x height) without power ring or escape routes. DRC/LVS off: no NCSU PDK here.
word_size = 64
num_words = 4096
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
output_path = "/work/shared/users/phd/sk3463/scratch/asicmem_openram/mid/out"
output_name = "sram_2rw_64x4096_freepdk45"
