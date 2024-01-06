"""
Default settings

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.1.5
"""
# default values for extents of domains [start lon, end lon, start lat, end lat]
DOMAIN_EXTENTS = {'atlantic': [290, 349.75, 16, 55.75],
                  'conus': [228, 299.75, 25, 56.75],
                  'ecmwf': [0, 359.75, -89.75, 90],
                  'full': [130, 369.75, 0.25, 80],
                  'global': [0, 359.75, -89.75, 90],
                  'hrrr': [225.90452026573686, 299.0828072281622, 21.138123000000018, 52.61565330680793],
                  'namnest-conus': [225.90387325951775, 299.08216099364034, 21.138, 52.61565399063001],
                  'nam-12km': [207.12137749594984, 310.58401341435564, 12.190000000000005, 61.30935757335816],
                  'pacific': [145, 234.75, 16, 55.75]}

# colors for plotted ground truth fronts
FRONT_COLORS = {'CF': 'blue', 'WF': 'red', 'SF': 'limegreen', 'OF': 'darkviolet', 'CF-F': 'darkblue', 'WF-F': 'darkred',
                'SF-F': 'darkgreen', 'OF-F': 'darkmagenta', 'CF-D': 'lightskyblue', 'WF-D': 'lightcoral', 'SF-D': 'lightgreen',
                'OF-D': 'violet', 'INST': 'gold', 'TROF': 'goldenrod', 'TT': 'orange', 'DL': 'chocolate', 'MERGED-CF': 'blue',
                'MERGED-WF': 'red', 'MERGED-SF': 'limegreen', 'MERGED-OF': 'darkviolet', 'MERGED-F': 'gray', 'MERGED-T': 'brown',
                'F_BIN': 'tab:red', 'MERGED-F_BIN': 'tab:red'}

# colormaps of probability contours for front predictions
CONTOUR_CMAPS = {'CF': 'Blues', 'WF': 'Reds', 'SF': 'Greens', 'OF': 'Purples', 'CF-F': 'Blues', 'WF-F': 'Reds', 'SF-F': 'Greens',
                 'OF-F': 'Purples', 'CF-D': 'Blues', 'WF-D': 'Reds', 'SF-D': 'Greens', 'OF-D': 'Purples', 'INST': 'YlOrBr',
                 'TROF': 'YlOrRd', 'TT': 'Oranges', 'DL': 'copper_r', 'MERGED-CF': 'Blues', 'MERGED-WF': 'Reds', 'MERGED-SF': 'Greens',
                 'MERGED-OF': 'Purples', 'MERGED-F': 'Greys', 'MERGED-T': 'YlOrBr', 'F_BIN': 'Greys', 'MERGED-F_BIN': 'Greys'}

# names of front types
FRONT_NAMES = {'CF': 'Cold front', 'WF': 'Warm front', 'SF': 'Stationary front', 'OF': 'Occluded front', 'CF-F ': 'Cold front (forming)',
               'WF-F': 'Warm front (forming)', 'SF-F': 'Stationary front (forming)', 'OF-F': 'Occluded front (forming)',
               'CF-D': 'Cold front (dying)', 'WF-D': 'Warm front (dying)', 'SF-D': 'Stationary front (dying)', 'OF-D': 'Occluded front (dying)',
               'INST': 'Outflow boundary', 'TROF': 'Trough', 'TT': 'Tropical trough', 'DL': 'Dryline', 'MERGED-CF': 'Cold front (any)',
               'MERGED-WF': 'Warm front (any)', 'MERGED-SF': 'Stationary front (any)', 'MERGED-OF': 'Occluded front (any)',
               'MERGED-F': 'CF, WF, SF, OF (any)', 'MERGED-T': 'Trough (any)', 'F_BIN': 'Binary front', 'MERGED-F_BIN': 'Binary front (any)'}

VARIABLE_NAMES = {"T": "Air temperature", "T_sfc": "2-meter Air temperature", "T_1000": "1000mb Air temperature", "T_950": "950mb Air temperature",
                  "T_900": "900mb Air temperature", "T_850": "850mb Air temperature",
                  "Td": "Dewpoint", "Td_sfc": "2-meter Dewpoint", "Td_1000": "1000mb Dewpoint", "Td_950": "950mb Dewpoint",
                  "Td_900": "900mb Dewpoint", "Td_850": "850mb Dewpoint",
                  "Tv": "Virtual temperature", "Tv_sfc": "2-meter Virtual temperature", "Tv_1000": "1000mb Virtual temperature",
                  "Tv_950": "950mb Virtual temperature", "Tv_900": "900mb Virtual temperature", "Tv_850": "850mb Virtual temperature",
                  "Tw": "Wet-bulb temperature", "Tw_sfc": "2-meter Wet-bulb temperature", "Tw_1000": "1000mb Wet-bulb temperature",
                  "Tw_950": "950mb Wet-bulb temperature", "Tw_900": "900mb Wet-bulb temperature", "Tw_850": "850mb Wet-bulb temperature",
                  "theta": "Potential temperature", "theta_sfc": "2-meter Potential temperature", "theta_1000": "1000mb Potential temperature",
                  "theta_950": "950mb Potential temperature", "theta_900": "900mb Potential temperature", "theta_850": "850mb Potential temperature",
                  "theta_e": "Theta-E", "theta_e_sfc": "2-meter Theta-E", "theta_e_1000": "1000mb Theta-E",
                  "theta_e_950": "950mb Theta-E", "theta_e_900": "900mb Theta-E", "theta_e_850": "850mb Theta-E",
                  "theta_v": "Virtual potential temperature", "theta_v_sfc": "2-meter Virtual potential temperature", "theta_v_1000": "1000mb Virtual potential temperature",
                  "theta_v_950": "950mb Virtual potential temperature", "theta_v_900": "900mb Virtual potential temperature", "theta_v_850": "850mb Virtual potential temperature",
                  "theta_w": "Wet-bulb potential temperature", "theta_w_sfc": "2-meter Wet-bulb potential temperature", "theta_w_1000": "1000mb Wet-bulb potential temperature",
                  "theta_w_950": "950mb Wet-bulb potential temperature", "theta_w_900": "900mb Wet-bulb potential temperature", "theta_w_850": "850mb Wet-bulb potential temperature",
                  "u": "U-wind", "u_sfc": "10-meter U-wind", "u_1000": "1000mb U-wind", "u_950": "950mb U-wind", "u_900": "900mb U-wind",
                  "u_850": "850mb U-wind",
                  "v": "V-wind", "v_sfc": "10-meter V-wind", "v_1000": "1000mb V-wind", "v_950": "950mb V-wind", "v_900": "900mb V-wind",
                  "v_850": "850mb V-wind",
                  "q": "Specific humidity", "q_sfc": "2-meter Specific humidity", "q_1000": "1000mb Specific humidity",
                  "q_950": "950mb Specific humidity", "q_900": "900mb Specific humidity", "q_850": "850mb Specific humidity",
                  "r": "Mixing ratio", "r_sfc": "2-meter Mixing ratio", "r_1000": "1000mb Mixing ratio",
                  "r_950": "950mb Mixing ratio", "r_900": "900mb Mixing ratio", "r_850": "850mb Mixing ratio",
                  "RH": "Relative humidity", "RH_sfc": "2-meter Relative humidity", "RH_1000": "1000mb Relative humidity",
                  "RH_950": "950mb Relative humidity", "RH_900": "900mb Relative humidity", "RH_850": "850mb Relative humidity",
                  "sp_z": "Pressure/heights", "sp_z_sfc": "Surface pressure", "sp_z_1000": "1000mb Geopotential height",
                  "sp_z_950": "950mb Geopotential height", "sp_z_900": "900mb Geopotential height", "sp_z_850": "850mb Geopotential height",
                  "mslp_z": "Pressure/heights", "mslp_z_sfc": "Mean sea level pressure", "mslp_z_1000": "1000mb Geopotential height",
                  "mslp_z_950": "950mb Geopotential height", "mslp_z_900": "900mb Geopotential height", "mslp_z_850": "850mb Geopotential height"}

VERTICAL_LEVELS = {"surface": "Surface", "1000": "1000mb", "950": "950mb", "900": "900mb", "850": "850mb", "700": "700mb"}

""" 
TIMESTEP_PREDICT_SIZE is the number of timesteps for which predictions will be processed at the same time. In other words, 
    if this parameter is 10, then up to 10 maps will be generated at the same time for 10 timesteps.

Typically, raising this parameter will result in faster predictions, but the memory requirements increase as well. The size 
    of the domain for which the predictions are being generated greatly affects the limits of this parameter.
    
NOTES:
    - Setting the values at a lower threshold may result in slower predictions but will not have negative effects on hardware.
    - Increasing the parameters above the default values is STRONGLY discouraged. Greatly exceeding the allocated RAM 
      will force your operating system to resort to virtual memory usage, which can cause major slowdowns, OS crashes, and GPU failure.

In the case of any hardware failures (including OOM errors from the GPU) or major slowdowns, reduce the parameter(s) for the 
    domain(s) until your system becomes stable.
"""
TIMESTEP_PREDICT_SIZE = {'conus': 128, 'full': 64, 'global': 16}  # All values here are adjusted for 16 GB of system RAM and 10 GB of GPU VRAM

"""
GPU_PREDICT_BATCH_SIZE is the number of images that the GPU will process at one time when generating predictions.
If predictions are being generated on the same GPU that the model was trained on, then this value should be equal to or greater than
the batch size used when training the model.

NOTES:
    - This value should ideally be a value of 2^n, where n is any integer. Using values not equal to 2^n may have negative effects
        on performance.
    - Decreasing this parameter will result in overall slower performance, but can help prevent OOM errors on the GPU.
    - Increasing this parameter can speed up predictions on high-performance GPUs, but a value too large can cause OOM errors
        and GPU failure.
"""
GPU_PREDICT_BATCH_SIZE = 8

"""
MAX_FILE_CHUNK_SIZE is the maximum number of ERA5, GDAS, and/or GFS netCDF files that will be loaded into an xarray dataset at one
time. Loading too many files / too much data into one xarray dataset can take a very long time and may lead to segmentation errors.
If segmentation errors are occurring, consider lowering this parameter until the error disappears.
"""
MAX_FILE_CHUNK_SIZE = 2500

"""
MAX_TRAIN_BUFFER_SIZE is the maximum number of elements within the training dataset that can be shuffled at one time. Tensorflow 
does not efficiently use RAM during shuffling on Windows machines and can lead to system crashes, so the buffer size should be 
relatively small. It is important to monitor RAM usage if you are training a model on Windows. Linux is able to shuffle much
larger datasets than Windows, but crashes can still occur if the maximum buffer size is too large.
"""
MAX_TRAIN_BUFFER_SIZE = 5000
