"""
Default settings

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.7.24
"""
DEFAULT_DOMAIN_EXTENTS = {'global': [0, 359.75, -89.75, 90],
                          'full': [130, 369.75, 0.25, 80],
                          'conus': [228, 299.75, 25, 56.75]}  # default values for extents of domains [start lon, end lon, start lat, end lat]
DEFAULT_DOMAIN_INDICES = {'global': [0, 1440, 0, 720],
                          'full': [0, 960, 0, 320],
                          'conus': [392, 680, 93, 221]}  # indices corresponding to default extents of domains [start lon, end lon, start lat, end lat]
DEFAULT_DOMAIN_IMAGES = {'global': [17, 9],
                         'full': [8, 3],
                         'conus': [3, 1]}  # default values for the number of images to use when making predictions [lon, lat]

# colors for plotted ground truth fronts
DEFAULT_FRONT_COLORS = {'CF': 'blue', 'WF': 'red', 'SF': 'limegreen', 'OF': 'darkviolet', 'CF-F': 'darkblue', 'WF-F': 'darkred',
                        'SF-F': 'darkgreen', 'OF-F': 'darkmagenta', 'CF-D': 'lightskyblue', 'WF-D': 'lightcoral', 'SF-D': 'lightgreen',
                        'OF-D': 'violet', 'INST': 'gold', 'TROF': 'goldenrod', 'TT': 'orange', 'DL': 'chocolate',
                        'MERGED-CF': 'blue', 'MERGED-WF': 'red', 'MERGED-SF': 'limegreen', 'MERGED-OF': 'darkviolet', 'MERGED-F': 'gray',
                        'MERGED-T': 'brown', 'F_BIN': 'tab:red', 'MERGED-F_BIN': 'tab:red'}

# colormaps of probability contours for front predictions
DEFAULT_CONTOUR_CMAPS = {'CF': 'Blues', 'WF': 'Reds', 'SF': 'Greens', 'OF': 'Purples', 'CF-F': 'Blues', 'WF-F': 'Reds',
                         'SF-F': 'Greens', 'OF-F': 'Purples', 'CF-D': 'Blues', 'WF-D': 'Reds', 'SF-D': 'Greens', 'OF-D': 'Purples',
                         'INST': 'YlOrBr', 'TROF': 'YlOrRd', 'TT': 'Oranges', 'DL': 'copper_r', 'MERGED-CF': 'Blues',
                         'MERGED-WF': 'Reds', 'MERGED-SF': 'Greens', 'MERGED-OF': 'Purples', 'MERGED-F': 'Greys', 'MERGED-T': 'YlOrBr',
                         'F_BIN': 'Greys', 'MERGED-F_BIN': 'Greys'}

# names of front types
DEFAULT_FRONT_NAMES = {'CF': 'Cold front', 'WF': 'Warm front', 'SF': 'Stationary front', 'OF': 'Occluded front', 'CF-F': 'Cold front (forming)',
                       'WF-F': 'Warm front (forming)', 'SF-F': 'Stationary front (forming)', 'OF-F': 'Occluded front (forming)',
                       'CF-D': 'Cold front (dying)', 'WF-D': 'Warm front (dying)', 'SF-D': 'Stationary front (dying)', 'OF-D': 'Occluded front (dying)',
                       'INST': 'Outflow boundary', 'TROF': 'Trough', 'TT': 'Tropical trough', 'DL': 'Dryline',
                       'MERGED-CF': 'Cold front (any)', 'MERGED-WF': 'Warm front (any)', 'MERGED-SF': 'Stationary front (any)', 'MERGED-OF': 'Occluded front (any)',
                       'MERGED-F': 'CF, WF, SF, OF (any)', 'MERGED-T': 'Trough (any)', 'F_BIN': 'Binary front', 'MERGED-F_BIN': 'Binary front (any)'}

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
