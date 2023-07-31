"""
Default settings

Code written by: Andrew Justin (andrewjustinwx@gmail.com)
Last updated: 7/30/2023 6:18 PM CT
"""
DEFAULT_DOMAIN_EXTENTS = {'global': [0, 359.75, -89.75, 90],
                          'full': [130, 369.75, 0.25, 80],
                          'conus': [228, 299.75, 25, 56.75]}  # default values for extents of domains [start lon, end lon, start lat, end lat]
DEFAULT_DOMAIN_INDICES = {'global': [0, 1440, 0, 720],
                          'full': [0, 960, 0, 320],
                          'conus': [392, 680, 93, 221]}  # indices corresponding to default extents of domains [start lon, end lon, start lat, end lat]

# colors for plotted ground truth fronts
DEFAULT_FRONT_COLORS = {'CF': 'blue', 'WF': 'red', 'SF': 'limegreen', 'OF': 'darkviolet', 'CF-F': 'darkblue', 'WF-F': 'darkred',
                        'SF-F': 'darkgreen', 'OF-F': 'darkmagenta', 'CF-D': 'lightskyblue', 'WF-D': 'lightcoral', 'SF-D': 'lightgreen',
                        'OF-D': 'violet', 'OFB': 'gold', 'TROF': 'goldenrod', 'TT': 'orange', 'DL': 'chocolate', 'CF-WF-SF-OF': 'tab:red',
                        'CF_any': 'blue', 'WF_any': 'red', 'SF_any': 'limegreen', 'OF_any': 'darkviolet', 'CF-WF-SF-OF_any': 'gray',
                        'TR_any': 'brown', 'F_BIN': 'tab:red', 'MERGED-F_BIN': 'tab:red'}

# colormaps of probability contours for front predictions
DEFAULT_CONTOUR_CMAPS = {'CF': 'Blues', 'WF': 'Reds', 'SF': 'Greens', 'OF': 'Purples', 'CF-F': 'Blues', 'WF-F': 'Reds',
                         'SF-F': 'Greens', 'OF-F': 'Purples', 'CF-D': 'Blues', 'WF-D': 'Reds', 'SF-D': 'Greens', 'OF-D': 'Purples',
                         'OFB': 'YlOrBr', 'TROF': 'YlOrRed', 'TT': 'Oranges', 'DL': 'copper_r', 'CF-WF-SF-OF': 'Reds', 'CF_any': 'Blues',
                         'WF_any': 'Reds', 'SF_any': 'Greens', 'OF_any': 'Purples', 'CF-WF-SF-OF_any': 'Greys', 'TR_any': 'YlOrBr',
                         'F_BIN': 'Greys', 'MERGED-F_BIN': 'Greys'}

# names of front types
DEFAULT_FRONT_NAMES = {'CF': 'Cold front', 'WF': 'Warm front', 'SF': 'Stationary front', 'OF': 'Occluded front', 'CF-F': 'Cold front (forming)',
                       'WF-F': 'Warm front (forming)', 'SF-F': 'Stationary front (forming)', 'OF-F': 'Occluded front (forming)',
                       'CF-D': 'Cold front (dying)', 'WF-D': 'Warm front (dying)', 'SF-D': 'Stationary front (dying)', 'OF-D': 'Occluded front (dying)',
                       'OFB': 'Outflow boundary', 'TROF': 'Trough', 'TT': 'Tropical trough', 'DL': 'Dryline', 'CF-WF-SF-OF': 'CF, WF, SF, OF',
                       'CF_any': 'Cold front (any)', 'WF_any': 'Warm front (any)', 'SF_any': 'Stationary front (any)', 'OF_any': 'Occluded front (any)',
                       'CF-WF-SF-OF_any': 'CF, WF, SF, OF (any)', 'TR_any': 'Trough (any)', 'F_BIN': 'Binary front', 'MERGED-F_BIN': 'Binary front (any)'}

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
GPU_PREDICT_BATCH_SIZE = 4

"""
MAX_FILE_CHUNK_SIZE is the maximum number of ERA5, GDAS, and/or GFS netCDF files that will be loaded into an xarray dataset at one
time. Loading too many files / too much data into one xarray dataset can take a very long time and may lead to segmentation errors.
If segmentation errors are occurring, consider lowering this parameter until the error disappears.
"""
MAX_FILE_CHUNK_SIZE = 100
