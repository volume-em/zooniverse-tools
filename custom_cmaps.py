"""
Custom colormap for pyplot used to display cleaner segmentation results
with a transparent background.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Create a custom colormap so that background is transparent
# get colormap
ncolors = 256
color_array = plt.get_cmap('hsv')(range(ncolors))
color_array = np.random.permutation(color_array)

# change alpha values
color_array[0, -1] = 0.0

# create a colormap object
map_object = LinearSegmentedColormap.from_list(name='hsv_alpha', colors=color_array)

# register this new colormap with matplotlib
plt.register_cmap(cmap=map_object)

# Create a custom colormap so that background is transparent
# get colormap
ncolors = 256
color_array = plt.get_cmap('hsv')(range(ncolors))
color_array = np.random.permutation(color_array)

# change alpha values
color_array[0] = [0, 0, 0, 1]

# create a colormap object
map_object = LinearSegmentedColormap.from_list(name='hsv_black', colors=color_array)

# register this new colormap with matplotlib
plt.register_cmap(cmap=map_object)