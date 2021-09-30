from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Create a custom colormap so that background is transparent
# get colormap
ncolors = 256
color_array = plt.get_cmap('plasma')(range(ncolors))

# change alpha values
#color_array[:,-1] = np.linspace(0.0, 1.0, ncolors)
color_array[0, -1] = 0.0

# create a colormap object
map_object = LinearSegmentedColormap.from_list(name='plasma_alpha', colors=color_array)

# register this new colormap with matplotlib
plt.register_cmap(cmap=map_object)