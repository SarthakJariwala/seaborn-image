"""
Apply multiple transformations to list of 2-D images
====================================================
"""

from scipy.ndimage import median_filter, sobel
from skimage.filters import gaussian

import seaborn_image as isns

pl = isns.load_image("fluorescence")
polymer = isns.load_image("polymer")

map_func = [gaussian, median_filter, sobel]
map_func_kw = [{"sigma": 1.5}, {"size": 10}, None]

g = isns.ImageGrid(
    [pl, polymer],
    map_func=map_func,
    map_func_kw=map_func_kw,
    col_wrap=2,
    cmap="inferno",
)
