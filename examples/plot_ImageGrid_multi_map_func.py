"""
Transform multi-dimensional images
==================================
"""

from skimage.exposure import adjust_gamma

import seaborn_image as isns

cells = isns.load_image("cells")

g = isns.ImageGrid(
    cells,
    map_func=adjust_gamma,
    gamma=0.5,
    cbar=False,
    height=1,
    col_wrap=10,
)
