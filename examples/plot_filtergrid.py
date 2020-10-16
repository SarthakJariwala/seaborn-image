"""
Interplay between filter parameters using FilterGrid
====================================================
"""

import seaborn_image as isns

img = isns.load_image("polymer")

g = isns.FilterGrid(
    img,
    "gaussian",
    row="sigma",
    col="mode",
    sigma=[2, 3, 4],
    mode=["reflect", "nearest", "mirror"],
    cmap="magma",
)
