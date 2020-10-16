"""
Apply image filter
==================
"""

import seaborn_image as isns

img = isns.load_image("polymer")

ax = isns.filterplot(img, "median", size=5, cmap="ice")
