"""
Image Filter
============
"""

import seaborn_image as isns

img = isns.load_image("polymer")

_ = isns.filterplot(img, "median", size=5, cmap="ice", despine=False)
