"""
Image plot
==========

"""

import seaborn_image as isns

img = isns.load_image("polymer")
img_scale = {"dx": 15, "units": "nm"}

_ = isns.imgplot(img, dx=15, units="nm", despine=False)
