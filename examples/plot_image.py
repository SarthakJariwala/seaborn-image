"""
2-D image visualization
=======================

"""

import seaborn_image as isns

img = isns.load_image("polymer")

ax = isns.imgplot(img, dx=15, units="nm")
