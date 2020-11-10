"""
Image filters
=============
"""

from skimage.filters import scharr

import seaborn_image as isns

pol = isns.load_image("polymer")

ax = isns.filterplot(pol, filt=scharr)
