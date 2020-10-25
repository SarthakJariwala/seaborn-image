"""
RGB channels in RGB image
=========================

Split and plot the channels of the RGB image
"""

from skimage.data import astronaut

import seaborn_image as isns

# set image origin
isns.set_image(origin="upper")

g = isns.rgbplot(astronaut())
