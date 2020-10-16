"""
RGB channels in RGB image
=========================

Split and plot the channels of the RGB image
"""

import seaborn_image as isns
from skimage.data import astronaut

# set image origin
isns.set_image(origin="upper")

g = isns.rgbplot(astronaut())