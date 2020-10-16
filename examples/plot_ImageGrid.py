"""
ImageGrid: Mutiple images on a single figure
============================================
"""

import seaborn_image as isns

# load images
pol = isns.load_image("polymer")
pl = isns.load_image("fluorescence")
pol_out = isns.load_image("polymer outliers")

g = isns.ImageGrid([pol, pl, pol_out], robust=[False, False, True], perc=[None, None, (2, 99.9)])
g = isns.ImageGrid([pol, pl])