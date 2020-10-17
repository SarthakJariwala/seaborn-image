"""
ImageGrid: Mutiple images on a single figure
============================================
"""

import seaborn_image as isns

# load images
pol = isns.load_image("polymer")
pl = isns.load_image("fluorescence")

g = isns.ImageGrid([pol, pl])