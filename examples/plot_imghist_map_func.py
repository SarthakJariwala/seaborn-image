"""
Map function to transform 2D image
==================================
"""


from skimage.exposure import adjust_gamma

import seaborn_image as isns

cells = isns.load_image("cells")[:, :, 32]

f = isns.imghist(cells, map_func=adjust_gamma, map_func_kw={"gamma": 0.5})
