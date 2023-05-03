"""
4D image data 
=============
"""

import seaborn_image as isns

cifar = isns.load_image("cifar10")

g = isns.ImageGrid(cifar, cbar=False, height=1, col_wrap=10)
