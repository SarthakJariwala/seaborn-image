"""
List of 3D images
=================
"""

import seaborn_image as isns

isns.set_image(origin="lower")

cifar_list = isns.load_image("cifar10 list")

g = isns.ImageGrid(cifar_list, cbar=False, height=1, col_wrap=10)
