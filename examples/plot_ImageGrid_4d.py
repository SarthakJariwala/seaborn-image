"""
Arrays of images with color
========================
"""

import seaborn_image as isns

cells = isns.load_image("cifar10")

g = isns.ImageGrid(cells, cbar=False, height=1, col_wrap=10)