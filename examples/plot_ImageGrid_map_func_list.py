"""
Apply multiple transformations to 2-D image
===========================================
"""

from skimage.filters import frangi, hessian, meijering, sato

import seaborn_image as isns

retina = isns.load_image("retina-gray")

g = isns.ImageGrid(
    retina,
    map_func=[meijering, sato, frangi, hessian],
    col_wrap=4,
    map_func_kwargs=[{"mode": "reflect", "sigmas": [1]} for _ in range(4)],
)
