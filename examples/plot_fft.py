"""
Fast Fourier Transform
======================
"""

import seaborn_image as isns

img = isns.load_image("polymer")

_ = isns.fftplot(img, cmap="viridis", cbar=False, despine=False)
