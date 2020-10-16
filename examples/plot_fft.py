"""
Fast Fourier Transform
======================
"""

import seaborn_image as isns

img = isns.load_image("polymer")

ax = isns.fftplot(img, window_type="hann", cmap="viridis")
