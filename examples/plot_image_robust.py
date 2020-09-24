"""
Outlier correction
==================
"""

import matplotlib.pyplot as plt

import seaborn_image as isns

img = isns.load_image("polymer outliers")

f, axes = plt.subplots(1, 2)

_ = isns.imgplot(img, ax=axes[0], cmap="inferno", despine=False)
_ = isns.imgplot(img, ax=axes[1], robust=True, perc=(2, 99.99), cmap="inferno")
