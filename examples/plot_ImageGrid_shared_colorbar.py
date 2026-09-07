"""
ImageGrid: A shared color scale and colorbar
==========================================

A shared colorbar uses the same value-to-color mapping for every scalar image.
Limits are calculated from the selected images, after any transformations.
Use this when the images measure the same quantity in the same units.
"""

import seaborn_image as isns

cells = isns.load_image("cells")

# Automatic spacing with a single colorbar on the right.
g = isns.ImageGrid(
    cells, slices=[0, 10, 20, 30, 40, 50], cbar="shared", cbar_label="Intensity"
)

# %%
# Touching images with a shared colorbar below. The bar does not shrink the
# images or change their spacing. Robust limits use pooled pixel percentiles.
g = isns.ImageGrid(
    cells,
    slices=[0, 10, 20, 30, 40, 50],
    gap=0,
    cbar="shared",
    orientation="h",
    cbar_label="Intensity",
    robust=True,
)

# For exports without additional padding:
# g.fig.savefig("cells.png", pad_inches=0)
