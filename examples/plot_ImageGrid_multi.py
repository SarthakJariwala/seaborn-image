"""
Multi-dimensional images
========================
"""

import seaborn_image as isns

cells = isns.load_image("cells")

g = isns.ImageGrid(cells, cbar=False, height=1, col_wrap=5, step=2, cmap="inferno")
