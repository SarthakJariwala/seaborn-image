import numpy as np

import seaborn_image as isns

# Load data
data = np.loadtxt("../data/PolymerImage.txt", skiprows=1) * 1e9

# Plot
isns.imgplot(data, dx=4 / 256, units='um', cbar_label="Height (nm)")
