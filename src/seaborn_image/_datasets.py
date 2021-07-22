import numpy as np
import pooch
from skimage import color, data, io

__all__ = ["load_image"]


POOCH = pooch.create(
    # Use the default cache folder for the OS
    path=pooch.os_cache("seaborn-image"),
    # The remote data is on Github
    base_url="https://github.com/SarthakJariwala/seaborn-image/raw/master/data/",
    # The registry specifies the files that can be fetched
    registry={
        # The registry is a dict with file names and their SHA256 hashes
        "PolymerImage.txt": "7b6798865080adf3ecf11e342f3d86d7b52ea0700020a1f062544ee825fb8a0e",
        "Perovskite.txt": "3228eeade5afec3c2b1ed116b2d4fe35877224d2d9bf7b4a17e04a432e6135c5",
        "cells.tif": "2120cfe08e0396324793a10a905c9bbcb64b117215eb63b2c24b643e1600c8c9",
    },
)


def load_image(name):
    """Load image data shippped with seaborn-image.

    Parameters
    ----------
    name : str
        Name of the image dataset

    Raises
    ------
    ValueError
        If the name of the dataset specified doesn't exist

    Returns
    -------
    `numpy.ndarray`
        Image data as a `numpy` array

    Examples
    --------
    >>> import seaborn_image as isns
    >>> img = isns.load_image("polymer")
    """

    if name == "polymer":
        path = POOCH.fetch("PolymerImage.txt")
        img = np.loadtxt(path, skiprows=1)
        img = img * 1e9  # convert height data from m to nm

    elif name == "polymer outliers":
        path = POOCH.fetch("PolymerImage.txt")
        img = np.loadtxt(path, skiprows=1)
        img = img * 1e9  # convert height data from m to nm
        img[25, 25] = 80  # assign an outlier value to a random pixel

    elif name == "fluorescence":
        path = POOCH.fetch("Perovskite.txt")
        img = np.loadtxt(path)

    elif name == "cells":
        path = POOCH.fetch("cells.tif")
        img = io.imread(path).T

    elif name == "retina-gray":
        img = color.rgb2gray(data.retina())[300:700, 700:900]

    else:
        raise ValueError(
            f"No '{name}' image dataset. Available image datasets include: polymer, polymer outliers, fluorescence, cells"
        )

    return img
