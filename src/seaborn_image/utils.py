import os

import matplotlib.pyplot as plt
import numpy as np
import pooch
from matplotlib import ticker
from skimage import io

__all__ = ["scientific_ticks", "despine", "load_image"]


def scientific_ticks(ax, which="y"):
    """Convert axis ticks to scientific

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Axis where ticks are to be converted
    which : str, optional
        Which axis ticks to convert to scientific, default to "y".
        Options include : "y", "x", "both"

    Raises
    ------
    ValueError
        If `which` is not one of ['y', 'x', 'both']

    Examples
    --------

    Set colorbar yaxis ticks to scientific

    .. plot::
        :context: close-figs

        >>> import seaborn_image as isns
        >>> img = isns.load_image("polymer") * 1e-9
        >>> ax = isns.imgplot(img)
        >>> # get colorbar axes
        >>> cax = plt.gcf().axes[1]
        >>> isns.scientific_ticks(cax)

    Set colorbar xaxis ticks to scientific

    .. plot::
        :context: close-figs

        >>> import seaborn_image as isns
        >>> img = isns.load_image("polymer") * 1e-9
        >>> ax = isns.imgplot(img, orientation="h")
        >>> # get colorbar axes
        >>> cax = plt.gcf().axes[1]
        >>> isns.scientific_ticks(cax, which="x")
    """

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    # formatter.set_powerlimits((-1,1))

    if which == "both":
        ax.yaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_formatter(formatter)
    elif which == "y":
        ax.yaxis.set_major_formatter(formatter)
    elif which == "x":
        ax.xaxis.set_major_formatter(formatter)
    else:
        raise ValueError("Options include: 'both', 'y', 'x'")


def despine(fig=None, ax=None, which="all"):
    """Remove the specified spine/s from a given
    `matplotlib.Axes` or all the axes from a given
    `matplotlib.Figure`.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`, optional
        The figure where all the axes are to be despined, by default None
    ax : `matplotlib.axes.Axes`, optional
        The axes to despine, by default None
    which : str or list, optional
        The specific spine to remove, by default "all"

    Raises
    ------
    ValueError
        If the `which` is not in the list of spines :
        ["all", "top", "bottom", "right", "left"]
    TypeError
        If `which` is not a str or list

    Examples
    --------

    Despine all axes in a figure

    .. plot::
        :context: close-figs

        >>> import seaborn_image as isns
        >>> fig, axes = plt.subplots(nrows=2, ncols=3)
        >>> isns.despine()

    Or equivalently

    .. plot::
        :context: close-figs

        >>> import seaborn_image as isns
        >>> fig, axes = plt.subplots(nrows=2, ncols=3)
        >>> isns.despine(fig)

    Despine a specific axis

    .. plot::
        :context: close-figs

        >>> import seaborn_image as isns
        >>> fig, axes = plt.subplots(nrows=2, ncols=3)
        >>> isns.despine(ax=axes[-1][-1])

    Despine only top and right axes

    .. plot::
        :context: close-figs

        >>> import seaborn_image as isns
        >>> fig, axes = plt.subplots(nrows=2, ncols=3)
        >>> isns.despine(which=["top", "right"])
    """

    if fig is None and ax is None:
        axes = plt.gcf().axes
    elif fig is not None:
        axes = fig.axes
    elif ax is not None:
        if isinstance(ax, np.ndarray):
            axes = ax.ravel()
        else:
            axes = [ax]

    _all = ["top", "bottom", "right", "left"]

    if isinstance(which, str):
        if which == "all":
            _to_despine = _all
        elif which in _all:
            _to_despine = [which]
        else:
            raise ValueError(
                f"Specify spine that is to be despined from the following : {_all.append('all')}"
            )

    elif isinstance(which, list):
        _to_despine = []
        for _which in which:
            if _which in _all:
                _to_despine.append(_which)

    else:
        raise TypeError(
            f"{which} must be of the type 'str' or 'list'. Options are : {_all.append('all')}"
        )

    for ax in axes:
        for spine in _to_despine:
            ax.spines[spine].set_visible(False)


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
        try:
            path = "data/PolymerImage.txt"
            img = np.loadtxt(path, skiprows=1)
        except OSError:  # pragma: no cover
            try:
                # if building docstrings
                path = "../data/PolymerImage.txt"
                img = np.loadtxt(path, skiprows=1)
            except OSError:  # pragma: no cover
                # if building tuorial
                path = "../../data/PolymerImage.txt"
                img = np.loadtxt(path, skiprows=1)

        img = img * 1e9  # convert height data from m to nm

    elif name == "polymer outliers":
        try:  # not very pretty fix to the path issue
            path = "data/PolymerImage.txt"
            img = np.loadtxt(path, skiprows=1)
        except OSError:  # pragma: no cover
            try:
                # if building docstrings
                path = "../data/PolymerImage.txt"
                img = np.loadtxt(path, skiprows=1)
            except OSError:  # pragma : no cover
                # if building tutorial
                path = "../../data/PolymerImage.txt"
                img = np.loadtxt(path, skiprows=1)

        img = img * 1e9  # convert height data from m to nm
        img[0, 0] = 80  # assign an outlier value to a random pixel

    elif name == "fluorescence":
        try:  # not very pretty fix to the path issue
            path = "data/Perovskite.txt"
            img = np.loadtxt(path)
        except OSError:  # pragma: no cover
            try:
                # if building docstrings
                path = "../data/Perovskite.txt"
                img = np.loadtxt(path)
            except OSError:  # pragma: no cover
                # if building tutorial
                path = "../../data/Perovskite.txt"
                img = np.loadtxt(path)

    elif name == "cells":
        fname = pooch.retrieve(
            url="https://github.com/scikit-image/skimage-tutorials/raw/master/images/cells.tif",
            known_hash="2120cfe08e0396324793a10a905c9bbcb64b117215eb63b2c24b643e1600c8c9",
        )
        img = io.imread(fname).T

    else:
        raise ValueError(f"No '{name}' image dataset")

    return img
