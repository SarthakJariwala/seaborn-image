import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import scipy.stats as ss
from scipy.fftpack import fftn, fftshift
from skimage.filters import difference_of_gaussians, window

from ._general import imgplot

__all__ = ["filterplot", "fftplot", "implemented_filters"]

implemented_filters = [
    "sobel",
    "gaussian",
    "median",
    "max",
    "diff_of_gaussians",
    "gaussian_gradient_magnitude",
    "gaussian_laplace",
    "laplace",
    "min",
    "percentile",
    "prewitt",
    "rank",
    "uniform",
]


def filterplot(
    data,
    filter="gaussian",
    *,
    ax=None,
    cmap=None,
    vmin=None,
    vmax=None,
    dx=None,
    units=None,
    dimension=None,
    describe=False,
    cbar=True,
    orientation="v",
    cbar_label=None,
    cbar_fontdict=None,
    cbar_ticks=None,
    showticks=False,
    despine=True,
    title=None,
    title_fontdict=None,
    **kwargs,
):
    """
    Apply N-dimensional filters and plot the filterd data as 2-D image with options to
    add scalebar, colorbar, titles and configure similar to `imgplot`

    Args:
        data: Image data (array-like). Supported array shapes are all
            `matplotlib.pyplot.imshow` array shapes
        filter (str, optional): Image filter to be used for processing.
            Defaults to "gaussian".
            Options include: "sobel", "gaussian", "median", "max", "diff_of_gaussians",
            "gaussian_gradient_magnitude", "gaussian_laplace", "laplace", "min", "percentile",
            "prewitt", "rank", "uniform".
        cmap (str or `matplotlib.colors.Colormap`, optional): Colormap for image.
            Can be a seaborn-image colormap or default matplotlib colormaps or
            any other colormap converted to a matplotlib colormap. Defaults to None.
        vmin (float, optional): Minimum data value that colormap covers. Defaults to None.
        vmax (float, optional): Maximum data value that colormap covers. Defaults to None.
        dx (float, optional): Size per pixel of the image data. If scalebar
            is required, `dx` and `units` must be sepcified. Defaults to None.
        units (str, optional): Units of `dx`. Defaults to None.
        dimension (str, optional): dimension of `dx` and `units`.
            Options include :
                - "si" : scale bar showing km, m, cm, etc.
                - "imperial" : scale bar showing in, ft, yd, mi, etc.
                - "si-reciprocal" : scale bar showing 1/m, 1/cm, etc.
                - "angle" : scale bar showing °, ʹ (minute of arc) or ʹʹ (second of arc).
                - "pixel" : scale bar showing px, kpx, Mpx, etc.
            Defaults to None.
        describe (bool, optional): Brief statistical description of the original and filterd
            image. Defaults to False.
        cbar (bool, optional): Specify if a colorbar is required or not.
            Defaults to True.
        orientation (str, optional): Specify the orientaion of colorbar.
            Option include :
                - 'h' or 'horizontal' for a horizontal colorbar to the bottom of the image.
                - 'v' or 'vertical' for a vertical colorbar to the right of the image.
            Defaults to 'v'.
        cbar_label (str, optional): Colorbar label. Defaults to None.
        cbar_fontdict (dict, optional): Font specifications for colorbar label - `cbar_label`.
            Defaults to None.
        cbar_ticks (list, optional): List of colorbar ticks. If None, min and max of
            the data are used. If `vmin` and `vmax` are specified, `vmin` and `vmax` values
            are used for colorbar ticks. Defaults to None.
        showticks (bool, optional): Show image x-y axis ticks. Defaults to False.
        despine (bool, optional): Remove axes spines from image axes as well as colorbar axes.
            Defaults to True.
        title (str, optional): Image title. Defaults to None.
        title_fontdict (dict, optional): Font specifications for `title`. Defaults to None.
        **kwargs : Any additional parameters to be passed to the specific filter chosen.
            For instance, "sigma" or "size" or "mode" etc.

    Raises:
        TypeError: if `filter` is not a string type
        TypeError: if `fft` is not a bool type
        NotImplementedError: if a `filter` that is not implemented is specified

    Returns:
        (tuple): tuple containing:

            (`matplotlib.figure.Figure`): Matplotlib figure.
            (`matplotlib.axes.Axes`): Matplotlib axes where the image is drawn.
            (`numpy.array`): Filtered image data

    Example:
        >>> import seaborn_image as isns
        >>> isns.filterplot(data) # use default gaussian filter
        >>> isns.filterplot(data, "percentile", percentile=35) # specify a filter with specific parameter
        >>> isns.filterplot(data, dx=3, units="um") # specify scalebar for the filterplot
    """

    if not isinstance(filter, str):
        raise TypeError("filter must be a string")

    if not isinstance(describe, bool):
        raise TypeError("describe must be a bool - 'True' or 'False")

    # check if the filter is implemented in seaborn-image
    if filter not in implemented_filters:
        raise NotImplementedError(
            f"'{filter}' filter is not implemented. Following are implented: {implemented_filters}"
        )

    else:
        if filter == "sobel":
            func_kwargs = {}
            func_kwargs.update(**kwargs)

            filtered_data = ndi.sobel(data, **func_kwargs)

        elif filter == "gaussian":
            func_kwargs = {}
            if "sigma" not in kwargs:  # assign sensible default if user didn't specify
                func_kwargs.update({"sigma": 1})
            func_kwargs.update(**kwargs)

            filtered_data = ndi.gaussian_filter(data, **func_kwargs)

        elif filter == "median":
            func_kwargs = {}
            if "size" not in kwargs:  # assign sensible default if user didn't specify
                func_kwargs.update({"size": 5})
            func_kwargs.update(**kwargs)

            filtered_data = ndi.median_filter(data, **func_kwargs)

        elif filter == "max":
            func_kwargs = {}
            if "size" not in kwargs:  # assign sensible default if user didn't specify
                func_kwargs.update({"size": 5})
            func_kwargs.update(**kwargs)

            filtered_data = ndi.maximum_filter(data, **func_kwargs)

        elif filter == "diff_of_gaussians":
            func_kwargs = {}
            if (
                "low_sigma" not in kwargs
            ):  # assign sensible default if user didn't specify
                func_kwargs.update({"low_sigma": 1})
            func_kwargs.update(**kwargs)

            filtered_data = difference_of_gaussians(data, **func_kwargs,)

        elif filter == "gaussian_gradient_magnitude":
            func_kwargs = {}
            if "sigma" not in kwargs:  # assign sensible default if user didn't specify
                func_kwargs.update({"sigma": 1})
            func_kwargs.update(**kwargs)

            filtered_data = ndi.gaussian_gradient_magnitude(data, **func_kwargs)

        elif filter == "gaussian_laplace":
            func_kwargs = {}
            if "sigma" not in kwargs:  # assign sensible default if user didn't specify
                func_kwargs.update({"sigma": 1})
            func_kwargs.update(**kwargs)

            filtered_data = ndi.gaussian_laplace(data, **func_kwargs)

        elif filter == "laplace":
            func_kwargs = {}
            func_kwargs.update(**kwargs)

            filtered_data = ndi.laplace(data, **func_kwargs)

        elif filter == "min":
            func_kwargs = {}
            if "size" not in kwargs:  # assign sensible default if user didn't specify
                func_kwargs.update({"size": 5})
            func_kwargs.update(**kwargs)

            filtered_data = ndi.minimum_filter(data, **func_kwargs)

        elif filter == "percentile":
            func_kwargs = {}
            if (
                "percentile" not in kwargs
            ):  # assign sensible default if user didn't specify
                func_kwargs.update({"percentile": 10})
            if "size" or "footprint" not in kwargs:
                warnings.warn(
                    "'size' or 'footprint' not specified; using 'size'=10", UserWarning
                )
                func_kwargs.update({"size": 10})
            func_kwargs.update(**kwargs)

            filtered_data = ndi.percentile_filter(data, **func_kwargs)

        elif filter == "prewitt":
            func_kwargs = {}
            func_kwargs.update(**kwargs)

            filtered_data = ndi.prewitt(data, **func_kwargs)

        elif filter == "rank":
            func_kwargs = {}
            if "rank" not in kwargs:  # assign sensible default if user didn't specify
                func_kwargs.update({"rank": 1})
            if "size" or "footprint" not in kwargs:
                warnings.warn(
                    "'size' or 'footprint' not specified; using 'size'=10", UserWarning
                )
                func_kwargs.update({"size": 10})
            func_kwargs.update(**kwargs)

            filtered_data = ndi.rank_filter(data, **func_kwargs)

        elif filter == "uniform":
            func_kwargs = {}
            func_kwargs.update(**kwargs)

            filtered_data = ndi.uniform_filter(data, **func_kwargs)

    # finally, plot the filtered image
    f, ax, cax = imgplot(
        filtered_data,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        dx=dx,
        units=units,
        dimension=dimension,
        describe=False,
        cbar=cbar,
        orientation=orientation,
        cbar_label=cbar_label,
        cbar_fontdict=cbar_fontdict,
        cbar_ticks=cbar_ticks,
        showticks=showticks,
        despine=despine,
    )

    # Provide basic statistical results
    if describe:  # TODO move all stats to separate file
        result_1 = ss.describe(filtered_data.flatten())
        print("Original Image")
        print(f"No. of Obs. : {result_1.nobs}")
        print(f"Min. Value : {result_1.minmax[0]}")
        print(f"Max. Value : {result_1.minmax[1]}")
        print(f"Mean : {result_1.mean}")
        print(f"Variance : {result_1.variance}")
        print(f"Skewness : {result_1.skewness}")

    return ax, cax, filtered_data


def fftplot(
    data,
    *,
    ax=None,
    cmap=None,
    cbar=True,
    cbar_label=None,
    cbar_fontdict=None,
    cbar_ticks=None,
    showticks=False,
    title=None,
    title_fontdict=None,
):

    if cmap is None:
        cmap = "sunset-dark"

    # window image to improve fft
    w_data = data * window(
        "hann", data.shape
    )  # TODO provide user option for choosing window

    # perform fft
    data_f_mag = fftshift(np.abs(fftn(w_data)))

    f, ax, cax = imgplot(
        np.log(data_f_mag),
        ax=ax,
        cmap=cmap,
        cbar=cbar,
        cbar_fontdict=cbar_fontdict,
        cbar_ticks=cbar_ticks,
        showticks=showticks,
        describe=False,
        title=title,
        title_fontdict=title_fontdict,
    )

    return ax, cax
