import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import scipy.stats as ss
from scipy.fftpack import fftn, fftshift
from skimage.filters import difference_of_gaussians, window

from ._general import imgplot

__all__ = ["filterplot", "fftplot", "implemented_filters"]

implemented_filters = {
    "sobel": ndi.sobel,
    "gaussian": ndi.gaussian_filter,
    "median": ndi.median_filter,
    "max": ndi.maximum_filter,
    "diff_of_gaussians": difference_of_gaussians,
    "gaussian_gradient_magnitude": ndi.gaussian_gradient_magnitude,
    "gaussian_laplace": ndi.gaussian_laplace,
    "laplace": ndi.laplace,
    "min": ndi.minimum_filter,
    "percentile": ndi.percentile_filter,
    "prewitt": ndi.prewitt,
    "rank": ndi.rank_filter,
    "uniform": ndi.uniform_filter,
}


def filterplot(
    data,
    filt="gaussian",
    *,
    ax=None,
    cmap=None,
    vmin=None,
    vmax=None,
    robust=False,
    perc=(2, 98),
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
        filt (str or callable, optional): Filter name or function to be applied.
            Filter name can be a string from `seaborn_image.implemented_filters` or a callable
            filter. Defaults to "gaussian".
        cmap (str or `matplotlib.colors.Colormap`, optional): Colormap for image.
            Can be a seaborn-image colormap or default matplotlib colormaps or
            any other colormap converted to a matplotlib colormap. Defaults to None.
        vmin (float, optional): Minimum data value that colormap covers. Defaults to None.
        vmax (float, optional): Maximum data value that colormap covers. Defaults to None.
        robust (bool, optional): If True and vmin or vmax are None, colormap range is calculated
            based on the percentiles defined in `percentile` parameter. Defaults to False.
        perc (tuple or list, optional): If `robust` is True, colormap range is calculated based
            on the percentiles specified instead of the extremes. Defaults to (2,98) - 2nd and 98th
            percentiles for min and max values.
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
        **kwargs : Any additional parameters to be passed to the specific filt chosen.
            For instance, "sigma" or "size" or "mode" etc.

    Raises:
        TypeError: if `filt` is not a string type or callable function
        NotImplementedError: if a `filt` that is not implemented is specified
        TypeError: if `describe` is not a `bool`

    Returns:
        (tuple): tuple containing:

            (`matplotlib.axes.Axes`): Matplotlib axes where the image is drawn.
            (`matplotlib.axes.Axes`): Colorbar axes
            (`numpy.array`): Filtered image data

    Example:
        >>> import seaborn_image as isns

        >>> # use default gaussian filter
        >>> isns.filterplot(data, sigma=3)

        >>> # specify a filt with specific parameter
        >>> isns.filterplot(data, "percentile", percentile=35, size=10)

        >>> # callable filter
        >>> import scipy.ndimage as ndi
        >>> isns.filterplot(data, ndi.gaussian_filter, sigma=3)

        >>> # specify scalebar for the filterplot
        >>> isns.filterplot(data, "sobel", dx=3, units="um")
    """

    if not isinstance(describe, bool):
        raise TypeError("describe must be a bool - 'True' or 'False")

    # check if the filt is implemented in seaborn-image
    if isinstance(filt, str) and filt not in implemented_filters.keys():
        raise NotImplementedError(
            f"'{filt}' filt is not implemented. Following are implented: {implemented_filters.keys()}"
        )

    else:
        if isinstance(filt, str) and filt in implemented_filters.keys():
            filt_func = implemented_filters[filt]

        elif callable(filt):
            filt_func = filt

        else:
            raise TypeError("'filt' must either be a string or a function")

        func_kwargs = {}
        func_kwargs.update(**kwargs)

        filtered_data = filt_func(data, **func_kwargs)

    # finally, plot the filtered image
    f, ax, cax = imgplot(
        filtered_data,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        robust=robust,
        perc=perc,
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
