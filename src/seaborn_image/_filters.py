import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import scipy.stats as ss
from scipy.fftpack import fftn, fftshift
from skimage.filters import difference_of_gaussians, window

from ._general import imgplot


def filterplot(
    data,
    filter="gaussian",
    fft=False,
    cmap=None,
    vmin=None,
    vmax=None,
    dx=None,
    units=None,
    dimension=None,
    describe=False,
    cbar=True,
    cbar_label=None,
    cbar_fontdict=None,
    cbar_ticks=None,
    showticks=False,
    title1="Original Image",
    title2="Filtered Image",
    title_fontdict=None,
    **kwargs,
):
    """
    Plot original and filterd data as 2-D image with option to view the
    fast-fourier transform. Scalebar, colorbar, titles can be added and
    configured similar to `imgplot`

    Args:
        data: Image data (array-like). Supported array shapes are all
            `matplotlib.pyplot.imshow` array shapes
        filter (str, optional): Image filter to be used for processing.
            Defaults to "gaussian".
            Options include: "sobel", "gaussian", "median", "max", "diff_of_gaussians"
        fft (bool, optional): If True, fast-fourier transform of the original image and
            the filtered image is displayed. Defaults to False.
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
        cbar_label (str, optional): Colorbar label. Defaults to None.
        cbar_fontdict (dict, optional): Font specifications for colorbar label - `cbar_label`.
            Defaults to None.
        cbar_ticks (list, optional): List of colorbar ticks. If None, min and max of
            the data are used. If `vmin` and `vmax` are specified, `vmin` and `vmax` values
            are used for colorbar ticks. Defaults to None.
        showticks (bool, optional): Show image x-y axis ticks. Defaults to False.
        title (str, optional): Image title. Defaults to None.
        title1 (str, optional): Original image title. Defaults to "Original Image".
        title2 (str, optional): Filtered image title. Defaults to "Filtered Image".
        title_fontdict (dict, optional): Font specifications for `title`. Defaults to None.

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
        >>> isns.filterplot(data, filter="sobel", fft=True) # specify a filter and view fft
        >>> isns.filterplot(data, dx=3, units="um") # specify scalebar for the filterplot
    """

    if not isinstance(filter, str):
        raise TypeError
    if not isinstance(fft, bool):
        raise TypeError
    if not isinstance(describe, bool):
        raise TypeError

    # kwargs across filters
    axis = kwargs.get("axis", -1)
    mode = kwargs.get("mode", "reflect")
    cval = kwargs.get("cval", 0.0)
    size = kwargs.get("size", 5)
    footprint = kwargs.get("footprint", None)
    origin = kwargs.get("origin", 0)
    sigma = kwargs.get("sigma", 1)
    order = kwargs.get("order", 0)
    truncate = kwargs.get("truncate", 4.0)
    multichannel = kwargs.get("multichannel", False)
    low_sigma = kwargs.get("low_sigma", 1)
    high_sigma = kwargs.get("high_sigma", None)

    _implemented_filters = ["sobel", "gaussian", "median", "max", "diff_of_gaussians"]

    if filter not in _implemented_filters:
        raise NotImplementedError(
            f"'{filter}' filter is not implemented. Following are implented: {_implemented_filters}"
        )

    else:
        if filter == "sobel":
            filtered_data = ndi.sobel(data, axis=axis, mode=mode, cval=cval)

        elif filter == "gaussian":
            filtered_data = ndi.gaussian_filter(
                data, sigma=sigma, order=order, mode=mode, cval=cval, truncate=truncate
            )

        elif filter == "median":
            filtered_data = ndi.median_filter(
                data, size=size, mode=mode, cval=cval, origin=origin
            )

        elif filter == "max":
            filtered_data = ndi.maximum_filter(
                data,
                size=size,
                footprint=footprint,
                mode=mode,
                cval=cval,
                origin=origin,
            )

        elif filter == "diff_of_gaussians":
            filtered_data = difference_of_gaussians(
                data,
                low_sigma=low_sigma,
                high_sigma=high_sigma,
                mode=kwargs.get("mode", "nearest"),
                cval=cval,
                multichannel=multichannel,
                truncate=truncate,
            )

    if fft:  # move to FacetGrid like in seaborn (?) TODO
        # window image to improve fft
        w_data = data * window("hann", data.shape)
        w_filtered_data = filtered_data * window("hann", data.shape)

        # perform fft
        data_f_mag = fftshift(np.abs(fftn(w_data)))
        filt_data_f_mag = fftshift(np.abs(fftn(w_filtered_data)))

        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        imgplot(
            data,
            ax=ax[0][0],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            dx=dx,
            units=units,
            dimension=dimension,
            cbar=cbar,
            cbar_label=cbar_label,
            cbar_fontdict=cbar_fontdict,
            cbar_ticks=cbar_ticks,
            showticks=showticks,
            title=title1,
            title_fontdict=title_fontdict,
        )
        imgplot(
            filtered_data,
            ax=ax[0][1],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            dx=dx,
            units=units,
            dimension=dimension,
            cbar=cbar,
            cbar_label=cbar_label,
            cbar_fontdict=cbar_fontdict,
            cbar_ticks=cbar_ticks,
            showticks=showticks,
            title=title2,
            title_fontdict=title_fontdict,
        )

        imgplot(
            np.log(data_f_mag),
            ax=ax[1][0],
            cmap="sunset-dark",
            cbar=cbar,
            showticks=showticks,
            title="Original FFT Magnitude (log)",
        )

        imgplot(
            np.log(filt_data_f_mag),
            ax=ax[1][1],
            cmap="sunset-dark",
            cbar=cbar,
            showticks=showticks,
            title="Filtered FFT Magnitude (log)",
        )

    else:
        f, ax = plt.subplots(1, 2, figsize=(10, 5))

        imgplot(
            data,
            ax=ax[0],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            dx=dx,
            units=units,
            dimension=dimension,
            cbar=cbar,
            cbar_label=cbar_label,
            cbar_fontdict=cbar_fontdict,
            cbar_ticks=cbar_ticks,
            showticks=showticks,
            title=title1,
            title_fontdict=title_fontdict,
        )
        imgplot(
            filtered_data,
            ax=ax[1],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            dx=dx,
            units=units,
            dimension=dimension,
            cbar=cbar,
            cbar_label=cbar_label,
            cbar_fontdict=cbar_fontdict,
            cbar_ticks=cbar_ticks,
            showticks=showticks,
            title=title2,
            title_fontdict=title_fontdict,
        )

    if describe:
        result_1 = ss.describe(data.flatten())
        print("Original Image")
        print(f"No. of Obs. : {result_1.nobs}")
        print(f"Min. Value : {result_1.minmax[0]}")
        print(f"Max. Value : {result_1.minmax[1]}")
        print(f"Mean : {result_1.mean}")
        print(f"Variance : {result_1.variance}")
        print(f"Skewness : {result_1.skewness}")

        result_2 = ss.describe(data.flatten())
        print("Flitered Image")
        print(f"No. of Obs. : {result_2.nobs}")
        print(f"Min. Value : {result_2.minmax[0]}")
        print(f"Max. Value : {result_2.minmax[1]}")
        print(f"Mean : {result_2.mean}")
        print(f"Variance : {result_2.variance}")
        print(f"Skewness : {result_2.skewness}")

    return f, ax, filtered_data
