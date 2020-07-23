import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
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

    if not isinstance(filter, str):
        raise TypeError
    if not isinstance(fft, bool):
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

    if fft:
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
            cbar=cbar,
            cbar_label=cbar_label,
            cbar_fontdict=cbar_fontdict,
            cbar_ticks=cbar_ticks,
            showticks=showticks,
            title=title2,
            title_fontdict=title_fontdict,
        )

    return f, ax, filtered_data
