import itertools
import warnings
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from ._filters import filterplot
from ._general import imgplot
from .utils import despine

__all__ = ["FilterGrid", "ImageGrid", "rgbplot"]


class ImageGrid:
    """
    Figure level : plot a collection of 2-D images or 3-D image data
    along a grid. This class also supports slicing of the 3-D image
    along different axis with variable step sizes and start/end indexes.

    Parameters
    ----------
    data :
        3-D Image data (array-like) or list of 2-D image data. Supported array shapes are all
        `matplotlib.pyplot.imshow` array shapes
    slices : int or list, optional
        If `data` is 3-D, `slices` will index the specific slice from the last axis and only plot
        the resulting images. If None, it will plot all the slices from the last axis, by default None
    axis : int, optional
        Axis along which the data will be sliced, by default -1
    step : int, optional
        Step along the given axis, by default 1
    start : int, optional
        Starting index to select from the the data, by default None
    stop : int, optional
        Stopping index to select from the data, by default None
    map_func : callable, optional
        Transform input image data using this function. All function arguments must be passed as kwargs.
    col_wrap : int, optional
        Number of columns to display. Defaults to None.
    height : int or float, optional
        Size of the individual images. Defaults to 3.
    aspect : int or float, optional
        Aspect ratio of individual images. Defaults to 1.
    cmap : str or `matplotlib.colors.Colormap` or list, optional
        Image colormap. If input data is a list of images,
        `cmap` can be a list of colormaps. Defaults to None.
    robust : bool or list, optional
        If True, colormap range is calculated based on the percentiles
        defined in `perc` parameter. If input data is a list of images,
        robust can be a list of bools, by default False
    perc : tuple or list, optional
        If `robust` is True, colormap range is calculated based
        on the percentiles specified instead of the extremes, by default (2, 98) -
        2nd and 98th percentiles for min and max values. Can be a list of tuples, if
        input data is a list of images
    alpha : float or array-like, optional
        `matplotlib.pyplot.imshow` alpha blending value from 0 (transparent) to 1 (opaque),
        by default None
    origin : str, optional
        Image origin, by default None
    interpolation : str, optional
        `matplotlib.pyplot.imshow` interpolation method used, by default None
    dx : float or list, optional
        Size per pixel of the image data. If scalebar
        is required, `dx` and `units` must be sepcified.
        Can be a list of floats, if input data is a list of images.
        Defaults to None.
    units : str or list, optional
        Units of `dx`.
        Can be a list of str, if input data is a list of images.
        Defaults to None.
    dimension : str or list, optional
        Dimension of `dx` and `units`.
        Options include :
            - "si" : scale bar showing km, m, cm, etc.
            - "imperial" : scale bar showing in, ft, yd, mi, etc.
            - "si-reciprocal" : scale bar showing 1/m, 1/cm, etc.
            - "angle" : scale bar showing °, ʹ (minute of arc) or ʹʹ (second of arc).
            - "pixel" : scale bar showing px, kpx, Mpx, etc.
        Can be a list of str, if input data is a list of images.
        Defaults to None.
    cbar : bool or list, optional
        Specify if a colorbar is required or not.
        Can be a list of bools, if input data is a list of images.
        Defaults to True.
    orientation : str, optional
        Specify the orientaion of colorbar.
        Option include :
            - 'h' or 'horizontal' for a horizontal colorbar to the bottom of the image.
            - 'v' or 'vertical' for a vertical colorbar to the right of the image.
        Defaults to 'v'.
    cbar_log : bool, optional
        Log scale colormap and colorbar
    cbar_label : str or list, optional
        Colorbar label.
        Can be a list of str, if input data is a list of images.
        Defaults to None.
    cbar_ticks : list, optional
        List of colorbar ticks. Defaults to None.
    showticks : bool, optional
        Show image x-y axis ticks. Defaults to False.
    despine : bool, optional
        Remove axes spines from image axes as well as colorbar axes.
        Defaults to None.

    Returns
    -------
        A `seaborn_image.ImageGrid` object

    Raises
    ------
    ValueError
        If `data` is None
    ValueError
        If `data` has more than 3 dimensions
    ValueError
        If `data` contains a 3D image within a list of images
    ValueError
        If `axis` is not 0, 1, 2 or -1
    TypeError
        If `map_func` is not a callable object

    Examples
    --------

    Plot a collection of images

    .. plot::
        :context: close-figs

        >>> import seaborn_image as isns
        >>> pol = isns.load_image("polymer")
        >>> pl = isns.load_image("fluorescence")
        >>> g = isns.ImageGrid([pol, pl])

    Common properties across images

    .. plot::
        :context: close-figs

        >>> g = isns.ImageGrid([pol, pl], cmap="inferno")

    Different scalebars for different images

    .. plot::
        :context: close-figs

        >>> g = isns.ImageGrid([pol, pl], dx=[0.15, 0.1], units="um")

    Specify properties only for specific images

    .. plot::
        :context: close-figs

        >>> g = isns.ImageGrid([pol, pl], dx=[None, 100], units=[None, "nm"])

    Different colormaps and colorbar titles

    .. plot::
        :context: close-figs

        >>> g = isns.ImageGrid([pol, pl], cmap=["deep", "magma"], cbar_label=["Height (nm)", "PL Intensity"])

    Correct colormap for outliers

    .. plot::
        :context: close-figs

        >>> pol_out = isns.load_image("polymer outliers")
        >>> g = isns.ImageGrid([pol, pl, pol_out], robust=[False, False, True], perc=[None, None, (0.5, 99.5)])

    Plot 3-D images; control number of columns

    .. plot::
        :context: close-figs

        >>> cells = isns.load_image("cells")
        >>> g = isns.ImageGrid(cells, col_wrap=5, cbar=False)

    Plot specific slices of the 3-D data cube

    .. plot::
        :context: close-figs

        >>> g = isns.ImageGrid(cells, slices=[10, 20, 30], cbar=False)

    Slice along different axis

    .. plot::
        :context: close-figs

        >>> g = isns.ImageGrid(cells, slices=[0, 4, 10, 32], axis=0, cbar=False)

    Select indexes with a specifc step size

    .. plot::
        :context: close-figs

        >>> g = isns.ImageGrid(cells, step=3, cbar=False)

    Map a function to the image data

    .. plot::
        :context: close-figs

        >>> from skimage.exposure import adjust_gamma
        >>> g = isns.ImageGrid(
        ...             cells,
        ...             map_func=adjust_gamma,
        ...             gamma=0.5,
        ...             cbar=False,
        ...             height=1,
        ...             col_wrap=10)

    Change colorbar orientation

    .. plot::
        :context: close-figs

        >>> g = isns.ImageGrid([pol, pl], orientation="h")

    Change figure size using height

    .. plot::
        :context: close-figs

        >>> g = isns.ImageGrid([pol, pl], height=4.5)

    """

    def __init__(
        self,
        data,
        *,
        slices=None,
        axis=-1,
        step=1,
        start=None,
        stop=None,
        map_func=None,
        col_wrap=None,
        height=3,
        aspect=1,
        cmap=None,
        robust=False,
        perc=(2, 98),
        alpha=None,
        origin=None,
        interpolation=None,
        dx=None,
        units=None,
        dimension=None,
        cbar=True,
        orientation="v",
        cbar_log=False,
        cbar_label=None,
        cbar_ticks=None,
        showticks=False,
        despine=None,
        **kwargs,
    ):
        if data is None:
            raise ValueError("image data can not be None")

        if isinstance(
            data, (list, tuple)
        ):  # using 'Iterable' numpy was being picked up
            # check the number of images to be plotted
            _nimages = len(data)

        elif data.ndim > 3:
            raise ValueError("image data can not have more than 3 dimensions")

        elif data.ndim == 3:
            if slices is None:
                if axis not in [0, 1, 2, -1]:
                    raise ValueError("Incorrect 'axis'; must be either 0, 1, 2, or -1")
                # slice the image array along specified axis;
                # if start, stop and step are not provided, default is step=1
                data = data[
                    (slice(None),) * (axis % data.ndim) + (slice(start, stop, step),)
                ]
                slices = np.arange(data.shape[axis])

            # if a single slice is provided and
            # it is not an interable
            elif not isinstance(slices, Iterable):
                slices = [slices]

            _nimages = len(slices)

        else:
            # if data dim is not >2,
            # TODO issue user warning to use imgplot() instead?
            _nimages = 1

        if map_func is not None:
            if not callable(map_func):
                raise TypeError("`map_func` must be a callable function object")

        # if no column wrap specified
        # set it to default 3
        if col_wrap is None:
            col_wrap = 3

            # don't create extra columns when there aren't enough images
        if col_wrap > _nimages:
            col_wrap = _nimages

        # Compute the grid shape if col_wrap is specified
        ncol = col_wrap
        nrow = int(np.ceil(_nimages / col_wrap))

        # Calculate the base figure size
        figsize = (ncol * height * aspect, nrow * height)

        fig = plt.figure(figsize=figsize)
        axes = fig.subplots(nrow, ncol, squeeze=False)

        # Public API
        self.data = data
        self.fig = fig
        self.axes = axes
        self.slices = slices
        self.axis = axis
        self.step = step
        self.start = start
        self.stop = stop
        self.col_wrap = col_wrap
        self.height = height
        self.aspect = aspect

        self.cmap = cmap
        self.robust = robust
        self.perc = perc
        self.alpha = alpha
        self.origin = origin
        self.interpolation = interpolation
        self.dx = dx
        self.units = units
        self.dimension = dimension
        self.cbar = cbar
        self.orientation = orientation
        self.cbar_log = cbar_log
        self.cbar_label = cbar_label
        self.cbar_ticks = cbar_ticks
        self.showticks = showticks
        self.despine = despine

        self._nrow = nrow
        self._ncol = ncol
        self._nimages = _nimages

        # map function to input data
        if map_func is not None:
            self._map_func_to_data(map_func, **kwargs)

        self._map_img_to_grid()
        self._cleanup_extra_axes()
        self._finalize_grid()

        return

    def _map_img_to_grid(self):
        """Map image data cube to the image grid."""

        _cmap = self.cmap
        _robust = self.robust
        _perc = self.perc
        _dx = self.dx
        _units = self.units
        _dimension = self.dimension
        _cbar = self.cbar
        _cbar_log = self.cbar_log
        _cbar_label = self.cbar_label

        for i in range(self._nimages):
            ax = self.axes.flat[i]

            if isinstance(self.data, (list, tuple)):
                _d = self.data[i]

                # check if the image has more than 2 dimensions
                if _d.ndim > 2:
                    raise ValueError(
                        "can not plot multiple 3D images. Supply an individual 3D image"
                    )

                if isinstance(self.cmap, (list, tuple)):
                    _cmap = self.cmap[i]

                if isinstance(self.robust, (list, tuple)):
                    _robust = self.robust[i]

                if isinstance(self.perc, (list)):
                    _perc = self.perc[i]

                if isinstance(self.dx, (list, tuple)):
                    _dx = self.dx[i]

                if isinstance(self.units, (list, tuple)):
                    _units = self.units[i]

                if isinstance(self.dimension, (list, tuple)):
                    _dimension = self.dimension[i]

                if isinstance(self.cbar, (list, tuple)):
                    _cbar = self.cbar[i]

                if isinstance(self.cbar_log, (list, tuple)):
                    _cbar_log = self.cbar_log[i]

                if isinstance(self.cbar_label, (list, tuple)):
                    _cbar_label = self.cbar_label[i]

            elif self.data.ndim == 3:
                if self.axis == 0:
                    _d = self.data[self.slices[i], :, :]
                elif self.axis == 1:
                    _d = self.data[:, self.slices[i], :]
                elif self.axis == 2 or self.axis == -1:
                    _d = self.data[:, :, self.slices[i]]

            else:
                # if a single 2D image is supplied
                # TODO issue a warning and direct the user to imgplot()
                _d = self.data

            imgplot(
                _d,
                ax=ax,
                cmap=_cmap,
                robust=_robust,
                perc=_perc,
                alpha=self.alpha,
                origin=self.origin,
                interpolation=self.interpolation,
                dx=_dx,
                units=_units,
                dimension=_dimension,
                cbar=_cbar,
                orientation=self.orientation,
                cbar_log=_cbar_log,
                cbar_label=_cbar_label,
                cbar_ticks=self.cbar_ticks,
                showticks=self.showticks,
                despine=self.despine,
                describe=False,
            )

        # FIXME - for common colorbar
        # self.fig.colorbar(ax.images[0], ax=list(self.axes.flat), orientation=self.orientation)

        return

    def _map_func_to_data(self, map_func, **kwargs):
        """Transform image data using the map_func callable object."""
        # if data is a list or tuple of 2D images
        if isinstance(self.data, (list, tuple)):
            for i in range(len(self.data)):
                self.data[i] = map_func(self.data[i], **kwargs)

        # if data is 3D
        else:
            self.data = map_func(self.data, **kwargs)

    def _cleanup_extra_axes(self):
        """Clean extra axes that are generated if col_wrap is specified."""
        # check if there are any extra axes that need to be clened up
        _rem = (self._ncol * self._nrow) - self._nimages
        if _rem > 0:
            rem_ax = self.axes.flat[-_rem:]
            for i in range(len(rem_ax)):
                rem_ax[i].set_yticks([])
                rem_ax[i].set_xticks([])

                rem_ax[i].set_ylabel("")
                rem_ax[i].set_xlabel("")

                despine(ax=rem_ax[i])  # remove axes spines for the extra generated axes

    def _finalize_grid(self):
        """Finalize grid with tight layout."""
        self.fig.tight_layout()


def rgbplot(
    data,
    *,
    col_wrap=3,
    height=3,
    aspect=1,
    cmap=None,
    alpha=None,
    origin=None,
    interpolation=None,
    dx=None,
    units=None,
    dimension=None,
    cbar=True,
    orientation="v",
    cbar_label=None,
    cbar_ticks=None,
    showticks=False,
    despine=None,
):
    """Split and plot the red, green and blue channels of an
    RGB image.

    Parameters
    ----------
    data :
        RGB image data as 3-D array.
    col_wrap : int, optional
        Number of columns to display. Defaults to 3.
    height : int or float, optional
        Size of the individual images. Defaults to 3.
    aspect : int or float, optional
        Aspect ratio of individual images. Defaults to 1.
    cmap : str or `matplotlib.colors.Colormap` or list, optional
        Image colormap or a list of colormaps. Defaults to None.
    alpha : float or array-like, optional
        `matplotlib.pyplot.imshow` alpha blending value from 0 (transparent) to 1 (opaque),
        by default None
    origin : str, optional
        Image origin, by default None
    interpolation : str, optional
        `matplotlib.pyplot.imshow` interpolation method used, by default None
    dx : float or list, optional
        Size per pixel of the image data. If scalebar
        is required, `dx` and `units` must be sepcified.
        Can be a list of floats.
        Defaults to None.
    units : str or list, optional
        Units of `dx`.
        Can be a list of str.
        Defaults to None.
    dimension : str or list, optional
        Dimension of `dx` and `units`.
        Options include :
            - "si" : scale bar showing km, m, cm, etc.
            - "imperial" : scale bar showing in, ft, yd, mi, etc.
            - "si-reciprocal" : scale bar showing 1/m, 1/cm, etc.
            - "angle" : scale bar showing °, ʹ (minute of arc) or ʹʹ (second of arc).
            - "pixel" : scale bar showing px, kpx, Mpx, etc.
        Can be a list of str.
        Defaults to None.
    cbar : bool or list, optional
        Specify if a colorbar is required or not.
        Can be a list of bools.
        Defaults to True.
    orientation : str, optional
        Specify the orientaion of colorbar.
        Option include :
            - 'h' or 'horizontal' for a horizontal colorbar to the bottom of the image.
            - 'v' or 'vertical' for a vertical colorbar to the right of the image.
        Defaults to 'v'.
    cbar_label : str or list, optional
        Colorbar label.
        Can be a list of str.
        Defaults to None.
    cbar_ticks : list, optional
        List of colorbar ticks. Defaults to None.
    showticks : bool, optional
        Show image x-y axis ticks. Defaults to False.
    despine : bool, optional
        Remove axes spines from image axes as well as colorbar axes.
        Defaults to None.

    Returns
    -------
    `seaborn_image.ImageGrid`

    Raises
    ------
    ValueError
        If `data` dimension is not 3
    ValueError
        If `data` channels are not 3

    Examples
    --------

    Split and plot the channels of a RGB image

    .. plot::
        :context: close-figs

        >>> import seaborn_image as isns; isns.set_image(origin="upper")
        >>> from skimage.data import astronaut
        >>> g = isns.rgbplot(astronaut())

    Hide colorbar

    .. plot::
        :context: close-figs

        >>> g = isns.rgbplot(astronaut(), cbar=False)

    Change colormap

    .. plot::
        :context: close-figs

        >>> g = isns.rgbplot(astronaut(), cmap="deep")

    .. plot::
        :context: close-figs

        >>> g = isns.rgbplot(astronaut(), cmap=["inferno", "viridis", "ice"])

    Horizontal colorbar

    .. plot::
        :context: close-figs

        >>> g = isns.rgbplot(astronaut(), orientation="h")
    """

    if not data.ndim == 3:
        raise ValueError("imput image must be a RGB image")

    if data.shape[-1] != 3:
        raise ValueError("input image must be a RGB image")

    # if no cmap, assign reds, greens and blues cmap
    if cmap is None:
        cmap = ["Reds", "Greens", "Blues"]

    # split RGB channels
    _d = [data[:, :, 0], data[:, :, 1], data[:, :, 2]]

    g = ImageGrid(
        _d,
        height=height,
        aspect=aspect,
        col_wrap=col_wrap,
        cmap=cmap,
        alpha=alpha,
        origin=origin,
        interpolation=interpolation,
        dx=dx,
        units=units,
        dimension=dimension,
        cbar=cbar,
        orientation=orientation,
        cbar_label=cbar_label,
        cbar_ticks=cbar_ticks,
        showticks=showticks,
        despine=despine,
    )

    return g


# TODO provide common cbar option
# TODO allow gridspec_kws and subplot_kws


class FilterGrid(object):
    """Generate a grid of images with the specific filter applied to all
    the images. This class allows exploration of different parameters of
    the filters across the rows and columns of the grid. Additional filter
    parameters that are not to be varied can also be passed.

    Parameters
    ----------
    data :
        Image data (array-like). Supported array shapes are all
        `matplotlib.pyplot.imshow` array shapes
    filt : str or callable
        Filter name or function to be applied.
    row : str, optional
        Parameter name that is to be displayed
        along the row. Defaults to None.
    col : str, optional
        Parameter name that is to be displayed
        along the column. Defaults to None.
    col_wrap : int, optional
        Number of columns to display if `col`
        is not None and `row` is None. Defaults to None.
    height : int or float, optional
        Size of the individual images. Defaults to 3.
    aspect : int or float, optional
        Aspect ratio of individual images. Defaults to 1.
    cmap : str or `matplotlib.colors.Colormap`, optional
        Image colormap. Defaults to None.
    alpha : float or array-like, optional
        `matplotlib.pyplot.imshow` alpha blending value from 0 (transparent) to 1 (opaque),
        by default None
    origin : str, optional
        Image origin, by default None
    interpolation : str, optional
        `matplotlib.pyplot.imshow` interpolation method used, by default None
    dx : float, optional
        Size per pixel of the image data. If scalebar
        is required, `dx` and `units` must be sepcified. Defaults to None.
    units : str, optional
        Units of `dx`. Defaults to None.
    dimension : str, optional
        Dimension of `dx` and `units`.
        Options include :
            - "si" : scale bar showing km, m, cm, etc.
            - "imperial" : scale bar showing in, ft, yd, mi, etc.
            - "si-reciprocal" : scale bar showing 1/m, 1/cm, etc.
            - "angle" : scale bar showing °, ʹ (minute of arc) or ʹʹ (second of arc).
            - "pixel" : scale bar showing px, kpx, Mpx, etc.
        Defaults to None.
    cbar : bool, optional
        Specify if a colorbar is required or not.
        Defaults to True.
    orientation : str, optional
        Specify the orientaion of colorbar.
        Option include :
            - 'h' or 'horizontal' for a horizontal colorbar to the bottom of the image.
            - 'v' or 'vertical' for a vertical colorbar to the right of the image.
        Defaults to 'v'.
    cbar_log : bool, optional
        Log scale colormap and colorbar
    cbar_label : str, optional
        Colorbar label. Defaults to None.
    cbar_ticks : list, optional
        List of colorbar ticks. Defaults to None.
    showticks : bool, optional
        Show image x-y axis ticks. Defaults to False.
    despine : bool, optional
        Remove axes spines from image axes as well as colorbar axes.
        Defaults to None.
    **kwargs : Additional parameters as keyword arguments to be passed to the underlying filter specified.

     Returns
    -------
        A `seabron_image.FilterGrid` object

    Raises
    ------
    TypeError
        If `row` is not a str
    ValueError
        If `row` is specified without passing the parameter as a keyword argument
    TypeError
        If `col` is not a str
    ValueError
        If `col` is specified without passing the parameter as a keyword argument
    ValueError
        If `col_wrap` is specified when `row` is not `None`

    Examples
    --------
    Specify a filter with different parameters along the columns

    .. plot::
        :context: close-figs

        >>> import seaborn_image as isns
        >>> img = isns.load_image("polymer")
        >>> g = isns.FilterGrid(img, "median", col="size", size=[2,3,4,5])

    Or rows

    .. plot::
        :context: close-figs

        >>> g = isns.FilterGrid(img, "median", row="size", size=[2,3,4,5])

    Use `col_wrap` to control column display

    .. plot::
        :context: close-figs

        >>> g = isns.FilterGrid(img, "median", col="size", size=[2,3,4,5], col_wrap=3)

    Use `col` and `row` to display different parameters along the columns and rows

    .. plot::
        :context: close-figs

        >>> g = isns.FilterGrid(img,
        ...                     "percentile",
        ...                     row="percentile",
        ...                     col="size",
        ...                     percentile=[10,20,30],
        ...                     size=[20,25,30],)

    Specify additional keyword arguments for the filter

    .. plot::
        :context: close-figs

        >>> g = isns.FilterGrid(img, "median", col="size", size=[2,3,4,5], mode="reflect")

    General image controls such as changing colormap, scalebar, etc.

    .. plot::
        :context: close-figs

        >>> g = isns.FilterGrid(
        ...                     img,
        ...                     "median",
        ...                     col="size",
        ...                     size=[2,3,4,5],
        ...                     cmap="inferno",
        ...                     dx=15,
        ...                     units="nm")
    """

    def __init__(
        self,
        data,
        filt,
        *,
        row=None,
        col=None,
        col_wrap=None,
        height=3,
        aspect=1,
        cmap=None,
        alpha=None,
        origin=None,
        interpolation=None,
        dx=None,
        units=None,
        dimension=None,
        cbar=True,
        orientation="v",
        cbar_log=False,
        cbar_label=None,
        cbar_ticks=None,
        showticks=False,
        despine=None,
        **kwargs,
    ):

        if data is None:
            raise ValueError("image data can not be None")

        if filt is None:
            raise ValueError("'filt' can not be None; must be a string or callable")

        row_params = []
        if row is not None:
            if not isinstance(row, str):
                raise TypeError("'row' parameter must be a string")
            if row not in kwargs:
                err = f"Specified '{row}' as 'row' without passing it as a kwargs"
                raise ValueError(err)
            else:
                row_params.extend(kwargs[f"{row}"])

        col_params = []
        if col is not None:
            if not isinstance(col, str):
                raise TypeError("'col' parameter must be a string")
            if col not in kwargs:
                err = f"Specified '{col}' as 'col' without passing it as a kwargs"
                raise ValueError(err)
            else:
                col_params.extend(kwargs[f"{col}"])

        # Compute the grid shape like FacetGrid
        nrow = 1 if row is None else len(kwargs[f"{row}"])
        ncol = 1 if col is None else len(kwargs[f"{col}"])

        # col_wrap can not be used with row option
        if col_wrap is not None:
            if row is not None:
                err = "Cannot use `row` and `col_wrap` together"
                raise ValueError(err)

            # recompute the grid shape if col_wrap is specified
            ncol = col_wrap
            nrow = int(np.ceil(len(kwargs[f"{col}"]) / col_wrap))

        # Calculate the base figure size
        figsize = (ncol * height * aspect, nrow * height)

        fig = plt.figure(figsize=figsize)
        axes = fig.subplots(nrow, ncol, squeeze=False)

        product_params = []
        if row and col:
            _p = itertools.product(row_params, col_params)
            for _r, _c in _p:
                product_params.append([_r, _c])
        elif row:
            for _r in row_params:
                product_params.append([_r])
        elif col:
            for _c in col_params:
                product_params.append([_c])

        product_params = np.array(product_params, dtype=object)

        # check if any additional kwargs are passed
        # that need to be passed to the underlying filter
        additional_kwargs = {}
        for k, v in kwargs.items():
            if row and col:
                if k not in [row, col]:
                    additional_kwargs.update({k: v})
            elif row:
                if k not in row:
                    additional_kwargs.update({k: v})
            elif col:
                if k not in col:
                    additional_kwargs.update({k: v})

        # Public API
        self.data = data
        self.filt = filt
        self.fig = fig
        self.axes = axes
        self.row = row
        self.col = col
        self.col_wrap = col_wrap
        self.param_product = product_params
        self.additional_kwargs = additional_kwargs
        self.height = height
        self.aspect = aspect

        self.cmap = cmap
        self.alpha = alpha
        self.origin = origin
        self.interpolation = interpolation
        self.dx = dx
        self.units = units
        self.dimension = dimension
        self.cbar = cbar
        self.orientation = orientation
        self.cbar_log = cbar_log
        self.cbar_label = cbar_label
        self.cbar_ticks = cbar_ticks
        self.showticks = showticks
        self.despine = despine

        self._nrow = nrow
        self._ncol = ncol

        self.map_filter_to_grid()
        self._cleanup_extra_axes()
        self._finalize_grid()

        return

    def map_filter_to_grid(self):
        """Map specified filter with row and col paramters
        to the image grid.
        """

        # any additional kwargs that need to be passed
        # to the underlying filter
        func_kwargs = self.additional_kwargs

        if self.row is None and self.col is None:
            imgplot(
                self.data, ax=self.axes.flat[0]
            )  # since squeeze is False, array needs to be flattened and indexed

        for i in range(len(self.param_product)):
            ax = self.axes.flat[i]
            p = self.param_product[i]

            # plot only col vars
            if self.row is None:
                func_kwargs.update({self.col: p[0]})
                self._plot(ax=ax, **func_kwargs)

                ax.set_title(f"{self.col} : {p[0]}")

            # plot only row vars
            if self.col is None:
                func_kwargs.update({self.row: p[0]})
                self._plot(ax=ax, **func_kwargs)

                ax.set_title(f"{self.row} : {p[0]}")

            # when both row and col vars are specified
            if self.row and self.col:
                func_kwargs.update({self.row: p[0], self.col: p[1]})
                self._plot(ax=ax, **func_kwargs)

                # set row labels only to the outermost column
                if not i % self._nrow:
                    ax.set_ylabel(f"{self.row} : {p[0]}")

                # set column labels only to the top row
                if i < self._ncol:
                    ax.set_title(f"{self.col} : {p[1]}")

        # FIXME - for common colorbar
        # self.fig.colorbar(ax.images[0], ax=list(self.axes.flat), orientation=self.orientation)

        return

    def _plot(self, ax, **func_kwargs):
        """Helper function to call the underlying filterplot

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axis to plot filtered image
        """

        filterplot(
            self.data,
            self.filt,
            ax=ax,
            cmap=self.cmap,
            alpha=self.alpha,
            origin=self.origin,
            interpolation=self.interpolation,
            dx=self.dx,
            units=self.units,
            dimension=self.dimension,
            cbar=self.cbar,
            orientation=self.orientation,
            cbar_log=self.cbar_log,
            cbar_label=self.cbar_label,
            cbar_ticks=self.cbar_ticks,
            showticks=self.showticks,
            despine=self.despine,
            **func_kwargs,
        )
        return

    def _cleanup_extra_axes(self):
        """Clean extra axes that are generated if col_wrap is specified."""
        if self.col_wrap is not None:
            # check if there are any extra axes that need to be clened up
            _rem = (self.col_wrap * self._nrow) - len(self.param_product)
            if _rem > 0:
                rem_ax = self.axes.flat[-_rem:]
                for i in range(len(rem_ax)):
                    rem_ax[i].set_yticks([])
                    rem_ax[i].set_xticks([])

                    rem_ax[i].set_ylabel("")
                    rem_ax[i].set_xlabel("")

                    despine(
                        ax=rem_ax[i]
                    )  # remove axes spines for the extra generated axes

    def _finalize_grid(self):
        """Finalize grid with tight layout."""
        self.fig.tight_layout()
