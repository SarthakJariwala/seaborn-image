import itertools
import warnings
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from ._filters import filterplot
from ._general import imgplot
from .utils import despine

__all__ = ["ParamGrid", "ImageGrid", "rgbplot", "FilterGrid"]


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
    map_func : callable or list/tuple or callables, optional
        Transform input image data using this function. All function arguments must be passed as map_func_kw.
    map_func_kw : dict or list/tuple of dict, optional
        kwargs to pass on to `map_func`. Must be dict for a single `map_func` and a list/tuple of dicts for a list/tuple of `map_func`
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
    vmin : float or list of floats, optional
        Minimum data value that colormap covers, by default None
    vmax : float or list of floats, optional
        Maximum data value that colormap covers, by default None
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
        If `map_func` is not a callable object or a list/tuple of callable objects
    ValueError
        If `map_func` is a list/tuple of callable objects when `data` is 3D

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

    Visulaize image intensities relative to other images on the grid

    .. plot::
        :context: close-figs

        >>> g = isns.ImageGrid(cells, vmin=0, vmax=1, height=1, col_wrap=5)

    Map a function to the image data

    .. plot::
        :context: close-figs

        >>> from skimage.exposure import adjust_gamma
        >>> g = isns.ImageGrid(
        ...             cells,
        ...             map_func=adjust_gamma,
        ...             map_func_kw={"gamma" : 0.5},
        ...             cbar=False,
        ...             height=1,
        ...             col_wrap=10)

    Map a list of functions to the input data. Pass function kwargs to `map_func_kw`.

    .. plot::
        :context: close-figs

        >>> from skimage.filters import meijering, sato, frangi, hessian
        >>> retina = isns.load_image("retina-gray")
        >>> g = isns.ImageGrid(
        ...             retina,
        ...             map_func=[meijering, sato, frangi, hessian],
        ...             col_wrap=4,
        ...             map_func_kw=[{"mode" : "reflect", "sigmas" : [1]} for _ in range(4)])

    If no kwargs are required for one or more of the functions, use `None`.

    .. plot::
        :context: close-figs

        >>> g = isns.ImageGrid(
        ...             retina,
        ...             map_func=[meijering, sato, frangi, hessian],
        ...             col_wrap=4,
        ...             map_func_kw=[{"mode" : "reflect", "sigmas" : [1]}, None, None, None])

    Apply a list of filters to a list of input images.

    .. plot::
        :context: close-figs

        >>> from skimage.filters import gaussian, median
        >>> g = isns.ImageGrid(
        ...             [pol, pl, retina],
        ...             map_func=[gaussian, median, hessian],
        ...             dx=[15, 100, None],
        ...             units="nm")

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
        map_func_kw=None,
        col_wrap=None,
        height=3,
        aspect=1,
        cmap=None,
        robust=False,
        perc=(2, 98),
        alpha=None,
        origin=None,
        vmin=None,
        vmax=None,
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
    ):
        if data is None:
            raise ValueError("image data can not be None")

        if isinstance(
            data, (list, tuple)
        ):  # using 'Iterable' numpy was being picked up
            # check the number of images to be plotted
            _nimages = len(data)

            # --- List/Tuple of 2D images with a List/Tuple of map_func ---
            # change the number of images on the grid accordingly
            map_func_type = self._check_map_func(map_func, map_func_kw)
            if map_func_type == "list/tuple":
                _nimages = len(data) * len(map_func)
                # no of columns should either be len of data list or len of map_func list
                # whichever is higher
                if col_wrap is None:
                    col_wrap = (
                        len(map_func) if len(map_func) >= len(data) else len(data)
                    )

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

            # ---- 3D image with an individual map_func ----
            map_func_type = self._check_map_func(map_func, map_func_kw)
            # raise a ValueError if a list of map_func is provided for 3d image
            # TODO - support multiple map_func if "chaining"?
            if map_func_type == "list/tuple":
                raise ValueError(
                    "Can not map multiple functions to a 3D image. Please provide a single `map_func`"
                )

        else:
            # if data dim is not >2,
            _nimages = 1

            # ---- 2D image with a list/tuple of map_func or individual map_func ------
            # check if map_func is a list/tuple of callables
            # and assign the new number of images
            map_func_type = self._check_map_func(map_func, map_func_kw)
            if map_func_type == "list/tuple":
                _nimages = len(map_func)
                # no of columns should now be len of map_func list
                col_wrap = len(map_func) if col_wrap is None else col_wrap

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
        self.vmin = vmin
        self.vmax = vmax
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
            self._map_func_to_data(map_func, map_func_kw)

        self._map_img_to_grid()
        self._cleanup_extra_axes()
        self._finalize_grid()

    def _check_map_func(self, map_func, map_func_kw):
        "Check if `map_func` passed is a list/tuple of callables or individual callable"
        if map_func is not None:
            if isinstance(map_func, (list, tuple)):
                for func in map_func:
                    if not callable(func):
                        raise TypeError(f"{func} must be a callable function object")
                if map_func_kw is not None:
                    if not isinstance(map_func_kw, (list, tuple)):
                        raise TypeError(
                            "`map_func_kw` must be list/tuple of dictionaries"
                        )
                    if len(map_func_kw) != len(map_func):
                        raise ValueError(
                            "number of `map_func_kw` passed must be the same as the number of `map_func` objects"
                        )
                return "list/tuple"

            elif callable(map_func):
                if map_func_kw is not None:
                    if not isinstance(map_func_kw, dict):
                        raise TypeError(
                            "`map_func_kw` must be a dictionary when a single `map_func` is passed as input"
                        )
                return "callable"

            else:
                raise TypeError(
                    "`map_func` must either be a callable object or a list/tuple of callable objects"
                )

    def _map_img_to_grid(self):
        """Map image data cube to the image grid."""

        _cmap = self.cmap
        _robust = self.robust
        _perc = self.perc
        _vmin = self.vmin
        _vmax = self.vmax
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

            if isinstance(self.cmap, (list, tuple)):
                self._check_len_wrt_n_images(self.cmap)
                _cmap = self.cmap[i]

            if isinstance(self.robust, (list, tuple)):
                self._check_len_wrt_n_images(self.robust)
                _robust = self.robust[i]

            if isinstance(self.vmin, (list, tuple)):
                self._check_len_wrt_n_images(self.vmin)
                _vmin = self.vmin[i]

            if isinstance(self.vmax, (list, tuple)):
                self._check_len_wrt_n_images(self.vmax)
                _vmax = self.vmax[i]

            if isinstance(self.perc, (list)):
                self._check_len_wrt_n_images(self.perc)
                _perc = self.perc[i]

            if isinstance(self.dx, (list, tuple)):
                self._check_len_wrt_n_images(self.dx)
                _dx = self.dx[i]

            if isinstance(self.units, (list, tuple)):
                self._check_len_wrt_n_images(self.units)
                _units = self.units[i]

            if isinstance(self.dimension, (list, tuple)):
                self._check_len_wrt_n_images(self.dimension)
                _dimension = self.dimension[i]

            if isinstance(self.cbar, (list, tuple)):
                self._check_len_wrt_n_images(self.cbar)
                _cbar = self.cbar[i]

            if isinstance(self.cbar_log, (list, tuple)):
                self._check_len_wrt_n_images(self.cbar_log)
                _cbar_log = self.cbar_log[i]

            if isinstance(self.cbar_label, (list, tuple)):
                self._check_len_wrt_n_images(self.cbar_label)
                _cbar_label = self.cbar_label[i]

            _ = imgplot(
                _d,
                ax=ax,
                cmap=_cmap,
                robust=_robust,
                perc=_perc,
                vmin=_vmin,
                vmax=_vmax,
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
        # if self.cbar and self.vmin is not None and self.vmax is not None:
        #     print("here")
        #     self.fig.colorbar(_im.images[0], ax=list(self.axes.ravel()), orientation=self.orientation)

    def _check_len_wrt_n_images(self, param_list):
        """If a specific parameter is supplied as a list/tuple, check that the
        length of the parameter list is the same as the number of images that the parameter is mapped onto
        """

        if len(param_list) != self._nimages:
            raise AssertionError(
                f"If supplying a list/tuple, length of {param_list} must be {self._nimages}."
            )

    def _adjust_param_list_len(self, map_func):
        """
        If the input data and map_func are both list-like,
        modify the parameter list such as dx, units, etc such that
        the length of new parameter list is the same as the number of images.

        # For example -
        # if data -> [img1, img2], map_func -> [func1, func2, func3]
        # and dx = [dx1, dx2] # same as len(data)
        # then for plotting, dx needs to be expanded such that the len(dx) == len(data) * len(map_func)
        # so, new dx -> [dx1, dx2] * len(map_func)
        """
        if isinstance(self.dx, (list, tuple)):
            self.dx = self.dx * len(map_func)

        if isinstance(self.units, (list, tuple)):
            self.units = self.units * len(map_func)

        if isinstance(self.dimension, (list, tuple)):
            self.dimension = self.dimension * len(map_func)

        if isinstance(self.cbar, (list, tuple)):
            self.cbar = self.cbar * len(map_func)

        if isinstance(self.cbar_label, (list, tuple)):
            self.cbar_label = self.cbar_label * len(map_func)

        if isinstance(self.cbar_log, (list, tuple)):
            self.cbar_log = self.cbar_log * len(map_func)

    def _map_func_to_data(self, map_func, map_func_kw):
        """Transform image data using the map_func callable object."""
        # if data is a list or tuple of 2D images
        if isinstance(self.data, (list, tuple)):
            if self._check_map_func(map_func, map_func_kw) == "list/tuple":
                self._adjust_param_list_len(map_func)
                _d = self.data
                # only pass on kwargs if not None
                if map_func_kw is not None:
                    # check if one of the supplied kwargs in the list is None
                    # if None - change it to empty {}
                    map_func_kw = [{} if kw is None else kw for kw in map_func_kw]
                    self.data = [
                        func(img, **kwargs)
                        for func, kwargs in zip(map_func, map_func_kw)
                        for img in _d
                    ]
                else:
                    self.data = [func(img) for func in map_func for img in _d]
            else:  # if map_func is callable
                for i in range(len(self.data)):
                    # only pass on kwargs if not None
                    if map_func_kw is not None:
                        self.data[i] = map_func(self.data[i], **map_func_kw)
                    else:
                        self.data[i] = map_func(self.data[i])

        # if data is 3D or 2D and map_func is single callable
        else:
            if self._check_map_func(map_func, map_func_kw) == "callable":
                # only pass on kwargs if not None
                if map_func_kw is not None:
                    self.data = map_func(self.data, **map_func_kw)
                else:
                    self.data = map_func(self.data)
            # list of callables -- only for 2D image
            else:
                _d = self.data
                # only pass on kwargs if not None
                if map_func_kw is not None:
                    # check if one of the supplied kwargs in the list is None
                    # if None - change it to empty {}
                    map_func_kw = [{} if kw is None else kw for kw in map_func_kw]
                    self.data = [
                        func(_d, **kwargs)
                        for func, kwargs in zip(map_func, map_func_kw)
                    ]
                else:
                    self.data = [func(_d) for func in map_func]

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
    vmin=None,
    vmax=None,
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
    vmin : float or list of floats, optional
        Minimum data value that colormap covers, by default None
    vmax : float or list of floats, optional
        Maximum data value that colormap covers, by default None
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
        vmin=vmin,
        vmax=vmax,
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


class ParamGrid(object):
    """This class allows exploration of different parameters of
    a function across the rows and columns of the grid. Additional function
    parameters that are not to be varied can also be passed.

    Generates a grid of images with the specific function applied to all
    the images.

    Parameters
    ----------
    data :
        Image data (array-like). Supported array shapes are all
        `matplotlib.pyplot.imshow` array shapes
    map_func : callable or str
        Function to be applied/mapped to data. Can be any callable that accepts data
        as the the first input parameter.
        If using a `str`, must one of the implemented filter functions in `seaborn_image`.
        You can check implemented filters using `seaborn_image.implemented_filters()`.
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
    vmin : float or list of floats, optional
        Minimum data value that colormap covers, by default None
    vmax : float or list of floats, optional
        Maximum data value that colormap covers, by default None
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
        A `seabron_image.ParamGrid` object

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
        >>> g = isns.ParamGrid(img, "median", col="size", size=[2,3,4,5])

    Or rows

    .. plot::
        :context: close-figs

        >>> g = isns.ParamGrid(img, "median", row="size", size=[2,3,4,5])

    Use `col_wrap` to control column display

    .. plot::
        :context: close-figs

        >>> g = isns.ParamGrid(img, "median", col="size", size=[2,3,4,5], col_wrap=3)

    Use `col` and `row` to display different parameters along the columns and rows

    .. plot::
        :context: close-figs

        >>> g = isns.ParamGrid(img,
        ...                     "percentile",
        ...                     row="percentile",
        ...                     col="size",
        ...                     percentile=[10,20,30],
        ...                     size=[20,25,30],)

    Specify additional keyword arguments for the filter

    .. plot::
        :context: close-figs

        >>> g = isns.ParamGrid(img, "median", col="size", size=[2,3,4,5], mode="reflect")

    General image controls such as changing colormap, scalebar, etc.

    .. plot::
        :context: close-figs

        >>> g = isns.ParamGrid(
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
        map_func,
        *,
        row=None,
        col=None,
        col_wrap=None,
        height=3,
        aspect=1,
        cmap=None,
        alpha=None,
        origin=None,
        vmin=None,
        vmax=None,
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

        if map_func is None:
            raise ValueError("'map_func' can not be None; must be a string or callable")

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
        self.map_func = map_func
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
        self.vmin = vmin
        self.vmax = vmax
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
            self.map_func,
            ax=ax,
            cmap=self.cmap,
            alpha=self.alpha,
            origin=self.origin,
            vmin=self.vmin,
            vmax=self.vmax,
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


class FilterGrid:
    """Deprecated - use `ParamGrid` instead."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "FilterGrid is depracted and will be removed in a future release."
            "Use ParamGrid instead with the same arguments.",
            UserWarning,
        )
        ParamGrid(*args, **kwargs)
