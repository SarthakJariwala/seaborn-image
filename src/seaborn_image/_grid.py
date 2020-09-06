import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np

from ._filters import filterplot
from ._general import imgplot
from .utils import despine

__all__ = ["FilterGrid"]

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
    height : int, optional
        Size of the individual images. Defaults to 3.
    aspect : int, optional
        Aspect ratio of individual images. Defaults to 1.
    cmap : str or `matplotlib.colors.Colormap`, optional
        Image colormap. Defaults to None.
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
            - 'h' or 'horizontal' for a horizontal colorbar and histogram to the bottom of the image.
            - 'v' or 'vertical' for a vertical colorbar and histogram to the right of the image.
        Defaults to 'v'.
    cbar_label : str, optional
        Colorbar label. Defaults to None.
    cbar_ticks : list, optional
        List of colorbar ticks. If None, min and max of
        the data are used. If `vmin` and `vmax` are specified, `vmin` and `vmax` values
        are used for colorbar ticks. Defaults to None.
    showticks : bool, optional
        Show image x-y axis ticks. Defaults to False.
    despine : bool, optional
        Remove axes spines from image axes as well as colorbar axes.
        Defaults to True.
    **kwargs : Additional parameters as keyword arguments to be passed to the underlying filter specified.

     Returns
    -------
        A `seabron_image.FilterGrid` object

    Raises
    ------
    TypeError
        If `row` is not a str
    ValueError
        If `row` is specified without passing the parameter as a keword argument
    TypeError
        If `col` is not a str
    ValueError
        If `col` is specified without passing the parameter as a keword argument
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
        dx=None,
        units=None,
        dimension=None,
        cbar=True,
        orientation="v",
        cbar_label=None,
        cbar_ticks=None,
        showticks=False,
        despine=True,
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
        self.dx = dx
        self.units = units
        self.dimension = dimension
        self.cbar = cbar
        self.orientation = orientation
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

        Args:
            ax (`matplotlib.axes.Axes`): Axis to plot filtered image
        """

        filterplot(
            self.data,
            self.filt,
            ax=ax,
            cmap=self.cmap,
            dx=self.dx,
            units=self.units,
            dimension=self.dimension,
            cbar=self.cbar,
            orientation=self.orientation,
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
