import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np

from ._filters import filterplot
from ._general import imgplot

# class ImageGrid(object):
#     def __init__(
#         self,
#         data=None,
#         row=None,
#         col=None,
#         col_wrap=None,
#         mosaic=None,  # available starting matplotlib 3.3
#         height=5,
#         aspect=1,
#     ):

#         # Set up the lists of names for the row and column facet variables
#         if row is None:
#             row_names = []
#         else:
#             if isinstance(row, int):
#                 row_names = [row]  # TODO add a hide_row_names option
#             if isinstance(row, str):
#                 row_names = [row]
#             elif isinstance(row, list):
#                 row_names = row

#         if col is None:
#             col_names = []
#         else:
#             if isinstance(col, int):
#                 col_names = [col]  # TODO add a hide_row_names option
#             if isinstance(col, str):
#                 col_names = [col]
#             elif isinstance(col, list):
#                 col_names = col

#         # Compute the grid shape like FacetGrid
#         ncol = 1 if col is None else len(col_names)
#         nrow = 1 if row is None else len(row_names)

#         if col_wrap is not None:
#             if row is not None:
#                 err = "Cannot use `row` and `col_wrap` together."
#                 raise ValueError(err)
#             ncol = col_wrap
#             nrow = int(np.ceil(len(col_names) / col_wrap))

#         # Calculate the base figure size
#         figsize = (ncol * height * aspect, nrow * height)

#         fig = plt.figure(figsize=figsize)

#         # Ignore everything and plot mosaic if specified
#         if mosaic is not None:
#             axes = fig.subplot_mosaic(mosaic)
#             if row or col or col_wrap:
#                 warnings.warn(
#                     "Ignoring `row`, `col` and `col_wrap` when `mosiac` is specified"
#                 )
#         else:
#             axes = fig.subplots(nrow, ncol)

#         # if self.data is not None:
#         #     if isinstance(self.data, list):
#         #         _total = len(self.data)
#         #         self.col_wrap = col_wrap

#         # Public API
#         self.data = data
#         self.fig = fig
#         self.axes = axes
#         self.row = row
#         self.col = col
#         self.height = height
#         self.aspect = aspect

#         return

#     def map(self, func, ax, *args, **kwargs):

#         # If color was a keyword argument, grab it here
#         kw_color = kwargs.pop("color", None)

#         if hasattr(func, "__module__"):
#             func_module = str(func.__module__)
#         else:
#             func_module = ""

#         # Some matplotlib functions don't handle pandas objects correctly
#         # if func_module.startswith("matplotlib"):
#         #     plot_args = [v.values for v in plot_args]

#         func(*args, **kwargs)
#         # Draw the plot
#         # self._facet_plot(func, ax, *args, **kwargs)

#         return self

#     def _facet_plot(self, func, ax, plot_args, plot_kwargs):

#         # Draw the plot

#         func(*plot_args, **plot_kwargs)

#         return self


class FilterGrid(object):
    def __init__(
        self,
        data=None,
        filter_name="gaussian",
        *,
        row=None,
        col=None,
        height=3,
        aspect=1,
        cmap=None,
        dx=None,
        units=None,
        dimension=None,
        describe=False,
        cbar=True,
        cbar_label=None,
        cbar_fontdict=None,
        cbar_ticks=None,
        showticks=False,
        **kwargs,
    ):  # TODO add despine option

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

        # Public API
        self.data = data
        self.filter = filter_name
        self.fig = fig
        self.axes = axes
        self.row = row
        self.col = col
        self.param_product = product_params
        self.height = height
        self.aspect = aspect

        self.cmap = cmap
        self.dx = dx
        self.units = units
        self.dimension = dimension
        self.describe = describe
        self.cbar = cbar
        self.cbar_label = cbar_label
        self.cbar_fontdict = cbar_fontdict
        self.cbar_ticks = cbar_ticks
        self.showticks = showticks

        self.map()
        self._finalize_grid()

        return

    def map(self):

        if self.row is None and self.col is None:
            imgplot(self.data, ax=self.axes)

        for i in range(len(self.param_product)):
            ax = self.axes.flat[i]
            p = self.param_product[i]

            if self.row is None:
                func_kwargs = {self.col: p[0]}
                self._plot(ax=ax, **func_kwargs)
                ax.set_title(f"{self.col} : {p[0]}")

            if self.col is None:
                func_kwargs = {self.row: p[0]}
                # print(func_kwargs)
                self._plot(ax=ax, **func_kwargs)
                ax.set_title(f"{self.row} : {p[0]}")

            if self.row and self.col:
                func_kwargs = {self.row: p[0], self.col: p[1]}
                # print(func_kwargs)
                self._plot(ax=ax, **func_kwargs)
                ax.set_title(f"{self.row} : {p[0]}, {self.col} : {p[1]}")

        return

    def _plot(self, ax, **func_kwargs):

        filterplot(
            self.data,
            self.filter,
            ax=ax,
            cmap=self.cmap,
            dx=self.dx,
            units=self.units,
            dimension=self.dimension,
            cbar=self.cbar,
            cbar_label=self.cbar_label,
            cbar_fontdict=self.cbar_fontdict,
            cbar_ticks=self.cbar_ticks,
            showticks=self.showticks,
            **func_kwargs,
        )
        return

    def _finalize_grid(self):
        self.fig.tight_layout()
