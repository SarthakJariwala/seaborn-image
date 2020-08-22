import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np

from ._filters import filterplot
from ._general import imgplot
from .utils import despine

# TODO add despine option
# TODO provide common cbar option


class FilterGrid(object):

    def __init__(
        self,
        data=None,
        filter_name="gaussian",
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
        orientation='v',
        cbar_label=None,
        cbar_fontdict=None,
        cbar_ticks=None,
        showticks=False,
        despine=True,
        **kwargs,
    ):

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
        axes = fig.subplots(
            nrow, ncol, squeeze=False
        )  # TODO allow gridspec_kws and subplot_kws

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
        self.col_wrap = col_wrap
        self.param_product = product_params
        self.height = height
        self.aspect = aspect

        self.cmap = cmap
        self.dx = dx
        self.units = units
        self.dimension = dimension
        self.cbar = cbar
        self.orientation = orientation
        self.cbar_label = cbar_label
        self.cbar_fontdict = cbar_fontdict
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

        if self.row is None and self.col is None:
            imgplot(self.data, ax=self.axes)

        for i in range(len(self.param_product)):
            ax = self.axes.flat[i]
            p = self.param_product[i]

            # plot only col vars
            if self.row is None:
                func_kwargs = {self.col: p[0]}
                self._plot(ax=ax, **func_kwargs)

                ax.set_title(f"{self.col} : {p[0]}")

            # plot only row vars
            if self.col is None:
                func_kwargs = {self.row: p[0]}
                self._plot(ax=ax, **func_kwargs)

                ax.set_title(f"{self.row} : {p[0]}")

            # when both row and col vars are specified
            if self.row and self.col:
                func_kwargs = {self.row: p[0], self.col: p[1]}
                self._plot(ax=ax, **func_kwargs)

                # set row labels only to the outermost column
                if not i % self._nrow:
                    ax.set_ylabel(f"{self.row} : {p[0]}")

                # set column labels only to the top row
                if i < self._ncol:
                    ax.set_title(f"{self.col} : {p[1]}")

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
            orientation=self.orientation,
            cbar_label=self.cbar_label,
            cbar_fontdict=self.cbar_fontdict,
            cbar_ticks=self.cbar_ticks,
            showticks=self.showticks,
            despine=self.despine,
            **func_kwargs,
        )
        return

    def _cleanup_extra_axes(self):
        """Clean extra axes that are generated if col_wrap is specified.
        """
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

                    despine(ax=rem_ax[i])  # remove axes spines for the extra generated axes

    def _finalize_grid(self):
        """Finalize grid with tight layout.
        """
        self.fig.tight_layout()
