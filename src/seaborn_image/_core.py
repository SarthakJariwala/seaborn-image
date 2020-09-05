import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import axes_size, make_axes_locatable

from ._colormap import _CMAP_QUAL
from .utils import despine, scientific_ticks

# dimensions for scalebar
_DIMENSIONS = {
    "si": "si-length",
    "si-reciprocal": "si-length-reciprocal",
    "imperial": "imperial-length",
    "angle": "angle",
    "pixel": "pixel-length",
}


def _check_dict(dictionary):
    if not isinstance(dictionary, dict):
        raise TypeError(f"{dictionary} must be a dictionary")


class _SetupImage(object):
    def __init__(
        self,
        data,
        ax=None,
        cmap=None,
        vmin=None,
        vmax=None,
        robust=False,
        perc=None,
        dx=None,
        units=None,
        dimension=None,
        cbar=None,
        orientation="v",
        cbar_label=None,
        cbar_ticks=None,
        showticks=False,
        despine=True,
    ):

        self.data = data
        self.ax = ax
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.robust = robust
        self.perc = perc
        self.dx = dx
        self.units = units
        self.dimension = dimension
        self.cbar = cbar
        self.orientation = orientation
        self.cbar_label = cbar_label
        self.cbar_ticks = cbar_ticks
        self.showticks = showticks
        self.despine = despine

    def _setup_figure(self):
        """Wrapper to setup image with the desired parameters"""
        if self.ax is None:
            f, ax = plt.subplots()
        else:
            f = plt.gcf()
            ax = self.ax

        return f, ax

    def _setup_scalebar(self, ax):
        """Setup scalebar for the image"""

        if self.dx:
            dx = self.dx
            if self.units:
                units = self.units
            else:
                raise AttributeError(
                    "'units' must be specified when 'dx' (scalebar) is used"
                )

        if self.dimension is None:
            _dimension = _DIMENSIONS.get("si")
        elif self.dimension in _DIMENSIONS.keys():
            _dimension = _DIMENSIONS.get(self.dimension)
        else:
            raise ValueError(
                f"Unsupported dimension. Supported dimensions are : {_DIMENSIONS.keys()}"
            )

        scalebar = ScaleBar(dx=dx, units=units, dimension=_dimension)

        ax.add_artist(scalebar)

    def plot(self):
        f, ax = self._setup_figure()

        if self.cmap in _CMAP_QUAL.keys():
            self.cmap = _CMAP_QUAL.get(self.cmap).mpl_colormap

        if self.robust:
            min_robust = False
            max_robust = False
            if self.vmin is None:
                min_robust = (
                    True  # remember that vmin was None and now set to new value
                )
                self.vmin = np.nanpercentile(self.data, self.perc[0])
            if self.vmax is None:
                max_robust = (
                    True  # remember that vmax was None and now set to new value
                )
                self.vmax = np.nanpercentile(self.data, self.perc[1])

        # TODO move everything other than data to kwargs
        _map = ax.imshow(self.data, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)

        if self.dx:
            self._setup_scalebar(ax)

        if self.cbar:
            divider = make_axes_locatable(ax)

            if self.orientation in ["vertical", "v"]:
                self.orientation = "vertical"  # plt.colorbar doesn't take 'v'
                width = axes_size.AxesY(ax, aspect=1.0 / 20)
                pad = axes_size.Fraction(0.5, width)
                cax = divider.append_axes("right", size=width, pad=pad)

            elif self.orientation in ["horizontal", "h"]:
                self.orientation = "horizontal"  # plt.colorbar doesn't take 'h'
                width = axes_size.AxesX(ax, aspect=1.0 / 20)
                pad = axes_size.Fraction(0.5, width)
                cax = divider.append_axes("bottom", size=width, pad=pad)

            else:
                raise ValueError(
                    "'orientation' must be either : 'horizontal' or 'h' / 'vertical' or 'v'"
                )

            # extend specific colorbar regions if robust is True
            if self.robust:
                if min_robust and max_robust:
                    cb = f.colorbar(
                        _map, cax=cax, orientation=self.orientation, extend="both"
                    )
                elif min_robust:
                    cb = f.colorbar(
                        _map, cax=cax, orientation=self.orientation, extend="min"
                    )
                elif max_robust:
                    cb = f.colorbar(
                        _map, cax=cax, orientation=self.orientation, extend="max"
                    )
            else:
                cb = f.colorbar(_map, cax=cax, orientation=self.orientation)

            if self.despine:
                cb.outline.set_visible(False)  # remove colorbar outline border

            # display only 3 tick marks on colorbar
            if self.orientation in ["vertical"]:
                cax.yaxis.set_major_locator(plt.MaxNLocator(3))
                cax.tick_params(axis="y", width=0, length=0)  # remove ytick lines

            if self.orientation in ["horizontal"]:
                cax.xaxis.set_major_locator(plt.MaxNLocator(3))
                cax.tick_params(axis="x", width=0, length=0)  # remove xtick lines

            # TODO add option for scientific ticks as part of inbuilt functions
            # scientific_ticks(ax=cax)

            if self.cbar_ticks is not None:
                cb.set_ticks(self.cbar_ticks)

            if self.cbar_label is not None:
                cb.set_label(self.cbar_label)

        else:
            cax = None

        if not self.showticks:
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_ylabel("")
            ax.set_xlabel("")
            # ax.get_yaxis().set_visible(False)
            # ax.get_xaxis().set_visible(False)

        if self.despine:
            despine(ax=ax, which="all")

        f.tight_layout()

        return f, ax, cax
