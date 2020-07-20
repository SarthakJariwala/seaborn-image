import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import axes_size, make_axes_locatable

from ._colormap import _CMAP_QUAL


def _check_dict(dictionary):
    if not isinstance(dictionary, dict):
        raise ValueError(f"{dictionary} must be a dictionary")


class _SetupImage(object):
    def __init__(
        self,
        data,
        ax=None,
        cmap=None,
        vmin=None,
        vmax=None,
        title=None,
        fontdict=None,
        dx=None,
        units=None,
        cbar=None,
        cbar_fontdict=None,
        cbar_label=None,
        cbar_ticks=None,
        showticks=False,
    ):

        self.data = data
        self.ax = ax
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.title = title
        self.fontdict = fontdict
        self.dx = dx
        self.units = units
        self.cbar = cbar
        self.cbar_fontdict = cbar_fontdict
        self.cbar_label = cbar_label
        self.cbar_ticks = cbar_ticks
        self.showticks = showticks

    def _setup_figure(self):
        """Wrapper to setup image with the desired parameters
        """
        if self.ax is None:
            f, ax = plt.subplots()
        else:
            f = plt.gcf()
            ax = self.ax

        if self.fontdict is not None:
            _check_dict(self.fontdict)

        if self.title is not None:
            ax.set_title(self.title, fontdict=self.fontdict)

        return f, ax

    def _setup_scalebar(self, ax):
        """Setup scalebar for the image
        """

        if self.dx:
            dx = self.dx
            if self.units:
                units = self.units
            else:
                raise AttributeError(
                    "'units' must be specified when 'dx' (scalebar) is used"
                )

        scalebar = ScaleBar(dx=dx, units=units)

        ax.add_artist(scalebar)

    def plot(self):
        f, ax = self._setup_figure()

        if self.cmap is None:  # TODO move default to _context
            self.cmap = _CMAP_QUAL.get("deep").mpl_colormap
        elif self.cmap in _CMAP_QUAL.keys():
            self.cmap = _CMAP_QUAL.get(self.cmap).mpl_colormap

        _map = ax.imshow(self.data, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)

        if self.dx:
            self._setup_scalebar(ax)

        if self.cbar:
            divider = make_axes_locatable(ax)
            width = axes_size.AxesY(ax, aspect=1.0 / 20)
            pad = axes_size.Fraction(0.5, width)
            cax = divider.append_axes("right", size=width, pad=pad)

            cb = f.colorbar(_map, cax=cax)
            if self.vmin is not None:
                _min = self.vmin
            else:
                _min = np.min(self.data)
            if self.vmax is not None:
                _max = self.vmax
            else:
                _max = np.max(self.data)

            if self.cbar_ticks is None:
                cb.set_ticks(
                    [_min, (_min + _max) / 2, _max]
                )  # min, middle, max for colorbar ticks
            else:
                cb.set_ticks(self.cbar_ticks)

            if self.cbar_fontdict is not None:
                _check_dict(self.cbar_fontdict)

            if self.cbar_label is not None:
                cb.set_label(self.cbar_label, fontdict=self.cbar_fontdict)

        if not self.showticks:
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)

        f.tight_layout()

        return f, ax
