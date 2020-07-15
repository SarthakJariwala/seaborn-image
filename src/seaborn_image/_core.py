import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar


def _check_dict(dictionary):
    if not isinstance(dictionary, dict):
        raise ValueError(f"{dictionary} must be a dictionary")


class _SetupImage(object):
    def __init__(
        self,
        data,
        ax=None,
        title=None,
        fontdict=None,
        scalebar=None,
        dx=None,
        units=None,
        scalebar_params=None,
        cbar=None,
        cbar_fontdict=None,
        cbar_label=None,
        showticks=False,
    ):
        # self.new_fig = True
        # self.figsize = None
        self.data = data
        self.ax = ax
        self.title = title
        self.fontdict = fontdict
        self.scalebar = scalebar
        self.dx = dx
        self.units = units
        self.scalebar_params = scalebar_params
        self.cbar = cbar
        self.cbar_fontdict = cbar_fontdict
        self.cbar_label = cbar_label
        self.showticks = showticks

    def _setup_figure(self):
        """Wrapper to setup image with the desired parameters
        """
        if self.ax is None:
            f = plt.gcf()
            ax = plt.gca()
        else:
            f = plt.gcf()  # need to test
            ax = self.ax

        if self.fontdict is None:
            self.fontdict = {}

        _check_dict(self.fontdict)

        self.fontdict.setdefault("fontsize", 30)
        self.fontdict.setdefault("fontweight", "bold")

        if self.title is not None:
            ax.set_title(self.title, fontdict=self.fontdict)

        return f, ax

    def _setup_scalebar(self):
        """Setup scalebar for the image
        """

        if self.scalebar is not None:
            if self.dx is not None and self.units is not None:
                dx = self.dx
                units = self.units
            else:
                raise AttributeError(
                    "'dx' and 'units' must be assigned when scalebar is True"
                )

        if self.scalebar_params is None:
            self.scalebar_params = {}

        self.scalebar_params.setdefault("color", "white")
        self.scalebar_params.setdefault("height_fraction", 0.05)
        self.scalebar_params.setdefault("length_fraction", 0.3)
        self.scalebar_params.setdefault("scale_loc", "top")
        self.scalebar_params.setdefault("location", "lower right")
        self.scalebar_params.setdefault("box_alpha", 0)

        _check_dict(self.scalebar_params)

        color = self.scalebar_params.get("color")
        height_fraction = self.scalebar_params.get("height_fraction")
        length_fraction = self.scalebar_params.get("length_fraction")
        scale_loc = self.scalebar_params.get("scale_loc")
        location = self.scalebar_params.get("location")
        box_alpha = self.scalebar_params.get("box_alpha")

        scalebar = ScaleBar(
            dx=dx,
            units=units,
            color=color,
            height_fraction=height_fraction,
            length_fraction=length_fraction,
            scale_loc=scale_loc,
            location=location,
            box_alpha=box_alpha,
            font_properties=dict(size="x-large", weight="bold"),
        )

        plt.gca().add_artist(scalebar)

    def plot(self):
        f, ax = self._setup_figure()

        _map = ax.imshow(self.data)

        if self.scalebar:
            self._setup_scalebar()

        if self.cbar:

            cb = f.colorbar(_map, ax=ax)

            if self.cbar_fontdict is None:
                self.cbar_fontdict = {}

            _check_dict(self.cbar_fontdict)

            self.cbar_fontdict.setdefault("fontsize", 25)
            self.cbar_fontdict.setdefault("fontweight", "bold")

            if self.cbar_label is not None:
                cb.set_label(self.cbar_label, fontdict=self.cbar_fontdict)

        if not self.showticks:
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.gca().axes.get_xaxis().set_visible(False)


def plot_image(
    data,
    scalebar=True,
    dx=0.4,
    units="um",
    scalebar_params=None,
    cbar=True,
    cbar_label=None,
    cbar_fontdict=None,
    showticks=False,
    title=None,
    title_fontdict=None,
):

    img_plotter = _SetupImage(
        data=data,
        scalebar=scalebar,
        dx=dx,
        units=units,
        scalebar_params=scalebar_params,
        cbar=cbar,
        cbar_label=cbar_label,
        cbar_fontdict=cbar_fontdict,
        showticks=showticks,
        title=title,
        fontdict=title_fontdict,
    )
    img_plotter.plot()
