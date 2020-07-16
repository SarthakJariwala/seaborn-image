from ._core import _SetupImage


def imgplot(
    data,
    ax=None,
    cmap=None,
    scalebar=False,
    dx=None,
    units=None,
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
        ax=ax,
        cmap=cmap,
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