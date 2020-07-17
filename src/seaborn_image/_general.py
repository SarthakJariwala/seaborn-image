from ._core import _SetupImage


def imgplot(
    data,
    ax=None,
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
    title=None,
    title_fontdict=None,
):

    img_plotter = _SetupImage(
        data=data,
        ax=ax,
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
        title=title,
        fontdict=title_fontdict,
    )

    f, ax = img_plotter.plot()

    return f, ax, data
