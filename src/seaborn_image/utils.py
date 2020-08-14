import matplotlib.pyplot as plt
from matplotlib import ticker

__all__ = ["scientific_ticks", "despine"]


def scientific_ticks(ax):
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    # formatter.set_powerlimits((-1,1))
    ax.yaxis.set_major_formatter(formatter)


def despine(fig=None, ax=None, which="all"):
    if fig is None and ax is None:
        axes = plt.gcf().axes
    elif fig is not None:
        axes = fig.axes
    elif ax is not None:
        axes = [ax]

    _all = ["top", "bottom", "right", "left"]

    if isinstance(which, str):
        if which == "all":
            _to_despine = _all
        elif which in _all:
            _to_despine = [which]
        else:
            raise ValueError(
                f"Specify spine that is to be despined from the following : {_all.append('all')}"
            )

    elif isinstance(which, list):
        _to_despine = []
        for _which in which:
            if _which in _all:
                _to_despine.append(_which)

    else:
        raise TypeError(
            f"{which} must be of the type 'str' or 'list'. Options are : {_all.append('all')}"
        )

    for ax in axes:
        for spine in _to_despine:
            ax.spines[spine].set_visible(False)
