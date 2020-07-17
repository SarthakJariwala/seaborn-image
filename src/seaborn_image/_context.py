import matplotlib as mpl
import matplotlib.pyplot as plt

# TODO implement a set() fuction for all underlying set_*


def set_context(mode="paper", fontfamily="arial", fontweight="bold", rc=None):
    # plt.rc("axes.spines", left=False, right=False, top=False, bottom=False)
    if mode == "paper":
        plt.rc('axes', linewidth=1.5)
        plt.rc('axes', titlesize=15, titleweight=fontweight)
        plt.rc('axes', labelsize=15, labelweight=fontweight)
        font = {
            'family' : fontfamily,
            'weight' : fontweight,
            'size'   : 10}

    if mode == "notebook" or mode == "presentation":
        plt.rc('axes', linewidth=2.5)
        plt.rc('axes', titlesize=20, titleweight=fontweight)
        plt.rc('axes', labelsize=20, labelweight=fontweight)
        font = {
            'family' : fontfamily,
            'weight' : fontweight,
            'size'   : 15}

    if mode == "poster":
        plt.rc('axes', linewidth=3.5)
        plt.rc('axes', titlesize=25, titleweight=fontweight)
        plt.rc('axes', labelsize=25, labelweight=fontweight)
        font = {
            'family' : fontfamily,
            'weight' : fontweight,
            'size'   : 20}

    plt.rc('font', **font)

    if rc is not None:
        assert type(rc) == dict, "'rc' must be a dict"
        mpl.rcParams.update(rc)


def set_save_context(dpi=300):
    plt.rc('savefig', dpi=dpi, bbox="tight")


def reset_defaults():
    mpl.rcParams.update(mpl.rcParamsDefault)


def set_scalebar(rc=None):
    """Set scalebar properties such as color, scale_loc,
    height_fraction, length_fraction, box_alpha, etc

    Args:
        rc (dict, optional): dictionary of scalebar properties to be set.
        Defaults to None.

    Examples
    --------
    >>> isns.set_scalebar({"color":"red"})
    >>> isns.set_scalebar({"scale_loc":"bottom"})
    """
    mpl.rcParams.update({"scalebar.color": "white"})
    mpl.rcParams.update({"scalebar.height_fraction": 0.05})
    mpl.rcParams.update({"scalebar.length_fraction": 0.3})
    mpl.rcParams.update({"scalebar.scale_loc": "top"})
    mpl.rcParams.update({"scalebar.location": "lower right"})
    mpl.rcParams.update({"scalebar.box_alpha": 0})

    if rc is not None:
        assert type(rc) == dict, "'rc' must be a dict"
        for key, value in rc.items():
            mpl.rcParams.update({f"scalebar.{key}": value})
