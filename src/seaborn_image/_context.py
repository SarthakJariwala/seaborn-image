import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import register_cmap

from ._colormap import _CMAP_QUAL

__all__ = [
    "set_context",
    "set_save_context",
    "reset_defaults",
    "set_image",
    "set_scalebar",
]

# TODO implement a set() fuction for all underlying set_*


def set_context(mode="paper", fontfamily="sans-serif", fontweight="normal", rc=None):
    """
    Set context for images with mode, fontfamily and fontweight. Additional,
    rc params can also be passed as dict

    Parameters
    ----------
    mode : str, optional
        Plotting context mode. Depending on the context, axes width, fontsize, layout etc.
        are scaled. Options are 'paper', 'notebook', 'presentation', 'talk' and 'poster',
        by default "paper".
    fontfamily : str, optional
        Font-family to use, by default "sans-serif".
    fontweight : str, optional
        Font-weight to use. Options include 'normal' and 'bold', by default "bold".
    rc : dict, optional
        Additional `matplotlib.rcParams` to be passed to matplotlib, by default None.

    Examples
    --------
        >>> import seaborn_image as isns
        >>> isns.set_context(mode="poster", fontfamily="sans-serif")
        >>> isns.set_context(rc={"axes.edgecolor": "red"})

    """

    if mode == "paper":
        plt.rc("axes", linewidth=1.5)
        plt.rc("axes", titlesize=15, titleweight=fontweight)
        plt.rc("axes", labelsize=15, labelweight=fontweight)
        mpl.rcParams.update({"figure.constrained_layout.wspace": 0.2})
        font = {"family": fontfamily, "weight": fontweight, "size": 10}

    if mode == "notebook" or mode == "presentation" or mode == "talk":
        plt.rc("axes", linewidth=2.5)
        plt.rc("axes", titlesize=20, titleweight=fontweight)
        plt.rc("axes", labelsize=20, labelweight=fontweight)
        mpl.rcParams.update({"figure.constrained_layout.wspace": 0.3})
        font = {"family": fontfamily, "weight": fontweight, "size": 15}

    if mode == "poster":
        plt.rc("axes", linewidth=3.5)
        plt.rc("axes", titlesize=25, titleweight=fontweight)
        plt.rc("axes", labelsize=25, labelweight=fontweight)
        mpl.rcParams.update({"figure.constrained_layout.wspace": 0.4})
        font = {"family": fontfamily, "weight": fontweight, "size": 20}

    plt.rc("font", **font)

    if rc is not None:
        assert isinstance(rc, dict), "'rc' must be a dict"
        mpl.rcParams.update(rc)


def set_save_context(dpi=300):
    """
    Set dpi for saving figures to disk

    Parameters
    ----------
    dpi : int, optional
        Image dpi for saving, by default 300.

    Examples
    --------
        >>> import seaborn_image as isns
        >>> isns.set_save_context(dpi=200)
    """
    plt.rc("savefig", dpi=dpi, bbox="tight")


def reset_defaults():
    """
    Reset rcParams to matplotlib defaults

    Examples
    --------
        >>> import seaborn_image as isns
        >>> isns.reset_defaults()
    """
    mpl.rcParams.update(mpl.rcParamsDefault)


def set_image(cmap="deep", origin="lower", interpolation="nearest", despine=False):
    """
    Set deaults for plotting images

    Parameters
    ----------
    cmap : str, optional
        Colormap to use accross images, by default to "deep".
    origin : str, optional
        Image origin - same as in `matplotlib.pyplot.imshow`, by default "lower".
    interpolation : str, optional
        Image interpolation - same as in `matplotlib.pyplot.imshow`, by default "nearest".
    despine : bool, optional
        Despine image and colorbar axes, by default False.

    Examples
    --------
        >>> import seaborn_image as isns
        >>> isns.set_image(cmap="inferno", interpolation="bicubic")
        >>> isns.set_image(despine=False)

    """

    if cmap in _CMAP_QUAL.keys():  # doesn't work currently
        cmap_mpl = _CMAP_QUAL.get(cmap).mpl_colormap
        register_cmap(name=cmap, cmap=cmap_mpl)

    # change the axes spines
    # "not" is required because of the despine parameter name
    # if depine is True ---> you don't want the axes spines
    # and therfore, axes.spines.bottom == False (or not True)
    mpl.rcParams.update(
        {
            "axes.spines.bottom": not despine,
            "axes.spines.left": not despine,
            "axes.spines.right": not despine,
            "axes.spines.top": not despine,
        }
    )

    plt.rc("image", cmap=cmap, origin=origin, interpolation=interpolation)


def set_scalebar(
    color="white",
    location="lower right",
    height_fraction=0.025,
    length_fraction=0.3,
    scale_loc="top",
    box_alpha=0,
    rc=None,
):
    """Set scalebar properties such as color, scale_loc,
    height_fraction, length_fraction, box_alpha, etc.
    To pass more properties that are not specified as key word
    argument, use the `rc` parameter. Refer to https://github.com/ppinard/matplotlib-scalebar
    for more information on additional parameters.

    Parameters
    ----------
        color : str, optional
            Color of the scalebar, by default "white".
        location : str, optional
            Scalebar location on the image (same as `matplotlib` legend).
            by default "lower right".
        height_fraction : float, optional
            By default 0.025.
        length_fraction : float, optional
            By default 0.3
        scale_loc :str, optional
            Location of the scale number and units with respect to
            the bar, by default "top".
        box_alpha : float, optional
            Transparency of the box that contains
            the scalebar artist, by default 0.
        rc : dict, optional
            Dictionary of scalebar properties to be set, by default None.

    Examples
    --------
        >>> import seaborn_image as isns
        >>> isns.set_scalebar(color = "red")
        >>> isns.set_scalebar(scale_loc = "bottom")

    """
    mpl.rcParams.update({"scalebar.color": color})
    mpl.rcParams.update({"scalebar.height_fraction": height_fraction})
    mpl.rcParams.update({"scalebar.length_fraction": length_fraction})
    mpl.rcParams.update({"scalebar.scale_loc": scale_loc})
    mpl.rcParams.update({"scalebar.location": location})
    mpl.rcParams.update({"scalebar.box_alpha": box_alpha})

    if rc is not None:
        assert isinstance(rc, dict), "'rc' must be a dict"
        for key, value in rc.items():
            mpl.rcParams.update({f"scalebar.{key}": value})
