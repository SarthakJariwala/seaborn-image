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


def set_context(mode="talk", fontfamily="arial", fontweight="bold", rc=None):
    """
    Set context for images with mode, fontfamily and fontweight. Additional,
    rc params can also be passed as dict

    Args:
        mode (str, optional): context mode. Options are 'paper', 'notebook',
            'presentation', 'talk' and 'poster'. Defaults to "talk".
        fontfamily (str, optional): font-family to use. Defaults to "arial".
        fontweight (str, optional): font-weight to use. Options include 'normal'
            and 'bold'. Defaults to "bold".
        rc (dict, optional): additional rc params to be passed to matplotlib.
            Defaults to None.

    Example:
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

    Args:
        dpi (int, optional): image dpi. Defaults to 300.

    Example:
        >>> import seaborn_image as isns
        >>> isns.set_save_context(dpi=200)

    """
    plt.rc("savefig", dpi=dpi, bbox="tight")


def reset_defaults():
    """
    Reset rcParams to matplotlib defaults

    Example:
        >>> import seaborn_image as isns
        >>> isns.reset_deafults()

    """
    mpl.rcParams.update(mpl.rcParamsDefault)


def set_image(cmap="ice", origin="lower", interpolation="nearest"):
    """
    Set deaults for plotting images

    Args:
        cmap (str, optional): Colormap to use accross images.
            Defaults to "viridis".
        origin (str, optional): image origin - same as in matplotlib imshow.
            Defaults to "lower".
        interpolation (str, optional): image interpolation - same as in matplotlib imshow.
            Defaults to "nearest".

    Example:
        >>> import seaborn_image as isns
        >>> isns.set_image(cmap="inferno", interpolation="bicubic")

    """

    if cmap in _CMAP_QUAL.keys():  # doesn't work currently
        cmap_mpl = _CMAP_QUAL.get(cmap).mpl_colormap
        register_cmap(name=cmap, cmap=cmap_mpl)

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

    Args:
        color(str, optional): color of the scalebar. Defaults to "white".
        location (str, optional): scalebar location on the image (same as `matplotlib` legend).
            Defaults to "lower right".
        height_fraction (float, optional): Defaults to 0.025.
        length_fraction (float, optional): Defaults to 0.3
        scale_loc (str, optional): location of the scale number and units with respect to the bar.
            Defaults to "top".
        box_alpha (float, optional): transparency of the box that contains the scalebar artist.
            Defaults to 0.
        rc (dict, optional): dictionary of scalebar properties to be set.
            Defaults to None.

    Example:
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
