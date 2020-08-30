import pytest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from skimage.color import rgb2gray
from skimage.data import astronaut

import seaborn_image as isns

matplotlib.use("AGG")  # use non-interactive backend for tests


data = np.random.random(2500).reshape((50, 50))


def test_axes_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, ax=np.array([1, 2]))


def test_cmap_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, cmap=["r", "b", "g"])


def test_describe_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, describe=["True"])


def test_robust_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, robust="True")


@pytest.mark.parametrize("percentile", [(2, 10, 88), (45, 40)])
def test_percentile(percentile):
    with pytest.raises(AssertionError):
        isns.imgplot(data, robust=True, percentile=percentile)


def test_cbar_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, cbar="True")


def test_orientation_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, orientation=1)


def test_cbar_label_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, cbar_label=["Title"])


def test_cbar_fontdict_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, cbar_fontdict=["fontsize", 20])


def test_showticks_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, showticks="True")


def test_despine_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, despine="True")


def test_title_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, title=["Title"])


def test_title_fontdict_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, title_fontdict=[{"fontsize": 20}])


@pytest.mark.parametrize("data", [data, astronaut()])
def test_imgplot_return(data):
    f, ax, cax = isns.imgplot(data)

    assert isinstance(f, Figure)
    assert isinstance(ax, Axes)
    if (
        data.ndim == 3
    ):  # if data dim is 3 it cbar will be set to False, and cax will be None
        assert cax is None
    else:
        assert isinstance(cax, Axes)

    plt.close("all")


@pytest.mark.parametrize("data", [data, astronaut()])
def test_imgplot_data_is_same_as_input(data):
    f, ax, cax = isns.imgplot(data)

    # check if data iput is what was plotted
    np.testing.assert_array_equal(ax.images[0].get_array().data, data)


def test_imgplot_gray_conversion_for_rgb():
    """Check if the plotted data is grayscale when input is RGB image
    and gray is True.
    """
    f, ax, cax = isns.imgplot(astronaut(), gray=True)

    np.testing.assert_array_equal(ax.images[0].get_array().data, rgb2gray(astronaut()))


@pytest.mark.parametrize("gray", [True, False])
@pytest.mark.parametrize("cmap", [None, "ice"])
@pytest.mark.parametrize("data", [data, astronaut()])
def test_gray_cmap_interplay(data, gray, cmap):
    f, ax, cax = isns.imgplot(data, cmap=cmap, gray=gray)
    plt.close("all")


@pytest.mark.parametrize("describe", [True, False])
def test_imgplot_w_describe(describe):
    f, ax, cax = isns.imgplot(data, describe=describe)
    plt.close("all")


@pytest.mark.parametrize("bins", ["random", 200.0, -400.13])
def test_imghist_bins_type(bins):
    with pytest.raises(TypeError):
        isns.imghist(data, bins=bins)


@pytest.mark.parametrize("bins", [-100, 0])
def test_imghist_bins_value(bins):
    with pytest.raises(ValueError):
        isns.imghist(data, bins=bins)


def test_imghist_orientation_value():
    with pytest.raises(ValueError):
        isns.imghist(data, orientation="right")


def test_imghist_return():
    f, axes, cax = isns.imghist(data)

    assert isinstance(f, Figure)
    assert isinstance(axes[0], Axes)
    assert isinstance(axes[1], Axes)
    assert isinstance(cax, Axes)

    plt.close("all")


def test_imghist_data_is_same_as_input():
    f, ax, cax = isns.imghist(data)

    # check if data iput is what was plotted
    np.testing.assert_array_equal(ax[0].images[0].get_array().data, data)


@pytest.mark.parametrize("cmap", [None, "acton"])
@pytest.mark.parametrize("bins", [None, 100])
@pytest.mark.parametrize("orientation", ["horizontal", "h", "vertical", "v"])
@pytest.mark.parametrize("showticks", [True, False])
@pytest.mark.parametrize("despine", [True, False])
def test_imghist_w_all_valid_inputs(
    cmap,
    bins,
    orientation,
    showticks,
    despine,
):
    f, axes, cax = isns.imghist(
        data,
        cmap=cmap,
        bins=bins,
        orientation=orientation,
        showticks=showticks,
        despine=despine,
    )

    plt.close("all")
