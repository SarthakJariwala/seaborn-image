import pytest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from skimage.color import rgb2gray
from skimage.data import astronaut
from skimage.exposure import adjust_gamma

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


def test_map_func_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, map_func="gaussian")


@pytest.mark.parametrize("perc", [(2, 10, 88), (45, 40)])
def test_percentile(perc):
    with pytest.raises(AssertionError):
        isns.imgplot(data, robust=True, perc=perc)


def test_cbar_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, cbar="True")


def test_orientation_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, orientation=1)


def test_cbar_label_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, cbar_label=["Title"])


def test_cbar_log_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, cbar_log=matplotlib.colors.LogNorm())


def test_showticks_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, showticks="True")


def test_despine_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, despine="True")


@pytest.mark.parametrize("data", [data, astronaut()])
def test_imgplot_return(data):
    ax = isns.imgplot(data)

    f = plt.gcf()

    assert isinstance(ax, Axes)
    if (
        data.ndim == 3
    ):  # if data dim is 3 it cbar will be set to False, and cax will be None
        pass  # no colorbar axes in this case
    else:
        assert isinstance(f.axes[1], Axes)

    plt.close("all")


@pytest.mark.parametrize("data", [data, astronaut()])
def test_imgplot_data_is_same_as_input(data):
    ax = isns.imgplot(data)

    # check if data iput is what was plotted
    np.testing.assert_array_equal(ax.images[0].get_array().data, data)


def test_imgplot_gray_conversion_for_rgb():
    """Check if the plotted data is grayscale when input is RGB image
    and gray is True.
    """
    ax = isns.imgplot(astronaut(), gray=True)

    np.testing.assert_array_equal(ax.images[0].get_array().data, rgb2gray(astronaut()))


@pytest.mark.parametrize("gray", [True, False])
@pytest.mark.parametrize("cmap", [None, "ice"])
@pytest.mark.parametrize("data", [data, astronaut()])
def test_gray_cmap_interplay(data, gray, cmap):
    _ = isns.imgplot(data, cmap=cmap, gray=gray)
    plt.close("all")


@pytest.mark.parametrize("describe", [True, False])
def test_imgplot_w_describe(describe):
    _ = isns.imgplot(data, describe=describe)
    plt.close("all")


def test_map_func():
    cells = isns.load_image("cells")[:, :, 32]
    ax = isns.imgplot(cells, map_func=adjust_gamma, gamma=0.5)

    np.testing.assert_array_equal(
        ax.images[0].get_array().data, adjust_gamma(cells, gamma=0.5)
    )

    # imshow with kwargs to test if they are passed on to imgplot
    ax = isns.imshow(cells, map_func=adjust_gamma, gamma=0.5)

    np.testing.assert_array_equal(
        ax.images[0].get_array().data, adjust_gamma(cells, gamma=0.5)
    )


def test_cbar_log_and_norm():
    # special case of log-norm
    _ = isns.imgplot(data, cbar_log=True)
    plt.close()

    # when only norm is specified
    _ = isns.imgplot(data, norm=matplotlib.colors.LogNorm())
    plt.close()

    # norm takes preference
    _ = isns.imgplot(data, norm=matplotlib.colors.LogNorm(), cbar_log=True)
    plt.close()


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


def test_imghist_3D_data():
    with pytest.raises(ValueError):
        isns.imghist(astronaut())


def test_imghist_return():
    f = isns.imghist(data)

    assert isinstance(f, Figure)
    assert isinstance(f.axes[0], Axes)
    assert isinstance(f.axes[1], Axes)
    assert isinstance(f.axes[2], Axes)

    plt.close("all")


def test_imghist_figsize():
    # check default
    f = isns.imghist(data)
    np.testing.assert_array_equal(f.get_size_inches(), (5 * 1.75, 5))
    plt.close()

    # check user specified
    f = isns.imghist(data, height=6, aspect=1.5)
    np.testing.assert_array_equal(f.get_size_inches(), (6 * 1.5, 6))
    plt.close()


def test_imghist_data_is_same_as_input():
    f = isns.imghist(data)

    # check if data iput is what was plotted
    np.testing.assert_array_equal(f.axes[0].images[0].get_array().data, data)


def test_imghist_robust_hist_cmap():
    """Check if the min/max patch color in histogram matches
    the min/max colors in the colorbar after robust"""

    polymer = isns.load_image("polymer")

    f = isns.imghist(polymer, robust=True, perc=(0.5, 50))

    _min = np.nanpercentile(polymer, 0.5)
    _max = np.nanpercentile(polymer, 50)

    # check the max patch facecolor and check it against the image cmap max
    np.testing.assert_array_equal(
        f.axes[0].images[0].cmap(_max), f.axes[-1].patches[-1].get_facecolor()
    )

    # check the min patch facecolor and check it against the image cmap min
    np.testing.assert_array_equal(
        f.axes[0].images[0].cmap(_min), f.axes[-1].patches[0].get_facecolor()
    )

    plt.close()


@pytest.mark.parametrize("cmap", [None, "acton"])
@pytest.mark.parametrize("bins", [None, 100])
@pytest.mark.parametrize("orientation", ["horizontal", "h", "vertical", "v"])
@pytest.mark.parametrize("showticks", [True, False])
@pytest.mark.parametrize("despine", [True, False])
@pytest.mark.parametrize("cbar_log", [True, False])
def test_imghist_w_all_valid_inputs(
    cmap,
    bins,
    orientation,
    showticks,
    despine,
    cbar_log,
):
    _ = isns.imghist(
        data,
        cmap=cmap,
        bins=bins,
        orientation=orientation,
        showticks=showticks,
        despine=despine,
        cbar_log=cbar_log,
    )

    plt.close("all")
