import pytest

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import seaborn_image as isns

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


def test_cbar_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, cbar="True")


def test_cbar_label_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, cbar_label=["Title"])


def test_cbar_fontdict_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, cbar_fontdict=["fontsize", 20])


def test_showticks_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, showticks="True")


def test_title_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, title=["Title"])


def test_title_fontdict_type():
    with pytest.raises(TypeError):
        isns.imgplot(data, title_fontdict=[{"fontsize": 20}])


def test_imgplot_return():
    f, ax, cax = isns.imgplot(data)

    assert isinstance(f, Figure)
    assert isinstance(ax, Axes)
    assert isinstance(cax, Axes)


@pytest.mark.parametrize("cmap", [None, "acton", "inferno"])
@pytest.mark.parametrize("cbar", [True, False])
@pytest.mark.parametrize("cbar_label", ["My title", None])
@pytest.mark.parametrize("cbar_fontdict", [{"fontsize": 20}, None])
@pytest.mark.parametrize("showticks", [True, False])
@pytest.mark.parametrize("title", ["My title", None])
@pytest.mark.parametrize("title_fontdict", [{"fontsize": 20}, None])
@pytest.mark.parametrize("describe", [True, False])
def test_imgplot_w_all_valid_inputs(
    cmap, cbar, cbar_label, cbar_fontdict, showticks, title, title_fontdict, describe
):
    f, ax, cax = isns.imgplot(
        data,
        ax=None,
        cmap=cmap,
        describe=describe,
        cbar=cbar,
        cbar_label=cbar_label,
        cbar_fontdict=cbar_fontdict,
        showticks=showticks,
        title=title,
        title_fontdict=title_fontdict,
    )
    plt.close("all")


@pytest.mark.parametrize("bins", ["random", 200.0, -400.13])
def test_imghist_bins_type(bins):
    with pytest.raises(TypeError):
        isns.imghist(data, bins=bins)


@pytest.mark.parametrize("bins", [-100, 0])
def test_imghist_bins_value(bins):
    with pytest.raises(ValueError):
        isns.imghist(data, bins=bins)


def test_imghist_return():
    f, axes, cax = isns.imghist(data)

    assert isinstance(f, Figure)
    assert isinstance(axes[0], Axes)
    assert isinstance(axes[1], Axes)
    assert isinstance(cax, Axes)


@pytest.mark.parametrize("cmap", [None, "acton", "inferno"])
@pytest.mark.parametrize("bins", [None, 500, 10])
@pytest.mark.parametrize("cbar", [True, False])
@pytest.mark.parametrize("cbar_label", ["My title", None])
@pytest.mark.parametrize("cbar_fontdict", [{"fontsize": 20}, None])
@pytest.mark.parametrize("showticks", [True, False])
@pytest.mark.parametrize("title", ["My title", None])
@pytest.mark.parametrize("title_fontdict", [{"fontsize": 20}, None])
@pytest.mark.parametrize("describe", [True, False])
def test_imghist_w_all_valid_inputs(
    cmap, bins, cbar, cbar_label, cbar_fontdict, showticks, title, title_fontdict, describe
):
    f, axes, cax = isns.imghist(
        data,
        cmap=cmap,
        bins=bins,
        describe=describe,
        cbar=cbar,
        cbar_label=cbar_label,
        cbar_fontdict=cbar_fontdict,
        showticks=showticks,
        title=title,
        title_fontdict=title_fontdict,
    )

    plt.close("all")
