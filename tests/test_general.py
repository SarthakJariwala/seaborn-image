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
    f, ax, d = isns.imgplot(data)

    assert isinstance(f, Figure)
    assert isinstance(ax, Axes)
    assert d.all() == data.all()


@pytest.mark.parametrize("cmap", ["acton"])
@pytest.mark.parametrize("cbar", [True, False])
@pytest.mark.parametrize("cbar_label", ["My title", None])
@pytest.mark.parametrize("cbar_fontdict", [{"fontsize": 20}])
@pytest.mark.parametrize("showticks", [True, False])
@pytest.mark.parametrize("title", ["My title", None])
@pytest.mark.parametrize("title_fontdict", [{"fontsize": 20}])
def test_all_valid_inputs(
    cmap, cbar, cbar_label, cbar_fontdict, showticks, title, title_fontdict
):
    isns.imgplot(
        data,
        ax=None,
        cmap=cmap,
        cbar=cbar,
        cbar_label=cbar_label,
        cbar_fontdict=cbar_fontdict,
        showticks=showticks,
        title=title,
        title_fontdict=title_fontdict,
    )
