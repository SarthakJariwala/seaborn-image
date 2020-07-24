import pytest

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import seaborn_image as isns

data = np.random.random(2500).reshape((50, 50))


def test_setup_figure():
    img_setup = isns._core._SetupImage(data)
    f, ax = img_setup._setup_figure()

    assert isinstance(f, Figure)
    assert isinstance(ax, Axes)


def test_setup_figure_check_title_dict():
    with pytest.raises(TypeError):
        img_setup = isns._core._SetupImage(data, title_dict=[{"fontsize": 20}])
        f, ax = img_setup._setup_figure()


def test_setup_scalebar():
    with pytest.raises(AttributeError):
        img_setup = isns._core._SetupImage(data, dx=1)
        f, ax = img_setup._setup_figure()
        img_setup._setup_scalebar(ax)


def test_plot_check_cbar_dict():
    with pytest.raises(TypeError):
        img_setup = isns._core._SetupImage(
            data, cbar=True, cbar_fontdict=[{"fontsize": 20}]
        )
        f, ax = img_setup.plot()


@pytest.mark.parametrize("cmap", [None, "acton"])
@pytest.mark.parametrize("vmin", [None])
@pytest.mark.parametrize("vmax", [None])
@pytest.mark.parametrize("title", [None, "My Title"])
@pytest.mark.parametrize("fontdict", [None, {"fontsize": 20}])
@pytest.mark.parametrize("dx", [None, 1])
@pytest.mark.parametrize(
    "units", ["m", "um"]
)  # units can't be None when dx is not None
@pytest.mark.parametrize("cbar", [None, True, False])
@pytest.mark.parametrize("cbar_fontdict", [None, {"fontsize": 20}])
@pytest.mark.parametrize("cbar_label", [None, "Cbar Label"])
@pytest.mark.parametrize("cbar_ticks", [None, [0, 1, 2]])
@pytest.mark.parametrize("showticks", [None, True, False])
def test_plot_w_all_inputs(
    cmap,
    vmin,
    vmax,
    title,
    fontdict,
    dx,
    units,
    cbar,
    cbar_fontdict,
    cbar_label,
    cbar_ticks,
    showticks,
):
    img_setup = isns._core._SetupImage(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        title=title,
        fontdict=fontdict,
        dx=dx,
        units=units,
        cbar=cbar,
        cbar_fontdict=cbar_fontdict,
        cbar_label=cbar_label,
        cbar_ticks=cbar_ticks,
        showticks=showticks,
    )
    f, ax = img_setup.plot()

    assert isinstance(f, Figure)
    assert isinstance(ax, Axes)
