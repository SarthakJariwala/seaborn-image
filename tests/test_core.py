import pytest

import matplotlib

matplotlib.use("AGG")  # use non-interactive backend for tests

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


def test_setup_scalebar():
    with pytest.raises(AttributeError):
        img_setup = isns._core._SetupImage(data, dx=1)
        f, ax = img_setup._setup_figure()
        img_setup._setup_scalebar(ax)


def test_setup_scalebar_dimension():
    with pytest.raises(ValueError):
        img_setup = isns._core._SetupImage(
            data, dx=1, units="nm", dimension="imperial-reciprocal"
        )
        f, ax = img_setup._setup_figure()
        img_setup._setup_scalebar(ax)


def test_cbar_orientation():
    with pytest.raises(ValueError):
        img_setup = isns._core._SetupImage(data, cbar=True, orientation="right")
        f, ax, cax = img_setup.plot()


def test_robust_param():
    img_setup = isns._core._SetupImage(data, robust=True, perc=(2, 98))
    f, ax, cax = img_setup.plot()
    assert img_setup.vmin == np.nanpercentile(data, 2)
    assert img_setup.vmax == np.nanpercentile(data, 98)
    plt.close()

    img_setup = isns._core._SetupImage(data, robust=True, perc=(2, 98), vmin=0)
    f, ax, cax = img_setup.plot()
    assert img_setup.vmin == 0
    assert img_setup.vmax == np.nanpercentile(data, 98)
    plt.close()

    img_setup = isns._core._SetupImage(data, robust=True, perc=(2, 98), vmax=1)
    f, ax, cax = img_setup.plot()
    assert img_setup.vmin == np.nanpercentile(data, 2)
    assert img_setup.vmax == 1
    plt.close()

    img_setup = isns._core._SetupImage(data, robust=True, perc=(2, 98), vmin=0, vmax=1)
    f, ax, cax = img_setup.plot()
    assert img_setup.vmin == 0
    assert img_setup.vmax == 1
    plt.close()


def test_log_scale_cbar():
    img_setup = isns._core._SetupImage(data, norm="cbar_log")
    f, ax, cax = img_setup.plot()
    assert isinstance(img_setup.norm, matplotlib.colors.LogNorm)
    plt.close()


def test_cbar_despine():
    # change the global despine state
    isns.set_image(despine=False)
    # cbar needs to be True for this test
    img_setup = isns._core._SetupImage(data, cbar=True)
    f, ax, cax = img_setup.plot()
    # changing the global state should reflect here
    assert img_setup.despine == False
    plt.close()

    # change the global despine state
    isns.set_image(despine=True)
    # cbar needs to be True for this test
    img_setup = isns._core._SetupImage(data, cbar=True)
    _ = img_setup.plot()
    # changing the global state should reflect here
    assert img_setup.despine == True
    plt.close()


def test_local_despine_wrt_global_despine():
    # Global despine=False
    isns.set_image(despine=False)

    img_setup = isns._core._SetupImage(data)
    f, ax, cax = img_setup.plot()
    for spine in ["top", "bottom", "right", "left"]:
        assert ax.spines[spine].get_visible() == True

    # if global state is despine=False but local state is despine=True
    # it should respect local state
    img_setup = isns._core._SetupImage(data, despine=True)
    f, ax, cax = img_setup.plot()
    for spine in ["top", "bottom", "right", "left"]:
        assert ax.spines[spine].get_visible() == False

    # Global despine=True
    isns.set_image(despine=True)

    img_setup = isns._core._SetupImage(data)
    f, ax, cax = img_setup.plot()
    for spine in ["top", "bottom", "right", "left"]:
        assert ax.spines[spine].get_visible() == False

    # if global state is despine=True but local state is despine=False
    # it should respect local state
    img_setup = isns._core._SetupImage(data, despine=False)
    f, ax, cax = img_setup.plot()
    for spine in ["top", "bottom", "right", "left"]:
        assert ax.spines[spine].get_visible() == True


def test_data_plotted_is_same_as_input():
    img_setup = isns._core._SetupImage(data)
    f, ax, cax = img_setup.plot()

    # check if data iput is what was plotted
    np.testing.assert_array_equal(ax.images[0].get_array().data, data)
    plt.close("all")


@pytest.mark.parametrize(
    "cmap", [None, "acton"]
)  # test if seaborn-image supplied cmaps are working
@pytest.mark.parametrize(
    "dx, units, dimension",
    [(None, None, None), (1, "nm", "si"), (1, "1/um", "si-reciprocal")],
)
@pytest.mark.parametrize("cbar", [True, False])
@pytest.mark.parametrize("orientation", ["horizontal", "h", "vertical", "v"])
@pytest.mark.parametrize("showticks", [True, False])
@pytest.mark.parametrize("despine", [True, False])
@pytest.mark.parametrize("vmin", [None, 0])
@pytest.mark.parametrize("vmax", [None, 1])
@pytest.mark.parametrize("robust", [True, False])
def test_plot_w_all_inputs(
    cmap,
    vmin,
    vmax,
    cbar,
    dx,
    units,
    dimension,
    orientation,
    showticks,
    despine,
    robust,
):
    img_setup = isns._core._SetupImage(
        data,
        cmap=cmap,
        vmin=None,
        vmax=None,
        robust=robust,
        perc=(2, 98),
        dx=dx,
        units=units,
        dimension=dimension,
        cbar=cbar,
        orientation=orientation,
        cbar_label="cbar label",
        cbar_ticks=[],
        showticks=showticks,
        despine=despine,
    )
    f, ax, cax = img_setup.plot()

    assert isinstance(f, Figure)
    assert isinstance(ax, Axes)
    if cbar is True:
        assert isinstance(cax, Axes)
    else:
        assert cax is None

    plt.close("all")
