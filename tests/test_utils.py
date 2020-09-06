import pytest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import seaborn_image as isns

matplotlib.use("AGG")  # use non-interactive backend for tests

_all = ["top", "bottom", "right", "left"]


@pytest.mark.parametrize(
    "which", ["all", "top", "bottom", "right", "left", ["top", "right"], _all]
)
@pytest.mark.parametrize(
    "fig_to_despine", [plt.subplots(), plt.subplots(nrows=2, ncols=3)]
)
@pytest.mark.parametrize("fig", [None, plt.gcf()])  # test when fig is None and not None
def test_despine(fig_to_despine, fig, which):  # TODO improve this test
    f, ax = fig_to_despine

    if which == "all":
        which = _all

    if isinstance(which, list):
        for side in which:
            isns.despine(which=side)
            # assert ax.spines[side] == False
    else:
        isns.despine(fig=fig, which=which)
        # assert ax.spines[which] == False

    # test axes despine when ndarray
    f, ax = fig_to_despine
    isns.despine(ax=ax)

    plt.close("all")


def test_despine_value():
    plt.subplots()
    with pytest.raises(ValueError):
        isns.despine(which="center")


def test_despine_type():
    plt.subplots()
    with pytest.raises(TypeError):
        isns.despine(which=0)


def test_load_image():
    img = isns.load_image("polymer")
    np.testing.assert_array_equal(
        img, np.loadtxt("data/PolymerImage.txt", skiprows=1) * 1e9
    )

    img = isns.load_image("polymer outliers")
    test_img = np.loadtxt("data/PolymerImage.txt", skiprows=1) * 1e9
    test_img[0, 0] = 80
    np.testing.assert_array_equal(img, test_img)


def test_load_image_error():
    with pytest.raises(ValueError):
        isns.load_image("coins")


def test_scientific_ticks():
    img = isns.load_image("polymer") * 1e-9

    f, ax, cax = isns.imgplot(img)
    isns.scientific_ticks(cax, which="y")
    plt.close()

    f, ax, cax = isns.imgplot(img, orientation="h")
    isns.scientific_ticks(cax, which="x")
    plt.close()

    f, ax, cax = isns.imgplot(img)
    isns.scientific_ticks(cax, which="both")
    plt.close()


def test_scientific_ticks_valueerror():
    with pytest.raises(ValueError):
        img = isns.load_image("polymer") * 1e-9
        f, ax, cax = isns.imgplot(img)
        isns.scientific_ticks(cax, which="all")
        plt.close()
