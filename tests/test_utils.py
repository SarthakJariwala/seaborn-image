import pytest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pooch
from skimage import io

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

    img = isns.load_image("fluorescence")
    test_img = np.loadtxt("data/Perovskite.txt")
    np.testing.assert_array_equal(img, test_img)


def test_load_image_from_skimage():
    img = isns.load_image("cells")

    fname = pooch.retrieve(
        url="https://github.com/scikit-image/skimage-tutorials/raw/master/images/cells.tif",
        known_hash="2120cfe08e0396324793a10a905c9bbcb64b117215eb63b2c24b643e1600c8c9",
    )
    test_img = io.imread(fname).T
    np.testing.assert_array_equal(img, test_img)


def test_load_image_error():
    with pytest.raises(ValueError):
        isns.load_image("coins")


def test_scientific_ticks():
    img = isns.load_image("polymer") * 1e-9

    _ = isns.imgplot(img)
    cax = plt.gcf().axes[1]
    isns.scientific_ticks(cax, which="y")
    plt.close()

    _ = isns.imgplot(img, orientation="h")
    cax = plt.gcf().axes[1]
    isns.scientific_ticks(cax, which="x")
    plt.close()

    _ = isns.imgplot(img)
    cax = plt.gcf().axes[1]
    isns.scientific_ticks(cax, which="both")
    plt.close()


def test_scientific_ticks_valueerror():
    with pytest.raises(ValueError):
        img = isns.load_image("polymer") * 1e-9
        _ = isns.imgplot(img)
        cax = plt.gcf().axes[1]
        isns.scientific_ticks(cax, which="all")
        plt.close()
