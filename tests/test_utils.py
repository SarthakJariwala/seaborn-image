import pytest

import matplotlib
import matplotlib.pyplot as plt

import seaborn_image as isns

matplotlib.use("AGG")  # use non-interactive backend for tests

_all = ["top", "bottom", "right", "left"]


@pytest.mark.parametrize(
    "which", ["all", "top", "bottom", "right", "left", ["top", "right"], _all]
)
def test_despine(which):
    f, ax = plt.subplots()

    if which == "all":
        which = _all

    if isinstance(which, list):
        for side in which:
            isns.despine(which=side)
            # assert ax.spines[side] == False
    else:
        isns.despine(which=which)
        # assert ax.spines[which] == False

    plt.close("all")


def test_despine_value():
    plt.subplots()
    with pytest.raises(ValueError):
        isns.despine(which="center")


def test_despine_type():
    plt.subplots()
    with pytest.raises(TypeError):
        isns.despine(which=0)
