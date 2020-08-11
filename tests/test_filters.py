import pytest

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import seaborn_image as isns

data = np.random.random(2500).reshape((50, 50))
filter_list = ["sobel", "gaussian", "median", "max", "diff_of_gaussians"]


def test_filter_not_implemented():
    with pytest.raises(NotImplementedError):
        isns.filterplot(data, filter="not-implemented-filter")


@pytest.mark.parametrize(
    "filter", [["gaussian"], ndi.gaussian_filter, None]
)
def test_filter_types(filter):
    with pytest.raises(TypeError):
        isns.filterplot(data, filter=filter)


@pytest.mark.parametrize("fft", ["True", "False", None, 1])
def test_fft_type(fft):
    with pytest.raises(TypeError):
        isns.filterplot(data, fft=fft)


@pytest.mark.parametrize("describe", ["True", "False", None, 1])
def test_describe_type(describe):
    with pytest.raises(TypeError):
        isns.imgplot(data, describe=describe)


@pytest.mark.parametrize("fft", [True, False])
@pytest.mark.parametrize(
    "filter", ["sobel", "gaussian", "median", "max", "diff_of_gaussians"]
)
@pytest.mark.parametrize("describe", [True, False])
def test_filters(filter, fft, describe):
    f, ax, filt_data = isns.filterplot(data, filter=filter, fft=fft, describe=describe)

    assert isinstance(f, Figure)
    assert isinstance(ax, np.ndarray)
    assert isinstance(ax.ravel().all(), Axes)
    assert filt_data.all() == data.all()

    plt.close("all")
