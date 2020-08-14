import pytest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from skimage.filters import difference_of_gaussians

import seaborn_image as isns

matplotlib.use("AGG")  # use non-interactive backend for tests


data = np.random.random(2500).reshape((50, 50))
filter_list = ["sobel", "gaussian", "median", "max", "diff_of_gaussians"]


def test_filter_not_implemented():
    with pytest.raises(NotImplementedError):
        isns.filterplot(data, filter="not-implemented-filter")


@pytest.mark.parametrize("filter", [["gaussian"], ndi.gaussian_filter, None])
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

    plt.close("all")


def test_filterplot_gaussian():
    f, ax, filt_data = isns.filterplot(data, filter="gaussian")

    np.testing.assert_array_equal(filt_data, ndi.gaussian_filter(data, sigma=1))

    plt.close("all")


def test_filterplot_sobel():
    f, ax, filt_data = isns.filterplot(data, filter="sobel")

    np.testing.assert_array_equal(filt_data, ndi.sobel(data))

    plt.close("all")


def test_filterplot_median():
    f, ax, filt_data = isns.filterplot(data, filter="median")

    np.testing.assert_array_equal(filt_data, ndi.median_filter(data, size=5))

    plt.close("all")


def test_filterplot_max():
    f, ax, filt_data = isns.filterplot(data, filter="max")

    np.testing.assert_array_equal(filt_data, ndi.maximum_filter(data, size=5))

    plt.close("all")


def test_filterplot_diff_of_gaussian():
    f, ax, filt_data = isns.filterplot(data, filter="diff_of_gaussians")

    np.testing.assert_array_equal(filt_data, difference_of_gaussians(data, low_sigma=1))

    plt.close("all")
