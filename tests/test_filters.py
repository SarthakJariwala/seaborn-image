import pytest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.fftpack import fftn, fftshift
from skimage.filters import difference_of_gaussians, window

import seaborn_image as isns

matplotlib.use("AGG")  # use non-interactive backend for tests


data = np.random.random(2500).reshape((50, 50))


def test_filter_not_implemented():
    with pytest.raises(NotImplementedError):
        isns.filterplot(data, filt="not-implemented-filt")


@pytest.mark.parametrize("filt", [["gaussian"], None])
def test_filter_types(filt):
    with pytest.raises(TypeError):
        isns.filterplot(data, filt=filt)


@pytest.mark.parametrize("describe", ["True", "False", None, 1])
def test_describe_type(describe):
    with pytest.raises(TypeError):
        isns.filterplot(data, "sobel", describe=describe)


@pytest.mark.parametrize("describe", [True, False])
def test_filterplot_describe(describe):
    ax = isns.filterplot(data, "sobel", describe=describe)

    assert isinstance(ax, Axes)

    plt.close("all")


def test_filterplot_callable_filt():
    "Test a callable filt parameter with additional parameters passed to the callable filt function"
    ax = isns.filterplot(data, ndi.uniform_filter, size=5, mode="nearest")

    np.testing.assert_array_equal(
        ax.images[0].get_array().data, ndi.uniform_filter(data, size=5, mode="nearest")
    )

    plt.close("all")


def test_filterplot_gaussian():
    ax = isns.filterplot(data, filt="gaussian", sigma=1)

    np.testing.assert_array_equal(
        ax.images[0].get_array().data, ndi.gaussian_filter(data, sigma=1)
    )

    plt.close("all")


def test_filterplot_sobel():
    ax = isns.filterplot(data, filt="sobel")

    np.testing.assert_array_equal(ax.images[0].get_array().data, ndi.sobel(data))

    plt.close("all")


def test_filterplot_median():
    ax = isns.filterplot(data, filt="median", size=5)

    np.testing.assert_array_equal(
        ax.images[0].get_array().data, ndi.median_filter(data, size=5)
    )

    plt.close("all")


def test_filterplot_max():
    ax = isns.filterplot(data, filt="max", size=5)

    np.testing.assert_array_equal(
        ax.images[0].get_array().data, ndi.maximum_filter(data, size=5)
    )

    plt.close("all")


def test_filterplot_diff_of_gaussian():
    ax = isns.filterplot(data, filt="diff_of_gaussians", low_sigma=1)

    np.testing.assert_array_equal(
        ax.images[0].get_array().data, difference_of_gaussians(data, low_sigma=1)
    )

    plt.close("all")


def test_filterplot_gaussian_gradient_magnitude():
    ax = isns.filterplot(data, filt="gaussian_gradient_magnitude", sigma=1)

    np.testing.assert_array_equal(
        ax.images[0].get_array().data, ndi.gaussian_gradient_magnitude(data, sigma=1)
    )

    plt.close("all")


def test_filterplot_gaussian_laplace():
    ax = isns.filterplot(data, filt="gaussian_laplace", sigma=1)

    np.testing.assert_array_equal(
        ax.images[0].get_array().data, ndi.gaussian_laplace(data, sigma=1)
    )

    plt.close("all")


def test_filterplot_laplace():
    ax = isns.filterplot(data, filt="laplace")

    np.testing.assert_array_equal(ax.images[0].get_array().data, ndi.laplace(data))

    plt.close("all")


def test_filterplot_min():
    ax = isns.filterplot(data, filt="min", size=5)

    np.testing.assert_array_equal(
        ax.images[0].get_array().data, ndi.minimum_filter(data, size=5)
    )

    plt.close("all")


def test_filterplot_percentile():
    ax = isns.filterplot(data, filt="percentile", percentile=10, size=10)

    np.testing.assert_array_equal(
        ax.images[0].get_array().data,
        ndi.percentile_filter(data, percentile=10, size=10),
    )

    plt.close("all")


def test_filterplot_prewitt():
    ax = isns.filterplot(data, filt="prewitt")

    np.testing.assert_array_equal(ax.images[0].get_array().data, ndi.prewitt(data))

    plt.close("all")


def test_filterplot_rank():
    ax = isns.filterplot(data, filt="rank", rank=1, size=10)

    np.testing.assert_array_equal(
        ax.images[0].get_array().data, ndi.rank_filter(data, rank=1, size=10)
    )

    plt.close("all")


def test_filterplot_uniform():
    ax = isns.filterplot(data, filt="uniform")

    np.testing.assert_array_equal(
        ax.images[0].get_array().data, ndi.uniform_filter(data)
    )

    plt.close("all")


def test_fftplot_plot():
    ax = isns.fftplot(data)

    assert isinstance(ax, Axes)

    plt.close("all")


def test_fftplot_fft():
    ax = isns.fftplot(data)

    w_data = data * window("hann", data.shape)
    data_f_mag = fftshift(np.abs(fftn(w_data)))

    np.testing.assert_array_equal(ax.images[0].get_array().data, np.log(data_f_mag))

    plt.close("all")
