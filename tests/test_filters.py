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
        isns.filterplot(data, filter="not-implemented-filter")


@pytest.mark.parametrize("filter", [["gaussian"], ndi.gaussian_filter, None])
def test_filter_types(filter):
    with pytest.raises(TypeError):
        isns.filterplot(data, filter=filter)


@pytest.mark.parametrize("describe", ["True", "False", None, 1])
def test_describe_type(describe):
    with pytest.raises(TypeError):
        isns.imgplot(data, describe=describe)


@pytest.mark.parametrize("filter", isns.implemented_filters)
@pytest.mark.parametrize("describe", [True, False])
def test_filters(filter, describe):
    ax, cax, filt_data = isns.filterplot(data, filter=filter, describe=describe)

    assert isinstance(ax, Axes)
    assert isinstance(cax, Axes)

    plt.close("all")


def test_filterplot_gaussian():
    _, _, filt_data = isns.filterplot(data, filter="gaussian")

    np.testing.assert_array_equal(filt_data, ndi.gaussian_filter(data, sigma=1))

    plt.close("all")


def test_filterplot_sobel():
    _, _, filt_data = isns.filterplot(data, filter="sobel")

    np.testing.assert_array_equal(filt_data, ndi.sobel(data))

    plt.close("all")


def test_filterplot_median():
    _, _, filt_data = isns.filterplot(data, filter="median")

    np.testing.assert_array_equal(filt_data, ndi.median_filter(data, size=5))

    plt.close("all")


def test_filterplot_max():
    _, _, filt_data = isns.filterplot(data, filter="max")

    np.testing.assert_array_equal(filt_data, ndi.maximum_filter(data, size=5))

    plt.close("all")


def test_filterplot_diff_of_gaussian():
    _, _, filt_data = isns.filterplot(data, filter="diff_of_gaussians")

    np.testing.assert_array_equal(filt_data, difference_of_gaussians(data, low_sigma=1))

    plt.close("all")


def test_filterplot_gaussian_gradient_magnitude():
    _, _, filt_data = isns.filterplot(data, filter="gaussian_gradient_magnitude")

    np.testing.assert_array_equal(
        filt_data, ndi.gaussian_gradient_magnitude(data, sigma=1)
    )

    plt.close("all")


def test_filterplot_gaussian_laplace():
    _, _, filt_data = isns.filterplot(data, filter="gaussian_laplace")

    np.testing.assert_array_equal(filt_data, ndi.gaussian_laplace(data, sigma=1))

    plt.close("all")


def test_filterplot_laplace():
    _, _, filt_data = isns.filterplot(data, filter="laplace")

    np.testing.assert_array_equal(filt_data, ndi.laplace(data))

    plt.close("all")


def test_filterplot_min():
    _, _, filt_data = isns.filterplot(data, filter="min")

    np.testing.assert_array_equal(filt_data, ndi.minimum_filter(data, size=5))

    plt.close("all")


def test_filterplot_percentile():
    _, _, filt_data = isns.filterplot(data, filter="percentile")

    np.testing.assert_array_equal(
        filt_data, ndi.percentile_filter(data, percentile=10, size=10)
    )

    plt.close("all")


def test_filterplot_prewitt():
    _, _, filt_data = isns.filterplot(data, filter="prewitt")

    np.testing.assert_array_equal(filt_data, ndi.prewitt(data))

    plt.close("all")


def test_filterplot_rank():
    _, _, filt_data = isns.filterplot(data, filter="rank")

    np.testing.assert_array_equal(filt_data, ndi.rank_filter(data, rank=1, size=10))

    plt.close("all")


def test_filterplot_uniform():
    _, _, filt_data = isns.filterplot(data, filter="uniform")

    np.testing.assert_array_equal(filt_data, ndi.uniform_filter(data))

    plt.close("all")


def test_fftplot_plot():
    ax, cax = isns.fftplot(data)

    assert isinstance(ax, Axes)
    assert isinstance(cax, Axes)

    plt.close("all")


def test_fftplot_fft():
    ax, cax = isns.fftplot(data)

    w_data = data * window("hann", data.shape)
    data_f_mag = fftshift(np.abs(fftn(w_data)))

    np.testing.assert_array_equal(ax.images[0].get_array().data, np.log(data_f_mag))

    plt.close("all")
