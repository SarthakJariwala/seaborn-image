import pytest

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import seaborn_image as isns

data = np.random.random(2500).reshape((50, 50))
filter_list = ["sobel", "gaussian", "median", "max", "diff_of_gaussians"]


def test_filter_not_implemented():
    with pytest.raises(NotImplementedError):
        isns.filterplot(data, filter="not-implemented-filter")


@pytest.mark.parametrize(
    "filter,fft", [(["gaussian"], True), ("gaussian", "True"), (["gaussian"], "True")]
)
def test_filter_types(filter, fft):
    with pytest.raises(TypeError):
        isns.filterplot(data, filter=filter, fft=fft)


@pytest.mark.parametrize("fft", [True, False])
@pytest.mark.parametrize(
    "filter", ["sobel", "gaussian", "median", "max", "diff_of_gaussians"]
)
def test_filters(filter, fft):
    f, ax, filt_data = isns.filterplot(data, filter=filter, fft=fft)

    assert isinstance(f, Figure)
    assert isinstance(ax, np.ndarray)
    assert isinstance(ax.ravel().all(), Axes)
    assert filt_data.all() == data.all()
