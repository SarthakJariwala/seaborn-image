import pytest

import numpy as np

import seaborn_image as isns

data = np.random.random(2500).reshape((50, 50))


def test_filter_not_implemented():
    with pytest.raises(NotImplementedError):
        isns.filterplot(data, filter="not-implemented-filter")


@pytest.mark.parametrize(
    "filter,fft", [(["gaussian"], True), ("gaussian", "True"), (["gaussian"], "True")]
)
def test_filter_types(filter, fft):
    with pytest.raises(TypeError):
        isns.filterplot(data, filter=filter, fft=fft)
