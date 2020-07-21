import pytest

import numpy as np

import seaborn_image as isns

data = np.random.random(2500).reshape((50, 50))


def test_filter_not_implemented():
    with pytest.raises(NotImplementedError):
        isns.filterplot(data, filter="min")
