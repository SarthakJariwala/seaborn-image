import pytest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import seaborn_image as isns

matplotlib.use("AGG")  # use non-interactive backend for tests


class TestFilterGrid(object):

    data = np.random.random(2500).reshape((50, 50))

    def test_none_data(self):
        with pytest.raises(ValueError):
            isns.FilterGrid(None, "sobel")

    def test_none_filt(self):
        with pytest.raises(ValueError):
            isns.FilterGrid(self.data, None)

    def test_self_data(self):
        g = isns.FilterGrid(self.data, "sobel")
        np.testing.assert_array_equal(self.data, g.data)
        plt.close()

    def test_self_fig(self):
        g = isns.FilterGrid(self.data, "sobel")
        assert isinstance(g.fig, Figure)
        plt.close()

    def test_self_axes(self):

        g0 = isns.FilterGrid(self.data, "sobel")
        for ax in g0.axes.flat:
            assert isinstance(ax, Axes)

        g1 = isns.FilterGrid(self.data, "gaussian", row="sigma", sigma=[1, 2, 3])
        for ax in g1.axes.flat:
            assert isinstance(ax, Axes)

        g2 = isns.FilterGrid(
            self.data,
            "gaussian",
            row="mode",
            col="sigma",
            sigma=[1, 2, 3],
            mode=["reflect", "nearest"],
        )
        for ax in g2.axes.flat:
            assert isinstance(ax, Axes)

        plt.close("all")

    def test_axes_shape(self):

        g0 = isns.FilterGrid(self.data, "sobel")
        assert g0.axes.shape == (1, 1)

        g1 = isns.FilterGrid(self.data, "gaussian", row="sigma", sigma=[1, 2, 3])
        assert g1.axes.shape == (3, 1)

        g2 = isns.FilterGrid(self.data, "gaussian", col="sigma", sigma=[1, 2, 3])
        assert g2.axes.shape == (1, 3)

        g3 = isns.FilterGrid(
            self.data,
            "gaussian",
            row="sigma",
            col="mode",
            sigma=[1, 2, 3],
            mode=["reflect", "nearest"],
        )
        assert g3.axes.shape == (3, 2)

        g4 = isns.FilterGrid(
            self.data,
            "gaussian",
            row="mode",
            col="sigma",
            sigma=[1, 2, 3],
            mode=["reflect", "nearest"],
        )
        assert g4.axes.shape == (2, 3)

        for ax in g4.axes.flat:
            assert isinstance(ax, Axes)

        plt.close("all")

    def test_col_wrap(self):

        g0 = isns.FilterGrid(
            self.data, "gaussian", col="sigma", sigma=[1, 2, 3, 4, 5], col_wrap=3
        )
        assert g0.axes.shape == (2, 3)
        plt.close()

        with pytest.raises(ValueError):
            isns.FilterGrid(
                self.data, "gaussian", row="sigma", sigma=[1, 2, 3, 4, 5], col_wrap=3
            )

        with pytest.raises(ValueError):
            isns.FilterGrid(
                self.data,
                "gaussian",
                row="mode",
                col="sigma",
                col_wrap=3,
                sigma=[1, 2, 3, 4, 5],
                mode=["reflect", "nearest"],
            )

    def test_additional_kwargs_for_filters(self):

        isns.FilterGrid(
            self.data, "gaussian", row="sigma", sigma=[1, 2, 3], mode="reflect"
        )
        plt.close()

        isns.FilterGrid(
            self.data, "gaussian", col="sigma", sigma=[1, 2, 3], mode="reflect"
        )
        plt.close()

        isns.FilterGrid(
            self.data,
            "gaussian",
            row="sigma",
            col="mode",
            sigma=[1, 2, 3],
            mode=["reflect", "nearest", "constant"],
            cval=0.2,
        )
        plt.close()

    def test_figure_size(self):

        g0 = isns.FilterGrid(self.data, "sobel")
        np.testing.assert_array_equal(g0.fig.get_size_inches(), (3, 3))

        g1 = isns.FilterGrid(self.data, "gaussian", row="sigma", sigma=[1, 2, 3])
        np.testing.assert_array_equal(g1.fig.get_size_inches(), (3, 9))

        g2 = isns.FilterGrid(
            self.data,
            "gaussian",
            row="sigma",
            col="mode",
            sigma=[1, 2, 3],
            mode=["reflect", "nearest"],
        )
        np.testing.assert_array_equal(g2.fig.get_size_inches(), (6, 9))

        g3 = isns.FilterGrid(
            self.data,
            "gaussian",
            row="sigma",
            col="mode",
            sigma=[1, 2, 3],
            mode=["reflect", "nearest"],
            height=2,
        )
        np.testing.assert_array_equal(g3.fig.get_size_inches(), (4, 6))

        g4 = isns.FilterGrid(
            self.data,
            "gaussian",
            row="sigma",
            col="mode",
            sigma=[1, 2, 3],
            mode=["reflect", "nearest"],
            height=2,
            aspect=1.5,
        )
        np.testing.assert_array_equal(g4.fig.get_size_inches(), (4 * 1.5, 6))

        plt.close("all")
