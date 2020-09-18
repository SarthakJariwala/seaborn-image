import pytest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import seaborn_image as isns

matplotlib.use("AGG")  # use non-interactive backend for tests


class TestImageGrid:

    img_3d = np.random.random(50 * 50 * 4).reshape((50, 50, 4))

    data = np.random.random(2500).reshape((50, 50))
    img_list = [data, data, data]

    def test_none_data(self):
        with pytest.raises(ValueError):
            isns.ImageGrid(None)

    def test_self_data(self):
        g = isns.ImageGrid(self.data)
        np.testing.assert_array_equal(self.data, g.data)
        plt.close()

        g = isns.ImageGrid(self.img_list)
        np.testing.assert_array_equal(self.img_list, g.data)
        plt.close()

        g = isns.ImageGrid(self.img_3d)
        np.testing.assert_array_equal(self.img_3d, g.data)
        plt.close()

    def test_self_fig(self):
        g = isns.ImageGrid(self.data)
        assert isinstance(g.fig, Figure)
        plt.close()

    def test_self_axes(self):

        g0 = isns.ImageGrid(self.data)
        for ax in g0.axes.flat:
            assert isinstance(ax, Axes)
        plt.close()

        g1 = isns.ImageGrid(self.img_list)
        for ax in g1.axes.flat:
            assert isinstance(ax, Axes)
        plt.close()

        g2 = isns.ImageGrid(self.img_3d)
        for ax in g2.axes.flat:
            assert isinstance(ax, Axes)
        plt.close()

    def test_axes_shape(self):

        g0 = isns.ImageGrid(self.data)
        assert g0.axes.shape == (1, 1)

        g1 = isns.ImageGrid(self.img_3d)
        assert g1.axes.shape == (2, 3)

        g2 = isns.ImageGrid(self.img_list)
        assert g2.axes.shape == (1, 3)

        plt.close("all")

    def test_col_wrap(self):

        g0 = isns.ImageGrid(self.data, col_wrap=3)
        # since it is only 1 image;
        # col_wrap should revert to min no of images
        assert g0.axes.shape == (1, 1)
        plt.close()

        g1 = isns.ImageGrid(self.img_list, col_wrap=2)
        assert g1.axes.shape == (2, 2)
        plt.close()

        g2 = isns.ImageGrid(self.img_3d, col_wrap=3)
        assert g2.axes.shape == (2, 3)
        plt.close()

    def test_slices(self):

        g = isns.ImageGrid(self.img_3d, slices=[0, 2])
        ax = g.axes.flat
        np.testing.assert_array_equal(
            ax[0].images[0].get_array().data, self.img_3d[:, :, 0]
        )
        np.testing.assert_array_equal(
            ax[1].images[0].get_array().data, self.img_3d[:, :, 2]
        )
        plt.close()

    def test_cbar_list(self):

        isns.ImageGrid(self.img_list, cmap=["acton", "inferno", "ice"])
        plt.close()

        isns.ImageGrid(self.img_list, cmap=["acton", None, None])
        plt.close()

        isns.ImageGrid(self.img_list, cmap=[None, "inferno", "ice"])
        plt.close()

    def test_robust(self):

        isns.ImageGrid(self.img_list, robust=True)
        plt.close()

        isns.ImageGrid(self.img_list, robust=True, perc=[(2, 98), (1, 99), (2, 99)])
        plt.close()

        isns.ImageGrid(
            self.img_list, robust=[True, False, True], perc=[(2, 98), (1, 99), (2, 99)]
        )
        plt.close()

        isns.ImageGrid(self.img_3d, robust=True)
        plt.close()

        isns.ImageGrid(self.data, robust=True)
        plt.close()

    def test_scalebar_list(self):

        isns.ImageGrid(
            self.img_list,
            dx=[1, 2, 3],
            units=["m", "m", "m"],
            dimension=["si", "si", "si"],
        )
        plt.close()

        isns.ImageGrid(
            self.img_list,
            dx=[1, 2, 3],
            units="m",
            dimension="si",
        )
        plt.close()

        isns.ImageGrid(
            self.img_list,
            dx=1,
            units=["m", "nm", "um"],
            dimension=["si", "si", "si"],
        )
        plt.close()

        isns.ImageGrid(
            self.img_list,
            dx=[1, None, 3],
            units=["m", None, "m"],
            dimension=["si", None, "si"],
        )
        plt.close()

    def test_cbar(self):

        isns.ImageGrid(
            self.img_list,
            cbar=[True, True, False],
        )
        plt.close()

        isns.ImageGrid(
            self.img_list,
            cbar_label=["X", "A", "B"],
        )
        plt.close()

        isns.ImageGrid(
            self.img_list,
            cbar_label=[None, "A", "B"],
        )
        plt.close()

        isns.ImageGrid(
            self.img_list,
            cbar=False,
            cbar_label=[None, "A", "B"],
        )
        plt.close()

    def test_figure_size(self):

        g0 = isns.ImageGrid(self.data)
        np.testing.assert_array_equal(g0.fig.get_size_inches(), (3, 3))
        plt.close()

        g1 = isns.ImageGrid(self.img_list)
        np.testing.assert_array_equal(g1.fig.get_size_inches(), (9, 3))
        plt.close()

        g2 = isns.ImageGrid(self.img_3d)
        np.testing.assert_array_equal(g2.fig.get_size_inches(), (9, 6))
        plt.close()

        g3 = isns.ImageGrid(self.img_3d, slices=[1, 2])
        np.testing.assert_array_equal(g3.fig.get_size_inches(), (6, 3))
        plt.close()

        g4 = isns.ImageGrid(
            self.img_3d,
            height=2,
            aspect=1.5,
        )
        np.testing.assert_array_equal(g4.fig.get_size_inches(), (3 * 2 * 1.5, 2 * 2))
        plt.close()


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
