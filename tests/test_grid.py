import warnings
import pytest

from copy import copy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import ndimage as ndi
from skimage.data import astronaut
from skimage.filters import gaussian, hessian, median

import seaborn_image as isns

# matplotlib.use("AGG")  # use non-interactive backend for tests

cells = isns.load_image("cells")


class TestImageGrid:
    img_3d = np.random.random(4 * 4 * 4).reshape((4, 4, 4))
    img_4d = np.random.random(4 * 4 * 4 * 3).reshape((4, 4, 4, 3))

    data = np.random.random(2500).reshape((50, 50))
    img_list = [data, data, data]

    data_3d = np.random.random(2500 * 3).reshape((50, 50, 3))
    img_3d_list = [data_3d, data_3d, data_3d]

    img_mixed_list = [data_3d, data, data_3d]

    data_3d_bad = np.random.random(2500 * 6).reshape((50, 50, 6))
    img_bad_list1 = [data, data_3d_bad, data_3d]
    img_bad_list2 = [data, img_4d, data_3d]

    def test_none_data(self):
        with pytest.raises(ValueError):
            isns.ImageGrid(None)
            plt.close()

    def test_higher_dim_data(self):
        with pytest.raises(ValueError):
            isns.ImageGrid(
                np.random.random(50 * 50 * 4 * 3 * 3).reshape((50, 50, 4, 3, 3))
            )
            plt.close()

    def test_incorrect_channels(self):
        with pytest.raises(ValueError):
            isns.ImageGrid(np.random.random(6 * 50 * 50 * 5).reshape((6, 50, 50, 5)))
            plt.close()

    def test_incorrect_axis_for_slicing(self):
        with pytest.raises(ValueError):
            isns.ImageGrid(self.img_3d, axis=3)
            plt.close()

    def test_self_data(self):
        with pytest.warns(RuntimeWarning):
            g = isns.ImageGrid(self.data)
            np.testing.assert_array_equal(self.data, g.data)
            plt.close()

        g = isns.ImageGrid(self.img_list)
        np.testing.assert_array_equal(self.img_list, g.data)
        plt.close()

        g = isns.ImageGrid(self.img_3d_list)
        np.testing.assert_array_equal(self.img_3d_list, g.data)
        plt.close()

        g = isns.ImageGrid(self.img_mixed_list)
        for idx, aux_img in enumerate(self.img_mixed_list):
            np.testing.assert_array_equal(aux_img, g.data[idx])
        plt.close()

        g = isns.ImageGrid(self.img_3d)
        np.testing.assert_array_equal(self.img_3d, g.data)
        plt.close()

        g = isns.ImageGrid(self.img_4d)
        np.testing.assert_array_equal(self.img_4d, g.data)
        plt.close()

        with pytest.raises(ValueError):
            g = isns.ImageGrid(self.img_bad_list1)
            plt.close()

        with pytest.raises(ValueError):
            g = isns.ImageGrid(self.img_bad_list2)
            plt.close()

        with pytest.raises(ValueError):
            g = isns.ImageGrid(0)
            plt.close()

    def test_self_fig(self):
        g = isns.ImageGrid([self.data])
        assert isinstance(g.fig, Figure)
        plt.close()

    def test_self_axes(self):
        g0 = isns.ImageGrid([self.data])
        for ax in g0.axes.flat:
            assert isinstance(ax, Axes)
        plt.close()

        g1 = isns.ImageGrid(self.img_list)
        for ax in g1.axes.flat:
            assert isinstance(ax, Axes)
        plt.close()

        g2 = isns.ImageGrid(self.img_3d_list)
        for ax in g2.axes.flat:
            assert isinstance(ax, Axes)
        plt.close()

        g3 = isns.ImageGrid(self.img_mixed_list)
        for ax in g3.axes.flat:
            assert isinstance(ax, Axes)
        plt.close()

        g4 = isns.ImageGrid(self.img_3d)
        for ax in g4.axes.flat:
            assert isinstance(ax, Axes)
        plt.close()

        g5 = isns.ImageGrid(self.img_4d)
        for ax in g5.axes.flat:
            assert isinstance(ax, Axes)
        plt.close()

    def test_axes_shape(self):
        g0 = isns.ImageGrid([self.data])
        assert g0.axes.shape == (1, 1)
        plt.close()

        g1 = isns.ImageGrid(self.img_3d)
        assert g1.axes.shape == (2, 3)
        plt.close()

        g2 = isns.ImageGrid(self.img_4d)
        assert g2.axes.shape == (2, 3)
        plt.close()

        g3 = isns.ImageGrid(self.img_list)
        assert g3.axes.shape == (1, 3)
        plt.close()

        g4 = isns.ImageGrid(self.img_3d_list)
        assert g4.axes.shape == (1, 3)
        plt.close()

        g5 = isns.ImageGrid(self.img_mixed_list)
        assert g5.axes.shape == (1, 3)
        plt.close()

    def test_map_func(self):
        # test map_func is callable
        with pytest.raises(TypeError):
            isns.ImageGrid(self.img_3d, map_func="gaussian")
            plt.close()

        # 3D image with single map_func
        g0 = isns.ImageGrid(self.img_3d, map_func=gaussian)
        ax = g0.axes.flat
        np.testing.assert_array_equal(
            ax[0].images[0].get_array().data, gaussian(self.img_3d)[:, :, 0]
        )
        np.testing.assert_array_equal(
            ax[1].images[0].get_array().data, gaussian(self.img_3d)[:, :, 1]
        )
        plt.close()

        # 4D image data with single map_func
        g1 = isns.ImageGrid(self.img_4d, map_func=gaussian)
        g_img_4d = gaussian(self.img_4d)
        for idx, ax in enumerate(g1.axes.flat):
            if ax.images:
                np.testing.assert_array_equal(
                    ax.images[0].get_array().data, g_img_4d[idx, :, :, :]
                )

        # List of 3D images with a single map_func
        g2 = isns.ImageGrid(self.img_3d_list, map_func=gaussian)
        ax = g2.axes.flat
        for idx, val in enumerate(self.img_3d_list):
            np.testing.assert_array_equal(
                ax[idx].images[0].get_array().data, gaussian(val)
            )
        plt.close()

        # List of 3D, 2D images with a single map_func
        g2 = isns.ImageGrid(self.img_mixed_list, map_func=gaussian)
        ax = g2.axes.flat
        for idx, val in enumerate(self.img_mixed_list):
            np.testing.assert_array_equal(
                ax[idx].images[0].get_array().data, gaussian(val)
            )
        plt.close()

        # List of 2D images with single map_func
        pol = isns.load_image("polymer")
        pl = isns.load_image("fluorescence")
        new_img_list = [pol, pl]
        g3 = isns.ImageGrid(new_img_list, map_func=gaussian)
        ax = g3.axes.flat
        np.testing.assert_array_equal(ax[0].images[0].get_array().data, gaussian(pol))
        np.testing.assert_array_equal(ax[1].images[0].get_array().data, gaussian(pl))
        plt.close()

        with pytest.warns(RuntimeWarning):
            # Single 2D image with a single map_func
            g = isns.ImageGrid(pol, map_func=gaussian)
            ax = g.axes.flat
            np.testing.assert_array_equal(
                ax[0].images[0].get_array().data, gaussian(pol)
            )
            plt.close()

        with pytest.warns(RuntimeWarning):
            # Single 2D image with a list of map_func
            g = isns.ImageGrid(pol, map_func=[gaussian, gaussian])
            ax = g.axes.flat
            np.testing.assert_array_equal(
                ax[0].images[0].get_array().data, gaussian(pol)
            )
            np.testing.assert_array_equal(
                ax[1].images[0].get_array().data, gaussian(pol)
            )
            plt.close()

        # List of 2D images with a list of map_func
        g = isns.ImageGrid([pol, pl], map_func=[gaussian, gaussian])
        ax = g.axes.flat
        np.testing.assert_array_equal(ax[0].images[0].get_array().data, gaussian(pol))
        np.testing.assert_array_equal(ax[1].images[0].get_array().data, gaussian(pl))
        np.testing.assert_array_equal(ax[2].images[0].get_array().data, gaussian(pol))
        np.testing.assert_array_equal(ax[3].images[0].get_array().data, gaussian(pl))
        plt.close()

        with pytest.warns(RuntimeWarning):
            g = isns.ImageGrid(pol, map_func=[gaussian, gaussian])
            plt.close()

        # 3 Image with list of map_func
        with pytest.raises(ValueError):
            g = isns.ImageGrid(self.img_3d, map_func=[gaussian, gaussian])
            plt.close()

        # List of map_func must all be callables
        with pytest.raises(TypeError):
            g = isns.ImageGrid([pol, pl], map_func=[gaussian, "hessian"])
            plt.close()

    def test_map_func_kw(self):
        # kwargs for a single map_func for 3D image
        g = isns.ImageGrid(self.img_3d, map_func=gaussian, map_func_kw={"sigma": 1.5})
        ax = g.axes.flat
        np.testing.assert_array_equal(
            ax[0].images[0].get_array().data, gaussian(self.img_3d, sigma=1.5)[:, :, 0]
        )
        np.testing.assert_array_equal(
            ax[1].images[0].get_array().data, gaussian(self.img_3d, sigma=1.5)[:, :, 1]
        )
        plt.close()

        # kwargs for single map_func for list of 2D images
        pol = isns.load_image("polymer")
        pl = isns.load_image("fluorescence")
        g = isns.ImageGrid([pol, pl], map_func=gaussian, map_func_kw={"sigma": 1.5})
        ax = g.axes.flat
        np.testing.assert_array_equal(
            ax[0].images[0].get_array().data, gaussian(pol, sigma=1.5)
        )
        np.testing.assert_array_equal(
            ax[1].images[0].get_array().data, gaussian(pl, sigma=1.5)
        )
        plt.close()

        with pytest.warns(RuntimeWarning):
            # kwargs for single map_func for a single 2D image
            g = isns.ImageGrid(pol, map_func=gaussian, map_func_kw={"sigma": 1.5})
            ax = g.axes.flat
            np.testing.assert_array_equal(
                ax[0].images[0].get_array().data, gaussian(pol, sigma=1.5)
            )
            plt.close()

        # kwargs for a list of map_func for a single 2D image
        # Also, test for any None elements in map_func_kw list
        map_func = [gaussian, ndi.median_filter, gaussian]
        map_func_kw = [{"sigma": 1.5}, {"size": 10}, None]
        with pytest.warns(RuntimeWarning):
            g = isns.ImageGrid(pol, map_func=map_func, map_func_kw=map_func_kw)
            ax = g.axes.flat
            np.testing.assert_array_equal(
                ax[0].images[0].get_array().data, gaussian(pol, sigma=1.5)
            )
            np.testing.assert_array_equal(
                ax[1].images[0].get_array().data, ndi.median_filter(pol, size=10)
            )
            np.testing.assert_array_equal(
                ax[2].images[0].get_array().data, gaussian(pol)
            )
            plt.close()

        # List of 2D images with a list of map_func and list of map_func_kw
        g = isns.ImageGrid([pol, pl], map_func=map_func, map_func_kw=map_func_kw)
        ax = g.axes.flat
        np.testing.assert_array_equal(
            ax[0].images[0].get_array().data, gaussian(pol, sigma=1.5)
        )
        np.testing.assert_array_equal(
            ax[1].images[0].get_array().data, gaussian(pl, sigma=1.5)
        )
        np.testing.assert_array_equal(
            ax[2].images[0].get_array().data, ndi.median_filter(pol, size=10)
        )
        np.testing.assert_array_equal(
            ax[3].images[0].get_array().data, ndi.median_filter(pl, size=10)
        )
        np.testing.assert_array_equal(ax[4].images[0].get_array().data, gaussian(pol))
        np.testing.assert_array_equal(ax[5].images[0].get_array().data, gaussian(pl))
        plt.close()

        # 4D image data with single map_func_kw
        g1 = isns.ImageGrid(self.img_4d, map_func=gaussian, map_func_kw={"sigma": 1.5})
        g_img_4d = gaussian(self.img_4d, sigma=1.5)
        for idx, ax in enumerate(g1.axes.flat):
            if ax.images:
                np.testing.assert_array_equal(
                    ax.images[0].get_array().data, g_img_4d[idx, :, :, :]
                )

        # List of 3D images with a single map_func_kw
        g2 = isns.ImageGrid(
            self.img_mixed_list, map_func=gaussian, map_func_kw={"sigma": 1.5}
        )
        ax = g2.axes.flat
        for idx, val in enumerate(self.img_mixed_list):
            np.testing.assert_array_equal(
                ax[idx].images[0].get_array().data, gaussian(val, sigma=1.5)
            )
        plt.close()

        # `map_func_kw` must be list/tuple of dictionaries if map_func is a list/tuple
        with pytest.raises(TypeError):
            g = isns.ImageGrid([pol, pl], map_func=map_func, map_func_kw={"sigma": 1.5})

        # number of `map_func_kw` passed must be the same as the number of `map_func` objects"
        with pytest.raises(ValueError):
            g = isns.ImageGrid(
                [pol, pl],
                map_func=map_func,
                map_func_kw=[{"sigma": 1.5}, {"sigma": 1.5}],
            )

        # map_func_kw` must be a dictionary when a single `map_func` is passed as input
        with pytest.raises(TypeError):
            g = isns.ImageGrid(
                [pol, pl], map_func=gaussian, map_func_kw=[{"sigma": 1.5}]
            )

    def test_map_func_does_not_alter_original_array(self):
        original_array = copy(self.data_3d)
        # plot with map_func
        _ = isns.ImageGrid(self.data_3d, map_func=gaussian)
        plt.close()
        # re-plot self.data and check if it is still original array
        # and not modified by gaussian filter
        g1 = isns.ImageGrid(self.data_3d)
        for i, ax in enumerate(g1.axes.flat):
            np.testing.assert_array_equal(
                ax.images[0].get_array().data, original_array[:, :, i]
            )
        plt.close()

    def test_param_list_with_map_func(self):
        """
        If the input data and map_func are both list-like,
        modify the parameter list such as dx, units, etc such that
        the length of new parameter list is the same as the number of images.

        # For example -
        # if data -> [img1, img2], map_func -> [func1, func2, func3]
        # and dx = [dx1, dx2] # same as len(data)
        # then for plotting, dx needs to be expanded such that the len(dx) == len(data) * len(map_func)
        # so, new dx -> [dx1, dx2] * len(map_func)
        # and len(dx) == len(nimages)
        """
        # when param is passed as list/tuple and data as well as map_func is a list
        pol = isns.load_image("polymer")
        pl = isns.load_image("fluorescence")
        g = isns.ImageGrid(
            [pol, pl],
            dx=[15, 100],
            units=["nm", "nm"],
            dimension=["si", "si"],
            cbar=[True, True],
            cbar_label=["Height (nm)", "Intensity (au)"],
            cbar_log=[False, True],
            map_func=[gaussian, median, hessian],
        )

        assert len(g.dx) == g._nimages
        assert len(g.units) == g._nimages
        assert len(g.dimension) == g._nimages
        assert len(g.cbar) == g._nimages
        assert len(g.cbar_label) == g._nimages
        assert len(g.cbar_log) == g._nimages

    def test_col_wrap(self):
        g0 = isns.ImageGrid([self.data], col_wrap=3)
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

        # test col_wrap with map_func
        pol = isns.load_image("polymer")
        pl = isns.load_image("fluorescence")
        map_func = [gaussian, ndi.median_filter, hessian]
        map_func_kw = [{"sigma": 1.5}, {"size": 10}, None]

        with pytest.warns(RuntimeWarning):
            g = isns.ImageGrid(pl, map_func=map_func, map_func_kw=map_func_kw)
            assert g.axes.shape == (1, 3)
            plt.close()

        g = isns.ImageGrid([pl, pol], map_func=map_func, map_func_kw=map_func_kw)
        assert g.axes.shape == (2, 3)
        plt.close()

        g = isns.ImageGrid(
            [pl, pol], map_func=map_func, map_func_kw=map_func_kw, col_wrap=2
        )
        assert g.axes.shape == (3, 2)
        plt.close()

        g = isns.ImageGrid([pl, pol], map_func=gaussian)
        assert g.axes.shape == (1, 2)
        plt.close()

    def test_slices(self):
        # along axis=-1
        g = isns.ImageGrid(self.img_3d, slices=[0, 2])
        ax = g.axes.flat
        np.testing.assert_array_equal(
            ax[0].images[0].get_array().data, self.img_3d[:, :, 0]
        )
        np.testing.assert_array_equal(
            ax[1].images[0].get_array().data, self.img_3d[:, :, 2]
        )
        plt.close()

        # along axis=0
        g = isns.ImageGrid(self.img_3d, slices=2, axis=0)
        ax = g.axes.flat
        np.testing.assert_array_equal(
            ax[0].images[0].get_array().data, self.img_3d[2, :, :]
        )
        plt.close()

        # along axis=1
        g = isns.ImageGrid(self.img_3d, slices=2, axis=1)
        ax = g.axes.flat
        np.testing.assert_array_equal(
            ax[0].images[0].get_array().data, self.img_3d[:, 2, :]
        )
        plt.close()

    def test_slices_4d(self):
        # along axis=0
        g = isns.ImageGrid(self.img_4d, slices=[0, 2])
        ax = g.axes.flat
        np.testing.assert_array_equal(
            ax[0].images[0].get_array().data, self.img_4d[0, :, :, :]
        )
        np.testing.assert_array_equal(
            ax[1].images[0].get_array().data, self.img_4d[2, :, :, :]
        )
        plt.close()

        # along axis=1
        g = isns.ImageGrid(self.img_4d, slices=2, axis=1)
        ax = g.axes.flat
        np.testing.assert_array_equal(
            ax[0].images[0].get_array().data, self.img_4d[:, 2, :, :]
        )
        plt.close()

        # along axis=2
        g = isns.ImageGrid(self.img_4d, slices=2, axis=2)
        ax = g.axes.flat
        np.testing.assert_array_equal(
            ax[0].images[0].get_array().data, self.img_4d[:, :, 2, :]
        )
        plt.close()

        # along axis=-1
        g = isns.ImageGrid(self.img_4d, slices=2, axis=-1)
        ax = g.axes.flat
        np.testing.assert_array_equal(
            ax[0].images[0].get_array().data, self.img_4d[:, :, :, 2]
        )
        plt.close()

    def test_axis(self):
        g = isns.ImageGrid(self.img_3d)
        ax = g.axes.flat
        for i in range(self.img_3d.shape[0]):
            np.testing.assert_array_equal(
                ax[i].images[0].get_array().data, self.img_3d[:, :, i]
            )
        plt.close()

        g = isns.ImageGrid(self.img_3d, axis=0)
        ax = g.axes.flat
        for i in range(self.img_3d.shape[0]):
            np.testing.assert_array_equal(
                ax[i].images[0].get_array().data, self.img_3d[i, :, :]
            )
        plt.close()

        g = isns.ImageGrid(self.img_3d, axis=1)
        ax = g.axes.flat
        for i in range(self.img_3d.shape[1]):
            np.testing.assert_array_equal(
                ax[i].images[0].get_array().data, self.img_3d[:, i, :]
            )
        plt.close()

        g = isns.ImageGrid(self.img_3d, axis=2)
        ax = g.axes.flat
        for i in range(self.img_3d.shape[2]):
            np.testing.assert_array_equal(
                ax[i].images[0].get_array().data, self.img_3d[:, :, i]
            )
        plt.close()

        g = isns.ImageGrid(self.img_3d, axis=-1)
        ax = g.axes.flat
        for i in range(self.img_3d.shape[-1]):
            np.testing.assert_array_equal(
                ax[i].images[0].get_array().data, self.img_3d[:, :, i]
            )
        plt.close()

        with pytest.raises(ValueError):
            g = isns.ImageGrid(self.img_3d, axis=3)
            plt.close()

        g = isns.ImageGrid(self.img_4d)
        ax = g.axes.flat
        for i in range(self.img_4d.shape[0]):
            np.testing.assert_array_equal(
                ax[i].images[0].get_array().data, self.img_4d[i, :, :, :]
            )
        plt.close()

        g = isns.ImageGrid(self.img_4d, axis=0)
        ax = g.axes.flat
        for i in range(self.img_4d.shape[0]):
            np.testing.assert_array_equal(
                ax[i].images[0].get_array().data, self.img_4d[i, :, :, :]
            )
        plt.close()

        g = isns.ImageGrid(self.img_4d, axis=1)
        ax = g.axes.flat
        for i in range(self.img_4d.shape[1]):
            np.testing.assert_array_equal(
                ax[i].images[0].get_array().data, self.img_4d[:, i, :, :]
            )
        plt.close()

        g = isns.ImageGrid(self.img_4d, axis=2)
        ax = g.axes.flat
        for i in range(self.img_4d.shape[2]):
            np.testing.assert_array_equal(
                ax[i].images[0].get_array().data, self.img_4d[:, :, i, :]
            )
        plt.close()

        g = isns.ImageGrid(self.img_4d, axis=3)
        ax = g.axes.flat
        for i in range(self.img_4d.shape[3]):
            np.testing.assert_array_equal(
                ax[i].images[0].get_array().data, self.img_4d[:, :, :, i]
            )
        plt.close()

        g = isns.ImageGrid(self.img_4d, axis=-1)
        ax = g.axes.flat
        for i in range(self.img_4d.shape[-1]):
            np.testing.assert_array_equal(
                ax[i].images[0].get_array().data, self.img_4d[:, :, :, i]
            )
        plt.close()

        with pytest.raises(ValueError):
            g = isns.ImageGrid(self.img_4d, axis=4)
            plt.close()

    def test_axis_w_step(self):
        g = isns.ImageGrid(self.img_3d, axis=0, step=2)
        ax = g.axes.flat
        np.testing.assert_array_equal(
            ax[0].images[0].get_array().data, self.img_3d[0, :, :]
        )
        np.testing.assert_array_equal(
            ax[1].images[0].get_array().data, self.img_3d[2, :, :]
        )  # should be the second image
        plt.close()

        g = isns.ImageGrid(self.img_3d, axis=1, step=2)
        ax = g.axes.flat
        np.testing.assert_array_equal(
            ax[0].images[0].get_array().data, self.img_3d[:, 0, :]
        )
        np.testing.assert_array_equal(
            ax[1].images[0].get_array().data, self.img_3d[:, 2, :]
        )  # should be the second image
        plt.close()

        g = isns.ImageGrid(self.img_3d, axis=-1, step=2)
        ax = g.axes.flat
        np.testing.assert_array_equal(
            ax[0].images[0].get_array().data, self.img_3d[:, :, 0]
        )
        np.testing.assert_array_equal(
            ax[1].images[0].get_array().data, self.img_3d[:, :, 2]
        )  # should be the second image
        plt.close()

    def test_axis_w_start_stop(self):
        g = isns.ImageGrid(self.img_3d, axis=-1, start=0, stop=2)
        ax = g.axes.flat
        np.testing.assert_array_equal(
            ax[0].images[0].get_array().data, self.img_3d[:, :, 0]
        )
        np.testing.assert_array_equal(
            ax[1].images[0].get_array().data, self.img_3d[:, :, 1]
        )  # should be the second image
        plt.close()

    def test_cbar_list(self):
        isns.ImageGrid(self.img_list, cmap=["acton", "inferno", "ice"])
        plt.close()

        isns.ImageGrid(self.img_list, cmap=["acton", None, None])
        plt.close()

        isns.ImageGrid(self.img_list, cmap=[None, "inferno", "ice"])
        plt.close()

        with pytest.raises(AssertionError):
            isns.ImageGrid(self.img_3d, cmap=["Reds"])
            plt.close()

    def test_norm_list(self):
        norm_list1 = [
            colors.LogNorm(vmin=1e-4, vmax=1),
            colors.CenteredNorm(),
            colors.PowerNorm(gamma=0.5),
        ]
        isns.ImageGrid(self.img_list, norm=norm_list1)
        plt.close()

        norm_list2 = [colors.LogNorm(vmin=1e-4, vmax=1), None, None]
        isns.ImageGrid(self.img_list, norm=norm_list2)
        plt.close()

        norm_list3 = [None, colors.CenteredNorm(), colors.PowerNorm(gamma=0.5)]
        isns.ImageGrid(self.img_list, norm=norm_list3)
        plt.close()

        with pytest.raises(AssertionError):
            isns.ImageGrid(self.img_3d, norm=[colors.LogNorm(vmin=1e-4, vmax=1)])
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

        isns.ImageGrid([self.data], robust=True)
        plt.close()

        with pytest.raises(AssertionError):
            isns.ImageGrid(self.img_3d, robust=[True, False])
            plt.close()

        with pytest.raises(AssertionError):
            isns.ImageGrid(self.img_3d, perc=[(2, 98), (1, 99)])
            plt.close()

    def test_diverging(self):
        isns.ImageGrid(self.img_list, diverging=True)
        plt.close()

        isns.ImageGrid(self.img_list, diverging=True, vmax=[1, 2, 1])
        plt.close()

        isns.ImageGrid(self.img_list, diverging=[True, False, True], vmax=[1, 2, 1])
        plt.close()

        isns.ImageGrid(self.img_3d, diverging=True)
        plt.close()

        isns.ImageGrid([self.data], diverging=True)
        plt.close()

        with pytest.raises(AssertionError):
            isns.ImageGrid(self.img_3d, diverging=[True, False])
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

        with pytest.raises(AssertionError):
            isns.ImageGrid(
                self.img_3d,
                dx=[1, 3],
                units=["m", "m", "m"],
                dimension=["si", "si", "si"],
            )
            plt.close()

        with pytest.raises(AssertionError):
            isns.ImageGrid(
                self.img_3d,
                dx=[1, 2, 3],
                units=["m", "m"],
                dimension=["si", "si"],
            )
            plt.close()

        with pytest.raises(AssertionError):
            isns.ImageGrid(
                self.img_3d,
                dx=[1, 2, 3],
                units=["m", "m", "m"],
                dimension=["si", "si"],
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
            cbar_log=[True, True, False],
        )
        plt.close()

        isns.ImageGrid(
            self.img_list,
            cbar_log=True,
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

        with pytest.raises(AssertionError):
            isns.ImageGrid(self.img_3d, cbar=[True, False])
            plt.close()

        with pytest.raises(AssertionError):
            isns.ImageGrid(self.img_3d, cbar_label=["A", "B"])
            plt.close()

        with pytest.raises(AssertionError):
            isns.ImageGrid(self.img_3d, cbar_log=[True, False])
            plt.close()

    def test_figure_size(self):
        g0 = isns.ImageGrid([self.data])
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
    
    def test_auto_aspect(self):
        imgsize0 = (10, 10)
        g0 = isns.ImageGrid([np.zeros(imgsize0) for i in range(10)], aspect='auto')
        assert np.isclose(imgsize0[1]/imgsize0[0], g0.aspect)
        plt.close()

        imgsize1 = (10, 5)
        g1 = isns.ImageGrid([np.zeros(imgsize1) for i in range(10)], aspect='auto')
        assert np.isclose(imgsize1[1]/imgsize1[0], g1.aspect)
        plt.close()
        
        imgsize2 = (5, 10)
        g2 = isns.ImageGrid([np.zeros(imgsize2) for i in range(10)], aspect='auto')
        assert np.isclose(imgsize2[1]/imgsize2[0], g2.aspect)
        plt.close()

        imglist = [np.zeros(imgsize0) for i in range(4)] + [np.zeros(imgsize1) for i in range(4)] + [np.zeros(imgsize2) for i in range(4)]
        g3 = isns.ImageGrid(imglist, aspect='auto')
        assert np.isclose(min([imgsize0[1]/imgsize0[0], imgsize1[1]/imgsize1[0], imgsize2[1]/imgsize2[0]]), g3.aspect)
        plt.close()
        
    def test_vmin_vmax(self):
        g = isns.ImageGrid(cells, vmin=0.5, vmax=0.75)
        for ax in g.axes.ravel():
            assert ax.images[0].colorbar.vmin == 0.5
            assert ax.images[0].colorbar.vmax == 0.75
        plt.close()

        g = isns.ImageGrid(astronaut(), vmin=[10, 20, 30], vmax=[200, 200, 200])
        ax = g.axes.ravel()

        assert ax[0].images[0].colorbar.vmin == 10
        assert ax[0].images[0].colorbar.vmax == 200

        assert ax[1].images[0].colorbar.vmin == 20
        assert ax[1].images[0].colorbar.vmax == 200

        assert ax[2].images[0].colorbar.vmin == 30
        assert ax[2].images[0].colorbar.vmax == 200
        plt.close()

        # when vmin/vmax provided as a list of floats,
        # length must be equal to the number of images
        with pytest.raises(AssertionError):
            _ = isns.ImageGrid(cells, vmin=[12, 23])
            plt.close()

        with pytest.raises(AssertionError):
            _ = isns.ImageGrid(cells, vmax=[12, 23])
            plt.close()


@pytest.mark.parametrize(
    "img",
    [
        np.random.random(2500).reshape((50, 50)),
        np.random.random(50 * 50 * 4).reshape((50, 50, 4)),
    ],
)
def test_rgbplot_data(img):
    with pytest.raises(ValueError):
        isns.rgbplot(img)
        plt.close()


def test_rgbplot_cmap():
    g = isns.rgbplot(astronaut())
    assert g.cmap == ["R", "G", "B"]
    plt.close()

    g = isns.rgbplot(astronaut(), cmap=["inferno", "viridis", "ice"])
    assert g.cmap == ["inferno", "viridis", "ice"]
    plt.close()


def test_rgbplot_vmin_vmax():
    g = isns.rgbplot(astronaut(), vmin=10, vmax=200)
    for ax in g.axes.ravel():
        assert ax.images[0].colorbar.vmin == 10
        assert ax.images[0].colorbar.vmax == 200
    plt.close()


class TestParamGrid(object):
    data = np.random.random(2500).reshape((50, 50))

    def test_none_data(self):
        with pytest.raises(ValueError):
            isns.ParamGrid(None, "sobel")
            plt.close()

    def test_none_filt(self):
        with pytest.raises(ValueError):
            isns.ParamGrid(self.data, None)
            plt.close()

    def test_self_data(self):
        g = isns.ParamGrid(self.data, "sobel")
        np.testing.assert_array_equal(self.data, g.data)
        plt.close()

    def test_self_fig(self):
        g = isns.ParamGrid(self.data, "sobel")
        assert isinstance(g.fig, Figure)
        plt.close()

    def test_rows(self):
        with pytest.raises(TypeError):
            _ = isns.ParamGrid(self.data, "gaussian", row=gaussian, sigma=[1, 2, 3])
            plt.close()

        with pytest.raises(ValueError):
            _ = isns.ParamGrid(self.data, "gaussian", row="sigma")
            plt.close()

    def test_cols(self):
        with pytest.raises(TypeError):
            _ = isns.ParamGrid(self.data, "gaussian", col=gaussian, sigma=[1, 2, 3])
            plt.close()

        with pytest.raises(ValueError):
            _ = isns.ParamGrid(self.data, "gaussian", col="sigma")
            plt.close()

    def test_self_axes(self):
        g0 = isns.ParamGrid(self.data, "sobel")
        for ax in g0.axes.flat:
            assert isinstance(ax, Axes)

        g1 = isns.ParamGrid(self.data, "gaussian", row="sigma", sigma=[1, 2, 3])
        for ax in g1.axes.flat:
            assert isinstance(ax, Axes)

        g2 = isns.ParamGrid(
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
        g0 = isns.ParamGrid(self.data, "sobel")
        assert g0.axes.shape == (1, 1)

        g1 = isns.ParamGrid(self.data, "gaussian", row="sigma", sigma=[1, 2, 3])
        assert g1.axes.shape == (3, 1)

        g2 = isns.ParamGrid(self.data, "gaussian", col="sigma", sigma=[1, 2, 3])
        assert g2.axes.shape == (1, 3)

        g3 = isns.ParamGrid(
            self.data,
            "gaussian",
            row="sigma",
            col="mode",
            sigma=[1, 2, 3],
            mode=["reflect", "nearest"],
        )
        assert g3.axes.shape == (3, 2)

        g4 = isns.ParamGrid(
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
        g0 = isns.ParamGrid(
            self.data, "gaussian", col="sigma", sigma=[1, 2, 3, 4, 5], col_wrap=3
        )
        assert g0.axes.shape == (2, 3)
        plt.close()

        with pytest.raises(ValueError):
            isns.ParamGrid(
                self.data, "gaussian", row="sigma", sigma=[1, 2, 3, 4, 5], col_wrap=3
            )
            plt.close()

        with pytest.raises(ValueError):
            isns.ParamGrid(
                self.data,
                "gaussian",
                row="mode",
                col="sigma",
                col_wrap=3,
                sigma=[1, 2, 3, 4, 5],
                mode=["reflect", "nearest"],
            )
            plt.close()

    def test_additional_kwargs_for_filters(self):
        isns.ParamGrid(
            self.data, "gaussian", row="sigma", sigma=[1, 2, 3], mode="reflect"
        )
        plt.close()

        isns.ParamGrid(
            self.data, "gaussian", col="sigma", sigma=[1, 2, 3], mode="reflect"
        )
        plt.close()

        isns.ParamGrid(
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
        g0 = isns.ParamGrid(self.data, "sobel")
        np.testing.assert_array_equal(g0.fig.get_size_inches(), (3, 3))

        g1 = isns.ParamGrid(self.data, "gaussian", row="sigma", sigma=[1, 2, 3])
        np.testing.assert_array_equal(g1.fig.get_size_inches(), (3, 9))

        g2 = isns.ParamGrid(
            self.data,
            "gaussian",
            row="sigma",
            col="mode",
            sigma=[1, 2, 3],
            mode=["reflect", "nearest"],
        )
        np.testing.assert_array_equal(g2.fig.get_size_inches(), (6, 9))

        g3 = isns.ParamGrid(
            self.data,
            "gaussian",
            row="sigma",
            col="mode",
            sigma=[1, 2, 3],
            mode=["reflect", "nearest"],
            height=2,
        )
        np.testing.assert_array_equal(g3.fig.get_size_inches(), (4, 6))

        g4 = isns.ParamGrid(
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

    def test_vmin_vmax(self):
        g = isns.ParamGrid(
            self.data,
            "gaussian",
            row="sigma",
            sigma=[1, 2, 3],
            mode="reflect",
            vmin=0,
            vmax=2,
        )
        for ax in g.axes.ravel():
            assert ax.images[0].colorbar.vmin == 0
            assert ax.images[0].colorbar.vmax == 2
        plt.close()


@pytest.mark.parametrize("shape", [(20, 20), (20, 40), (40, 20)])
@pytest.mark.parametrize("gap", [0, 0.15])
@pytest.mark.parametrize("count,wrap", [(6, 3), (5, 3), (3, 1), (1, 1)])
def test_image_grid_gap(shape, gap, count, wrap):
    g = isns.ImageGrid(
        [np.ones(shape)] * count,
        cbar=False,
        gap=gap,
        col_wrap=wrap,
    )
    try:
        g.fig.canvas.draw()
        boxes = [ax.get_window_extent() for ax in g.axes.flat][:count]
        for i, box in enumerate(boxes):
            assert box.width / box.height == pytest.approx(shape[1] / shape[0])
            if i % wrap:
                assert box.x0 - boxes[i - 1].x1 == pytest.approx(
                    gap * g.fig.dpi,
                    abs=1e-8,
                )
            if i >= wrap:
                assert boxes[i - wrap].y0 - box.y1 == pytest.approx(
                    gap * g.fig.dpi,
                    abs=1e-8,
                )
        assert boxes[0].x0 == pytest.approx(0, abs=1e-8)
        assert boxes[0].y1 == pytest.approx(g.fig.bbox.height)
    finally:
        plt.close(g.fig)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"extent": (0, 80, 0, 20)},
        {"map_func": lambda image: image[:, :10]},
    ],
)
def test_image_grid_gap_uses_plotted_geometry(kwargs):
    g = isns.ImageGrid(
        [np.ones((20, 40))] * 4,
        cbar=False,
        gap=0,
        col_wrap=2,
        **kwargs,
    )
    try:
        g.fig.canvas.draw()
        a, b, c, _ = [ax.get_window_extent() for ax in g.axes.flat]
        assert b.x0 - a.x1 == pytest.approx(0, abs=1e-8)
        assert a.y0 - c.y1 == pytest.approx(0, abs=1e-8)
    finally:
        plt.close(g.fig)


@pytest.mark.parametrize("gap", [-1, np.nan, np.inf, "auto", [0], 1j])
def test_image_grid_invalid_gap(gap):
    with pytest.raises(ValueError, match="gap must be"):
        isns.ImageGrid([np.ones((20, 40))], cbar=False, gap=gap)


@pytest.mark.parametrize("kwargs", [{}, {"cbar": False, "showticks": True}])
def test_image_grid_gap_rejects_decorations(kwargs):
    with pytest.raises(ValueError, match="gap requires cbar=False"):
        isns.ImageGrid([np.ones((20, 40))], gap=0, **kwargs)


def test_image_grid_gap_rejects_mixed_proportions():
    figures = plt.get_fignums()
    with pytest.raises(ValueError, match="matching displayed image proportions"):
        isns.ImageGrid([np.ones((20, 40)), np.ones((40, 20))], cbar=False, gap=0)
    assert plt.get_fignums() == figures


@pytest.mark.parametrize(
    "auto,constrained", [(True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("gap", [0, 0.15])
def test_image_grid_gap_overrides_layout_defaults(auto, constrained, gap):
    settings = {
        "figure.autolayout": auto,
        "figure.constrained_layout.use": constrained,
    }
    with matplotlib.rc_context(settings), warnings.catch_warnings():
        warnings.simplefilter("error")
        g = isns.ImageGrid(
            [np.ones((20, 40))] * 5,
            height=1,
            col_wrap=3,
            cbar=False,
            gap=gap,
        )
        try:
            assert g.fig.get_layout_engine() is None
            np.testing.assert_allclose(g.fig.get_size_inches(), (6 + 2 * gap, 2 + gap))
            for _ in range(2):
                g.fig.canvas.draw()
                a, b, c, d, e = [ax.get_window_extent() for ax in g.axes.flat][:5]
                assert b.x0 - a.x1 == pytest.approx(gap * g.fig.dpi, abs=1e-8)
                assert a.y0 - d.y1 == pytest.approx(gap * g.fig.dpi, abs=1e-8)
                assert a.x0 == pytest.approx(0, abs=1e-8)
                assert e.y0 == pytest.approx(0, abs=1e-8)
                assert c.x1 == pytest.approx(g.fig.bbox.width)
                assert a.y1 == pytest.approx(g.fig.bbox.height)
            for key, value in settings.items():
                assert matplotlib.rcParams[key] == value
        finally:
            plt.close(g.fig)


@pytest.mark.parametrize(
    "auto,constrained", [(True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("kwargs", [{}, {"gap": None}])
def test_image_grid_default_retains_layout(auto, constrained, kwargs, monkeypatch):
    engines = []
    tight_layout = Figure.tight_layout

    def record_layout(fig, *args, **kw):
        engines.append(type(fig.get_layout_engine()).__name__)
        return tight_layout(fig, *args, **kw)

    monkeypatch.setattr(Figure, "tight_layout", record_layout)
    with matplotlib.rc_context(
        {
            "figure.autolayout": auto,
            "figure.constrained_layout.use": constrained,
        }
    ):
        g = isns.ImageGrid([np.ones((20, 40))] * 2, cbar=False, **kwargs)
        plt.close(g.fig)
    assert engines == ["TightLayoutEngine" if auto else "ConstrainedLayoutEngine"]


@pytest.mark.parametrize("gap", [None, 0, 0.1])
@pytest.mark.parametrize("orientation", ["v", "h"])
def test_shared_colorbar_preserves_image_geometry(gap, orientation):
    data = [np.arange(800).reshape(20, 40) + i * 100 for i in range(5)]
    plain = isns.ImageGrid(data, cbar=False, gap=gap, height=1, col_wrap=3)
    g = isns.ImageGrid(
        data,
        cbar="shared",
        gap=gap,
        height=1,
        col_wrap=3,
        orientation=orientation,
        cbar_label="Intensity",
        cbar_ticks=[0, 600, 1200],
    )
    try:
        plain.fig.canvas.draw()
        for _ in range(2):
            g.fig.canvas.draw()
            boxes = np.array([ax.get_window_extent().bounds for ax in g.axes.flat])
            original = np.array(
                [ax.get_window_extent().bounds for ax in plain.axes.flat]
            )
            # All axes receive the same translation; no sizes or gaps change.
            np.testing.assert_allclose(boxes[:, 2:], original[:, 2:], atol=1e-8)
            np.testing.assert_allclose(
                boxes[:, :2] - boxes[0, :2],
                original[:, :2] - original[0, :2],
                atol=1e-8,
            )
        images = [ax.images[0] for ax in list(g.axes.flat)[:5]]
        assert all(im.norm is images[0].norm for im in images)
        assert all(im.get_cmap() is images[0].get_cmap() for im in images)
        assert images[0].get_clim() == (0, 1199)
        np.testing.assert_allclose(images[0].to_rgba(500), images[-1].to_rgba(500))
        assert len(g.fig.axes) == g.axes.size + 1
        assert g.colorbar.ax is g.cbar_ax
        np.testing.assert_array_equal(g.colorbar.get_ticks(), [0, 600, 1200])
        box = g.cbar_ax.get_tightbbox(g.fig.canvas.get_renderer())
        assert box.x0 >= -1e-8 and box.y0 >= -1e-8
        assert box.x1 <= g.fig.bbox.width + 1e-8
        assert box.y1 <= g.fig.bbox.height + 1e-8
    finally:
        plt.close(plain.fig)
        plt.close(g.fig)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, (1, 100)),
        ({"vmin": 0, "vmax": 200}, (0, 200)),
        ({"vmin": 0}, (0, 100)),
        ({"robust": True, "perc": (25, 75)}, (1, 25.75)),
        ({"diverging": True}, (-100, 100)),
        ({"diverging": True, "vmax": 50}, (-50, 50)),
        ({"cbar_log": True}, (1, 100)),
        ({"norm": colors.Normalize(vmin=0)}, (0, 100)),
    ],
)
def test_shared_colorbar_normalization(kwargs, expected):
    g = isns.ImageGrid([np.ones((1, 3)), np.array([[100.0]])], cbar="shared", **kwargs)
    try:
        image = g.axes.flat[0].images[0]
        assert image.get_clim() == expected
        if "norm" in kwargs:
            assert image.norm is kwargs["norm"]
        if kwargs.get("cbar_log"):
            assert isinstance(image.norm, colors.LogNorm)
        if kwargs.get("robust"):
            assert g.colorbar.extend == "both"
    finally:
        plt.close(g.fig)


def test_shared_colorbar_selected_transformed_data():
    data = np.stack([np.full((4, 4), n) for n in (1, 2, 999)], axis=-1)
    g = isns.ImageGrid(data, slices=[0, 1], map_func=lambda x: x * 10, cbar="shared")
    try:
        assert g.colorbar.mappable.get_clim() == (10, 20)
    finally:
        plt.close(g.fig)


def test_shared_colorbar_ignores_masked_and_nonfinite_values():
    data = [
        np.ma.array([[1, 999]], mask=[[False, True]]),
        np.array([[2, np.nan, np.inf]]),
    ]
    g = isns.ImageGrid(data, cbar="shared")
    try:
        assert g.colorbar.mappable.get_clim() == (1, 2)
    finally:
        plt.close(g.fig)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"cmap": ["gray", "viridis"]}, "single cmap"),
        ({"vmin": [0, 1]}, "single vmin"),
        ({"norm": colors.Normalize(), "vmin": 0}, "shared norm"),
        ({"norm": colors.Normalize(), "robust": True}, "shared norm"),
        ({"perc": [(2, 98), (2, 98)]}, "percentile pair"),
        ({"orientation": "bad"}, "orientation"),
    ],
)
def test_shared_colorbar_rejects_conflicting_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        isns.ImageGrid([np.ones((4, 4))] * 2, cbar="shared", **kwargs)


@pytest.mark.parametrize(
    "data,match",
    [
        ([np.ones((4, 4, 3))], "scalar images"),
        ([np.ones((4, 4, 4))], "scalar images"),
        ([np.full((4, 4), np.nan)], "finite, unmasked"),
    ],
)
def test_shared_colorbar_rejects_unsupported_images(data, match):
    figures = plt.get_fignums()
    with pytest.raises(ValueError, match=match):
        isns.ImageGrid(data, cbar="shared")
    assert plt.get_fignums() == figures


@pytest.mark.parametrize("gap", [None, 0])
@pytest.mark.parametrize(
    "setting", ["figure.autolayout", "figure.constrained_layout.use"]
)
def test_shared_colorbar_with_layout_defaults(gap, setting):
    with matplotlib.rc_context({setting: True}):
        g = isns.ImageGrid(
            [np.arange(8).reshape(2, 4)] * 4, gap=gap, cbar="shared", col_wrap=2
        )
        try:
            for _ in range(2):
                g.fig.canvas.draw()
                a, b, c, _ = [ax.get_window_extent() for ax in g.axes.flat]
                if gap == 0:
                    assert b.x0 - a.x1 == pytest.approx(0, abs=1e-8)
                    assert a.y0 - c.y1 == pytest.approx(0, abs=1e-8)
                assert g.cbar_ax.get_window_extent().x0 > b.x1
        finally:
            plt.close(g.fig)


def test_shared_colorbar_invalid_log_data_closes_figure():
    figures = plt.get_fignums()
    with pytest.raises(ValueError, match="positive limits and data"):
        isns.ImageGrid([np.full((4, 4), -1)], cbar="shared", cbar_log=True)
    assert plt.get_fignums() == figures


def test_FilterGrid_deprecation_warning():
    with pytest.warns(UserWarning, match="FilterGrid is depracted"):
        _ = isns.FilterGrid(
            np.random.random(2500).reshape((50, 50)),
            "median",
            col="size",
            size=[2, 3, 4, 5],
        )
        plt.close()
