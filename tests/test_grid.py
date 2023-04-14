import pytest

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

matplotlib.use("AGG")  # use non-interactive backend for tests

cells = isns.load_image("cells")


class TestImageGrid:

    img_3d = np.random.random(4 * 4 * 4).reshape((4, 4, 4))
    img_4d = np.random.random(4 * 4 * 4 * 3).reshape((4, 4, 4, 3))

    data = np.random.random(2500).reshape((50, 50))
    img_list = [data, data, data]

    data_3d = np.random.random(2500*3).reshape((50, 50, 3))
    img_3d_list = [data_3d, data_3d, data_3d]

    img_mixed_list = [data_3d, data, data_3d]

    data_3d_bad = np.random.random(2500*6).reshape((50, 50, 6))
    img_bad_list1 = [data, data_3d_bad, data_3d]
    img_bad_list2 = [data, img_4d, data_3d]

    def test_none_data(self):
        with pytest.raises(ValueError):
            isns.ImageGrid(None)

    def test_higher_dim_data(self):
        with pytest.raises(ValueError):
            isns.ImageGrid(np.random.random(50 * 50 * 4 * 3 * 3).reshape((50, 50, 4, 3, 3)))
    
    def test_incorrect_channels(self):
        with pytest.raises(ValueError):
            isns.ImageGrid(np.random.random(6 * 50 * 50 * 5).reshape((6, 50, 50, 5)))

    def test_incorrect_axis_for_slicing(self):
        with pytest.raises(ValueError):
            isns.ImageGrid(self.img_3d, axis=3)

    def test_self_data(self):
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
        
        with pytest.raises(ValueError):
            g = isns.ImageGrid(self.img_bad_list2)

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

        g0 = isns.ImageGrid(self.data)
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

        # List of 2D images with single map_func
        pol = isns.load_image("polymer")
        pl = isns.load_image("fluorescence")
        new_img_list = [pol, pl]
        g1 = isns.ImageGrid(new_img_list, map_func=gaussian)
        ax = g1.axes.flat
        np.testing.assert_array_equal(ax[0].images[0].get_array().data, gaussian(pol))
        np.testing.assert_array_equal(ax[1].images[0].get_array().data, gaussian(pl))
        plt.close()

        # Single 2D image with a single map_func
        g = isns.ImageGrid(pol, map_func=gaussian)
        ax = g.axes.flat
        np.testing.assert_array_equal(ax[0].images[0].get_array().data, gaussian(pol))
        plt.close()

        # Single 2D image with a list of map_func
        g = isns.ImageGrid(pol, map_func=[gaussian, gaussian])
        ax = g.axes.flat
        np.testing.assert_array_equal(ax[0].images[0].get_array().data, gaussian(pol))
        np.testing.assert_array_equal(ax[1].images[0].get_array().data, gaussian(pol))
        plt.close()

        # List of 2D images with a list of map_func
        g = isns.ImageGrid([pol, pl], map_func=[gaussian, gaussian])
        ax = g.axes.flat
        np.testing.assert_array_equal(ax[0].images[0].get_array().data, gaussian(pol))
        np.testing.assert_array_equal(ax[1].images[0].get_array().data, gaussian(pl))
        np.testing.assert_array_equal(ax[2].images[0].get_array().data, gaussian(pol))
        np.testing.assert_array_equal(ax[3].images[0].get_array().data, gaussian(pl))
        plt.close()

        # 3 Image with list of map_func
        with pytest.raises(ValueError):
            g = isns.ImageGrid(self.img_3d, map_func=[gaussian, gaussian])

        # List of map_func must all be callables
        with pytest.raises(TypeError):
            g = isns.ImageGrid([pol, pl], map_func=[gaussian, "hessian"])

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
        g = isns.ImageGrid(pol, map_func=map_func, map_func_kw=map_func_kw)
        ax = g.axes.flat
        np.testing.assert_array_equal(
            ax[0].images[0].get_array().data, gaussian(pol, sigma=1.5)
        )
        np.testing.assert_array_equal(
            ax[1].images[0].get_array().data, ndi.median_filter(pol, size=10)
        )
        np.testing.assert_array_equal(ax[2].images[0].get_array().data, gaussian(pol))
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

        # test col_wrap with map_func
        pol = isns.load_image("polymer")
        pl = isns.load_image("fluorescence")
        map_func = [gaussian, ndi.median_filter, hessian]
        map_func_kw = [{"sigma": 1.5}, {"size": 10}, None]

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
        g = isns.ImageGrid(self.img_3d, axis=0)
        ax = g.axes.flat
        for i in range(self.img_3d.shape[0]):
            np.testing.assert_array_equal(
                ax[i].images[0].get_array().data, self.img_3d[i, :, :]
            )
        plt.close()
        
        g = isns.ImageGrid(self.img_3d, axis=-1)
        ax = g.axes.flat
        for i in range(self.img_3d.shape[-1]):
            np.testing.assert_array_equal(
                ax[i].images[0].get_array().data, self.img_3d[:, :, i]
            )
        plt.close()

        g = isns.ImageGrid(self.img_4d, axis=0)
        ax = g.axes.flat
        for i in range(self.img_4d.shape[0]):
            np.testing.assert_array_equal(
                ax[i].images[0].get_array().data, self.img_4d[i, :, :, :]
            )
        plt.close()

        with pytest.raises(ValueError):
            g = isns.ImageGrid(self.img_3d, axis=3)
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
        
        norm_list1 = [colors.LogNorm(vmin=1e-4, vmax=1), colors.CenteredNorm(), colors.PowerNorm(gamma=0.5)]
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

        isns.ImageGrid(self.data, robust=True)
        plt.close()

        with pytest.raises(AssertionError):
            isns.ImageGrid(self.img_3d, robust=[True, False])
            plt.close()

        with pytest.raises(AssertionError):
            isns.ImageGrid(self.img_3d, perc=[(2, 98), (1, 99)])
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

        with pytest.raises(AssertionError):
            _ = isns.ImageGrid(cells, vmax=[12, 23])


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

    def test_none_filt(self):
        with pytest.raises(ValueError):
            isns.ParamGrid(self.data, None)

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

        with pytest.raises(ValueError):
            _ = isns.ParamGrid(self.data, "gaussian", row="sigma")

    def test_cols(self):
        with pytest.raises(TypeError):
            _ = isns.ParamGrid(self.data, "gaussian", col=gaussian, sigma=[1, 2, 3])

        with pytest.raises(ValueError):
            _ = isns.ParamGrid(self.data, "gaussian", col="sigma")

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


def test_FilterGrid_deprecation_warning():
    with pytest.warns(UserWarning, match="FilterGrid is depracted"):
        _ = isns.FilterGrid(
            np.random.random(2500).reshape((50, 50)),
            "median",
            col="size",
            size=[2, 3, 4, 5],
        )
        plt.close()
