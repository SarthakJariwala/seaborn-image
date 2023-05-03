=======================================
seaborn-image: image data visualization
=======================================

|img1| |img2| |img3| |img4|

.. |img1| image:: /auto_examples/images/thumb/sphx_glr_plot_image_hist_thumb.png
    :width: 190px
    :height: 180px

.. |img2| image:: /auto_examples/images/thumb/sphx_glr_plot_filter_thumb.png
    :width: 140px
    :height: 140px

.. |img3| image:: /auto_examples/images/thumb/sphx_glr_plot_fft_thumb.png
    :width: 120px
    :height: 120px

.. |img4| image:: /auto_examples/images/thumb/sphx_glr_plot_paramgrid_thumb.png
    :width: 140px
    :height: 140px

.. .. |img5| image:: /auto_examples/images/thumb/sphx_glr_plot_image_robust_thumb.png
..     :width: 200px
..     :height: 200px


Description
===========

Seaborn-image is a Python **image** visualization library based on matplotlib
and provides a high-level API to **draw attractive and informative images quickly**
**and effectively**.

It is heavily inspired by `seaborn <https://seaborn.pydata.org/>`_, a high-level visualization library
for drawing attractive statistical graphics in Python.

To view example images, check out the :doc:`gallery page <auto_examples/index>` and :doc:`reference <reference>`.
For specific how-to questions, refer to the :doc:`tutorial page <tutorial>`.

Check out the source code on `github <https://github.com/SarthakJariwala/seaborn-image>`_.
If you come across any bugs/issues, please open an `issue <https://github.com/SarthakJariwala/seaborn-image/issues>`_.


Installation
============

Using `pip`

.. code-block:: bash

    pip install -U seaborn-image

Using `conda`

.. code-block:: bash

    conda install -c conda-forge seaborn-image


Getting Started
===============

First, let's import the library and make some changes to the visualization settings.

.. code-block:: python

    import seaborn_image as isns

    # this will create thicker lines and larger fonts than usual
    isns.set_context("notebook")

    # change image related settings
    isns.set_image(cmap="deep", despine=True)  # set the colormap and despine the axes
    isns.set_scalebar(color="red")  # change scalebar color

.. note::

    This is only a quick look at the settings, see :doc:`reference <reference>` for more details.
    You can also simply use the default settings that come with `seaborn_image`.


Visualization 2-D images
************************

A quick way of attractive and descriptive visualization of 2D image data using `imgplot`.

.. code-block:: python

    # example 2D image data
    pol = isns.load_image("polymer")

    # image with a scalebar
    ax = isns.imgplot(pol, dx=0.01, units="um")

In the above example, the image is plotted with a scalebar of length 0.01 um or 10 nm.
The `dx` parameter specifies the physical size of the pixel and the `units` parameter specifies the units of the scalebar.


You can also pass `describe=True` to `imgplot` to get a summary of the image data along with the visualization.

.. code-block:: python

    # get basic image stats along with the visualization
    isns.imgplot(pol, describe=True)


Visualize image distribution
****************************

Sometimes you may want to visualize the distribution of an image. For that, you can use `imghist`.

.. code-block:: python

    f = isns.imghist(pol, dx=0.01, units="um")

.. note::

    There are no changes in the parameters specified in `imghist` compared to `imgplot`.
    For more details on specific parameters, please see :doc:`reference <reference>`.

Multi-dimensional images
************************

Image data is not always 2D and for those image data there is `ImageGrid`.

.. code-block:: python

    # example 3D image data
    cells = isns.load_image("cells")

    g = isns.ImageGrid(cells)


You can also specify the specific `slices` of the 3D data that you want to visualize.
You can also specify the `axis` along which you want to `slice` your 3D image data for visualization.

.. code-block:: python

    g = isns.ImageGrid(cells, slices=[10, 20, 30, 40], axis=1)

You can also plot a collection of 3D image data.

.. code-block:: python

    from skimage.data import astronaut, chelsea

    g = isns.ImageGrid([astronaut(), chelsea()], origin="upper")



This was a very short intro to `seaborn_image`. There are many other functions and options available in `seaborn_image`.
For more information check out examples in :doc:`tutorial <tutorial>`, :doc:`api <reference>` and :doc:`gallery <auto_examples/index>`.


Contents
========

.. toctree::
   :maxdepth: 1

   Gallery <auto_examples/index>
   API Reference <reference>
   Tutorial <tutorial>
   License <license>
   Changelog <https://github.com/SarthakJariwala/seaborn-image/releases>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
