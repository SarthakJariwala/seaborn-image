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

.. |img4| image:: /auto_examples/images/thumb/sphx_glr_plot_filtergrid_thumb.png
    :width: 140px
    :height: 140px

.. .. |img5| image:: /auto_examples/images/thumb/sphx_glr_plot_image_robust_thumb.png
..     :width: 200px
..     :height: 200px


Description
===========

Seaborn-image is a Python **image** visualization library based on matplotlib, scikit-image and scipy.

Seaborn-image provides a high-level API to **draw attractive and informative images quickly**
**and effectively**.

It is heavily inspired by `**seaborn** <https://seaborn.pydata.org/>`_, a high-level visualization library
for drawing attractive statistical graphics in Python.

To get started with ``seaborn_image``, check out the :doc:`quickstart page <quickstart>`.
To view example images, check out the :doc:`gallery page <auto_examples/index>`.
For specific how-to questions, refer to the :doc:`how-to page <how_to>`.

Check out the source code on `github <https://github.com/SarthakJariwala/seaborn-image>`_.
If you come across any bugs, please open an `issue <https://github.com/SarthakJariwala/seaborn-image/issues>`_.


Installation
============

.. code-block:: bash

    pip install seaborn-image

Usage
=====

Simple usage
************

.. code-block:: python

    import seaborn_image as isns

    """Global settings for images"""
    isns.set_context("notebook")
    isns.set_image(cmap="deep", despine=True)
    isns.set_scalebar(color="red")

    """Plot image"""
    isns.imgplot(data)

    """Image with a scalebar"""
    isns.imgplot(data, dx=1, units="um")

    """Get basic image stats"""
    isns.imgplot(data, describe=True)

Visualize image distribution
****************************

..code-block:: python

    isns.imghist(data)

Filter image and plot
*********************

.. code-block:: python

    isns.filterplot(data, filt="gaussian", sigma=2.5)

For more information check out examples in :doc:`api <reference>` and :doc:`gallery <auto_examples/index>`.


Contents
========

.. toctree::
   :maxdepth: 1

   Gallery <auto_examples/index>
   Reference <reference>
   How-to? <how_to>
   License <license>
   Authors <authors>
   Changelog <https://github.com/SarthakJariwala/seaborn-image/releases>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
