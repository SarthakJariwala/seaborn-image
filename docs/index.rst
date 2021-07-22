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

.. code-block:: bash

    pip install -U seaborn-image

Quick Usage
===========

Visualize 2-D images
********************

.. code-block:: python

    import seaborn_image as isns

    """Global settings for images"""
    isns.set_context("notebook")
    isns.set_image(cmap="deep", despine=True)
    isns.set_scalebar(color="red")

    pol = isns.load_image("polymer")

    """Image with a scalebar"""
    ax = isns.imgplot(pol, dx=0.01, units="um")

    """Get basic image stats"""
    isns.imgplot(data, describe=True)

Visualize image distribution
****************************

.. code-block:: python

    f = isns.imghist(pol)

Multi-dimensional images
************************

.. code-block:: python

    cells = isns.load_image("cells")

    g = isns.ImageGrid(cells, slices=[10, 20, 30, 40])

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
