=======================================
seaborn-image: image data visualization
=======================================

|img1| |img2| |img3|

.. |img1| image:: /auto_examples/images/thumb/sphx_glr_plot_image_hist_thumb.png
    :width: 30%

.. |img2| image:: /auto_examples/images/thumb/sphx_glr_plot_filtergrid_thumb.png
    :width: 30%

.. |img3| image:: /auto_examples/images/thumb/sphx_glr_plot_image_robust_thumb.png
    :width: 30%


Description
===========

Seaborn-image is a *seaborn like* Python **image** visualization and processing library
based on matplotlib, scikit-image and scipy.

The aim of seaborn-image is to provide a high-level API to **process and plot attractive images quickly**
**and effectively**.

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

Check out the :doc:`quickstart page <quickstart>` for a walk through
of the sample features below.

Simple usage
************

.. code-block:: python

    import seaborn_image as isns

    """Set context like seaborn"""
    isns.set_context("notebook")

    """Plot publishable quality image in one line"""
    isns.imgplot(data)

    """Add a scalebar"""
    isns.imgplot(data, dx=1, units="um")

Apply image filters (from scipy and skimage) and plot
*****************************************************

.. code-block:: python

    import seaborn_image as isns

    isns.filterplot(data, filter="gaussian")


For more information on getting strated, refer to the :doc:`quickstart guide <quickstart>`.


Contents
========

.. toctree::
   :maxdepth: 1

   Quickstart <quickstart>
   How-to? <how_to>
   Gallery <auto_examples/index>
   License <license>
   Authors <authors>
   Changelog <https://github.com/SarthakJariwala/seaborn-image/releases>
   Reference <reference>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
