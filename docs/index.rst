=======================================
seaborn-image: image data visualization
=======================================

.. list-table::

    * - .. image:: ../examples/image_0.png
            :width: 150px
            :height: 140px

        .. image:: ../examples/image_1.png
            :width: 150px
            :height: 140px

        .. image:: ../examples/image_3.png
            :width: 320px
            :height: 150px


Seaborn like image data visualization using matplotlib, scikit-image and scipy.


Description
===========

Seaborn-image is a seaborn like python **image** visualization and processing library
based on matplotlib.

The aim of seaborn-image is to provide a high-level API to **process and plot attractive images quickly**
**and effectively**.


Installation
============

``pip install seaborn-image``

Simple usage

.. code-block:: python

    import seaborn_image as isns

    """Set context like seaborn"""
    isns.set_context("notebook")

    """Plot publishable quality image in one line"""
    isns.imgplot(data)

    """Add a scalebar"""
    isns.imgplot(data, dx=1, units="um")

Apply image filters (from scipy and skimage) and plot

.. code-block:: python

    import seaborn_image as isns

    isns.filterplot(data, filter="gaussian")


Contents
========

.. toctree::
   :maxdepth: 1

   License <license>
   Authors <authors>
   Changelog <https://github.com/SarthakJariwala/seaborn-image/releases>
   Reference <reference>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
