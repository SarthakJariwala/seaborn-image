=======================================
seaborn-image: image data visualization
=======================================


Seaborn like image data visualization using matplotlib, scikit-image and scipy.


Description
===========

Seaborn-image is a seaborn like python **image** visualization and processing library
based on matplotlib.

The aim of seaborn-image is to provide a high-level API to **plot attractive images quickly**
**and effectively**.


Installation
============

``pip install seaborn-image``

Simple usage

.. code-block:: python

    import seaborn_image as isns

    """Set context like seaborn"""
    isns.set_context("notbook")

    """Plot publishable quality image in one line"""
    isns.imgplot(data)

    """Add a scalebar"""
    isns.imgplot(data, scalebar=True, dx=1, units="um")

Apply image filters and plot

.. code-block:: python

    import seaborn_image as isns

    isns.filterplot(data, filter="gaussian")

Note
====

This project was started because I was looking for a seaborn like library for images but couldn't find any.
The project is still a work in progress but give it a go and let me know...
