How to...?
==========

.. contents::
    :local:


Visualize distribution
----------------------

Add a histogram to the image
****************************

.. code-block:: python

    import seaborn_image as isns

    isns.imghist(data)


Change histogram bins for the distribution
******************************************

.. code-block:: python

    import seaborn_image as isns

    isns.imghist(data, bins=250)


Add and modify scale bar properties
-----------------------------------

Add a scale bar
***************

.. code-block:: python

    import seaborn_image as isns

    isns.imgplot(data, dx=1, units="um")


Modify scale bar properties
***************************

.. code-block:: python

    import seaborn_image as isns

    # change scalebar color
    isns.set_scalebar(rc={"color": "red"})

    # change scalebar position
    isns.set_scalebar(rc={"location": "upper right"})


Add label to colorbar
---------------------

.. code-block:: python

    import seaborn_image as isns

    isns.imgplot(data, cbar_label="Intensity (au)")


Use image filters
-----------------

.. code-block:: python

    import seaborn_image as isns

    isns.filterplot(data, filter="sobel")


Change image plot properties
----------------------------

.. code-block:: python

    import seaborn_image as isns

    isns.set_image(cmap="inferno", origin="upper")


General aesthetics
------------------

.. code-block:: python

    import seaborn_image as isns

    isns.set_context("paper") # or talk, notebook, presentation, poster

    # change font-family
    isns.set_context(fontfamily="sans-serif")

Arbitrary ``matplotlib`` rcParams can also be altered by passing them as
a dict to ``rc`` parameter.

.. code-block:: python

    # change axes title pad
    isns.set_context(rc={"axes.titlepad": 6})
