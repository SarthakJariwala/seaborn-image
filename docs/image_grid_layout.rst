ImageGrid spacing and shared colorbars
=====================================

Use :class:`~seaborn_image.ImageGrid` to arrange images, control the space
between them, and compare them using a common color scale. ``gap`` and
``cbar="shared"`` are independent: use either feature on its own or both together.

.. contents:: On this page
   :local:

.. _image-grid-gap:

Remove whitespace between images
-------------------------------

Set ``gap=0`` and ``cbar=False`` for touching images without individual
colorbars. No call to ``subplots_adjust`` is needed.

.. plot::

    import seaborn_image as isns

    cells = isns.load_image("cells")
    g = isns.ImageGrid(
        cells, slices=[0, 10, 20, 30, 40, 50], col_wrap=3,
        height=1.5, gap=0, cbar=False,
    )

``gap`` measures the distance between rendered images **in inches**, not
pixels or a fraction of an axes. For example, ``gap=0.1`` gives a 10-pixel
gap at 100 DPI. The same distance is used horizontally and vertically.
Omit ``gap`` (or use ``gap=None``) to retain automatic layout.

.. plot::

    import seaborn_image as isns

    cells = isns.load_image("cells")
    g = isns.ImageGrid(
        cells, slices=[0, 10, 20, 30, 40], col_wrap=3,
        height=1.5, gap=0.1, cbar=False,
    )

An incomplete final row still contains unused slots; ``gap=0`` does not
remove these slots or rearrange the grid.

Explicit ``gap`` sizes the figure using ``height`` and the displayed image
proportions, including changes from ``extent`` or ``map_func``, instead of
the ``aspect`` parameter. It also overrides automatic Matplotlib layout settings.

.. important::

   Explicit spacing requires ``showticks=False`` (the default), matching
   displayed image proportions, and either ``cbar=False`` or ``cbar="shared"``.
   Per-image colorbars are not supported with explicit spacing. Mixed
   proportions raise an error rather than stretching or cropping images;
   use ``gap=None`` for a collection with mixed proportions.

   Spacing applies at the initial figure size. Resizing the figure or calling
   ``tight_layout`` or ``subplots_adjust`` afterwards can change it.

.. _image-grid-shared-colorbar:

Use one colorbar for the whole grid
----------------------------------

Choose ``cbar="shared"`` when all images measure the **same quantity in the
same units**. Every image then uses one normalization and colormap: the same
value produces the same color throughout the grid. This is different from
merely hiding all but one of several independently scaled colorbars.

.. plot::

    import seaborn_image as isns

    cells = isns.load_image("cells")
    g = isns.ImageGrid(
        cells, slices=[0, 10, 20, 30, 40, 50], col_wrap=3,
        height=1.5, cbar="shared", cbar_label="Intensity",
    )

The bar is vertical and on the right by default. Use ``orientation="h"``
(or ``"horizontal"``) to put it below the grid. The figure grows to make
room for the bar and its labels without shrinking the images or changing
their gaps. ``cbar=True`` still creates individual colorbars; ``cbar=False``
creates none.

.. plot::

    import seaborn_image as isns

    cells = isns.load_image("cells")
    g = isns.ImageGrid(
        cells, slices=[0, 10, 20, 30, 40, 50], col_wrap=3,
        height=1.5, gap=0, cbar="shared", orientation="h",
        robust=True, perc=(2, 98), cbar_label="Intensity",
    )

Choose a shared color scale
~~~~~~~~~~~~~~~~~~~~~~~~~~

* By default, limits cover all finite, unmasked values in the **selected,
  transformed images**. Unselected slices do not affect the limits.
* Set ``vmin`` and ``vmax`` to use fixed limits, for example to compare
  separate figures. If only one is supplied, the other is inferred.
* ``robust=True`` uses pooled pixel percentiles (``perc=(2, 98)`` by default),
  not an average of per-image percentiles. Images with more pixels contribute
  more values. Explicit limits take precedence over percentile limits.
* ``diverging=True`` makes limits symmetric around zero.
* ``cbar_log=True`` uses logarithmic normalization. It needs positive limits;
  nonpositive values are masked by logarithmic normalization.
* Alternatively, pass a Matplotlib ``Normalize`` instance as ``norm``. Missing
  limits are autoscaled over all selected images. Do not combine it with
  ``vmin``, ``vmax``, ``robust=True``, or ``diverging=True``.

For fixed limits and custom ticks:

.. code-block:: python

    g = isns.ImageGrid(
        cells, slices=[0, 10, 20], cbar="shared",
        vmin=0, vmax=0.3, cbar_ticks=[0, 0.1, 0.2, 0.3],
        cbar_label="Intensity",
    )

The created Matplotlib colorbar is available as ``g.colorbar`` and its axes
as ``g.cbar_ax``. For example, ``g.colorbar.set_label("Intensity (a.u.)")``
updates the label. Leave room when adding or enlarging labels after creation.

Shared colorbars require scalar images, not RGB/RGBA data. Per-image
colormaps, limits, normalizations, and other per-image color settings are
rejected because a single bar cannot describe conflicting color scales.

.. _image-grid-export:

Save without extra export padding
---------------------------------

Image spacing and export padding are separate. To omit the padding added by
Matplotlib's tight bounding-box export:

.. code-block:: python

    g.fig.savefig("cells.png", dpi=150, bbox_inches="tight", pad_inches=0)

This does not remove space reserved for colorbar labels or empty grid slots.
For visible border strokes at the edge of an export, a small nonzero
``pad_inches`` can be useful.

See also
--------

* :doc:`api/ImageGrid` for all parameters.
* :doc:`auto_examples/plot_ImageGrid_shared_colorbar` for a downloadable example.
