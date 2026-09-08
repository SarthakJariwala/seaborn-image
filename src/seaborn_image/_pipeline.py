import numpy as np

from ._grid import ImageGrid

__all__ = ["pipelineplot"]


def _snapshot_image(data):
    image = np.array(data, copy=True)
    if image.ndim != 2 and not (image.ndim == 3 and image.shape[-1] in (3, 4)):
        raise ValueError("pipeline images must be 2-D grayscale or RGB/RGBA arrays")
    if not image.size:
        raise ValueError("pipeline images must not be empty")
    return image


def pipelineplot(data, steps, *, include_original=True, **grid_kws):
    """Apply named processing steps sequentially and plot every result.

    Parameters
    ----------
    data : array-like
        A 2-D grayscale or RGB/RGBA image. The input is not modified.
    steps : sequence of (str, callable, dict)
        Ordered triples of panel label, function, and keyword arguments.
        Each function receives the preceding image as its first argument and
        must return an image. Use an empty dictionary for no arguments.
        At least one step is required. Functions may change image shape.
    include_original : bool, optional
        Include the original as the first panel and first entry in ``images``.
        Defaults to True.
    **grid_kws
        Display options forwarded to :class:`ImageGrid`, such as ``col_wrap``,
        ``cmap``, ``cbar``, ``vmin``, and ``vmax``. Mapping and slicing options
        are not supported. Per-panel options include the original when shown.
        Color limits default to independent scaling; pass ``vmin`` and
        ``vmax`` explicitly to compare stages on a shared intensity scale.

    Returns
    -------
    pipeline : ImageGrid
        Grid with ``fig``, ``axes``, and ``images`` attributes. ``images`` is
        a list of image arrays in panel order, before display normalization.
        Each array is a snapshot: later steps, including in-place filters,
        cannot overwrite earlier results. ``images[-1]`` is the final result.
        Editing these arrays after plotting does not update the figure.

    Examples
    --------
    >>> import seaborn_image as isns
    >>> from skimage import data, filters
    >>> pipeline = isns.pipelineplot(
    ...     data.coins(),
    ...     steps=[
    ...         ("Gaussian (sigma=2)", filters.gaussian, {"sigma": 2}),
    ...         ("Sobel", filters.sobel, {}),
    ...     ],
    ...     cmap="gray", cbar=False,
    ... )
    >>> edges = pipeline.images[-1]
    """
    if not isinstance(include_original, bool):
        raise TypeError("include_original must be a bool")
    if not isinstance(steps, (list, tuple)) or not steps:
        raise ValueError("steps must be a non-empty list or tuple of triples")
    for step in steps:
        if not isinstance(step, (list, tuple)) or len(step) != 3:
            raise ValueError("each step must be a (label, callable, kwargs) triple")
        label, func, kwargs = step
        if (
            not isinstance(label, str)
            or not callable(func)
            or not isinstance(kwargs, dict)
        ):
            raise TypeError("each step requires a string label, callable, and dict")
    unsupported = set(grid_kws) & {
        "map_func",
        "map_func_kw",
        "slices",
        "axis",
        "step",
        "start",
        "stop",
    }
    if unsupported:
        raise TypeError(
            f"pipelineplot does not support: {', '.join(sorted(unsupported))}"
        )

    current = _snapshot_image(data)
    images = [current] if include_original else []
    labels = ["Original"] if include_original else []
    for label, func, kwargs in steps:
        current = _snapshot_image(func(current.copy(), **kwargs))
        images.append(current)
        labels.append(label)

    pipeline = ImageGrid(images, **grid_kws)
    pipeline.images = images
    for ax, label in zip(pipeline.axes.flat, labels):
        ax.set_title(label)
    pipeline.fig.tight_layout()
    return pipeline
