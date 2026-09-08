import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import seaborn_image as isns


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("include_original", [True, False])
def test_sequential_images_and_panels(include_original):
    source = np.array([[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]])
    calls = []

    def add(image, amount):
        calls.append("add")
        image += amount
        return image

    def multiply(image, factor):
        calls.append("multiply")
        image *= factor
        return image

    pipeline = isns.pipelineplot(
        source,
        [("Add", add, {"amount": 2}), ("Multiply", multiply, {"factor": 3})],
        include_original=include_original,
        col_wrap=2,
        cbar=False,
        cmap="gray",
        vmin=0,
        vmax=21,
    )
    expected = [
        np.array([[3.0, 6.0, 4.0], [5.0, 2.0, 7.0]]),
        np.array([[9.0, 18.0, 12.0], [15.0, 6.0, 21.0]]),
    ]
    labels = ["Add", "Multiply"]
    if include_original:
        expected.insert(0, source)
        labels.insert(0, "Original")
    assert calls == ["add", "multiply"]
    assert isinstance(pipeline, isns.ImageGrid)
    assert len(pipeline.images) == len(expected)
    for image, ax, result, label in zip(
        pipeline.images, pipeline.axes.flat, expected, labels
    ):
        np.testing.assert_array_equal(image, result)
        np.testing.assert_array_equal(ax.images[0].get_array(), result)
        assert ax.get_title() == label
        assert ax.images[0].get_clim() == (0, 21)
        assert not np.shares_memory(image, source)
    np.testing.assert_array_equal(source, [[1, 4, 2], [3, 0, 5]])
    pipeline.images[-1][:] = 0
    np.testing.assert_array_equal(pipeline.images[-2], expected[-2])


def test_rgb_to_grayscale_and_single_panel():
    source = np.arange(24).reshape(2, 4, 3) / 24
    pipeline = isns.pipelineplot(
        source,
        [("Red channel", lambda image: image[:, :, 0], {})],
        include_original=False,
        cbar=False,
    )
    assert pipeline.axes.shape == (1, 1)
    np.testing.assert_array_equal(pipeline.images[0], source[:, :, 0])


@pytest.mark.parametrize(
    "steps, error",
    [
        ([], ValueError),
        ([("Incomplete", np.negative)], ValueError),
        ([("Not callable", "gaussian", {})], TypeError),
        ([("Bad kwargs", np.negative, None)], TypeError),
        ([(1, np.negative, {})], TypeError),
    ],
)
def test_invalid_steps(steps, error):
    with pytest.raises(error):
        isns.pipelineplot(np.ones((2, 3)), steps)


@pytest.mark.parametrize(
    "data", [None, np.ones(3), np.ones((2, 3, 2)), np.empty((0, 3))]
)
def test_invalid_input_or_output(data):
    with pytest.raises(ValueError, match="pipeline images"):
        isns.pipelineplot(data, [("Identity", lambda image: image, {})])
    with pytest.raises(ValueError, match="pipeline images"):
        isns.pipelineplot(np.ones((2, 3)), [("Invalid", lambda image: data, {})])


def test_reject_mapping_options():
    with pytest.raises(TypeError, match="map_func"):
        isns.pipelineplot(
            np.ones((2, 3)), [("Negative", np.negative, {})], map_func=np.negative
        )
