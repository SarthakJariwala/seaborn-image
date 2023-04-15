import pytest

import matplotlib as mpl

import seaborn_image as isns

# without this, it suddenly started giving matplotlib backend issues locally
mpl.use("AGG")


@pytest.mark.parametrize(
    "context,outputs",
    [
        ("notebook", [2.5, 20, "normal", 20, "normal"]),
        ("talk", [2.5, 20, "normal", 20, "normal"]),
        ("presentation", [2.5, 20, "normal", 20, "normal"]),
        ("paper", [1.5, 15, "normal", 15, "normal"]),
        ("poster", [3.5, 25, "normal", 25, "normal"]),
    ],
)
def test_set_context(context, outputs):
    isns.set_context(context)

    assert mpl.rcParams["axes.linewidth"] == outputs[0]
    assert mpl.rcParams["axes.titlesize"] == outputs[1]
    assert mpl.rcParams["axes.titleweight"] == outputs[2]
    assert mpl.rcParams["axes.labelsize"] == outputs[3]
    assert mpl.rcParams["axes.labelweight"] == outputs[4]


@pytest.mark.parametrize("dpi", [300, 100, 0])
def test_set_save_context(dpi):
    isns.set_save_context(dpi)

    assert mpl.rcParams["savefig.dpi"] == dpi
    assert mpl.rcParams["savefig.bbox"] == "tight"


@pytest.mark.parametrize("cmap", ["ice", "acton", "viridis", "afmhot", "R"])
@pytest.mark.parametrize("origin", ["lower", "upper"])
@pytest.mark.parametrize("interpolation", ["nearest", "bicubic"])
def test_set_image(cmap, origin, interpolation):
    isns.set_image(cmap, origin, interpolation)

    assert mpl.rcParams["image.cmap"] == cmap
    assert mpl.rcParams["image.origin"] == origin
    assert mpl.rcParams["image.interpolation"] == interpolation


@pytest.mark.parametrize("despine", [True, False])
def test_image_despine(despine):
    isns.set_image(despine=despine)

    _d = not despine
    assert mpl.rcParams["axes.spines.bottom"] == _d
    assert mpl.rcParams["axes.spines.top"] == _d
    assert mpl.rcParams["axes.spines.left"] == _d
    assert mpl.rcParams["axes.spines.right"] == _d


@pytest.mark.parametrize("color", ["white", "k", "C4"])
@pytest.mark.parametrize("width_fraction", [0.025])
@pytest.mark.parametrize("length_fraction", [0.3])
@pytest.mark.parametrize("scale_loc", ["top", "bottom"])
@pytest.mark.parametrize("location", ["upper right", "center"])
@pytest.mark.parametrize("box_alpha", [0, 0.4])
def test_set_scalebar(
    color, width_fraction, length_fraction, scale_loc, location, box_alpha
):
    isns.set_scalebar(
        color=color,
        width_fraction=width_fraction,
        length_fraction=length_fraction,
        scale_loc=scale_loc,
        location=location,
        box_alpha=box_alpha,
    )

    assert mpl.rcParams["scalebar.color"] == color
    assert mpl.rcParams["scalebar.width_fraction"] == width_fraction
    assert mpl.rcParams["scalebar.length_fraction"] == length_fraction
    assert mpl.rcParams["scalebar.scale_loc"] == scale_loc
    assert mpl.rcParams["scalebar.location"] == location
    assert mpl.rcParams["scalebar.box_alpha"] == box_alpha


def test_height_fraction_deprecation():
    with pytest.deprecated_call():
        isns.set_scalebar(height_fraction=0.2)


def test_set_context_w_rc():
    isns.set_context(
        rc={"axes.edgecolor": "red", "axes.labelweight": "normal", "axes.labelsize": 30}
    )

    assert mpl.rcParams["axes.edgecolor"] == "red"
    assert mpl.rcParams["axes.labelweight"] == "normal"
    assert mpl.rcParams["axes.labelsize"] == 30


def test_scalebar_w_rc():
    # values other than deafult scalebar values
    isns.set_scalebar(
        rc={
            "color": "red",
            "width_fraction": 0.01,
            "length_fraction": 0.5,
            "scale_loc": "bottom",
            "location": "upper right",
            "box_alpha": 1,
            "label_loc": "top",
        }
    )

    assert mpl.rcParams["scalebar.color"] == "red"
    assert mpl.rcParams["scalebar.width_fraction"] == 0.01
    assert mpl.rcParams["scalebar.length_fraction"] == 0.5
    assert mpl.rcParams["scalebar.scale_loc"] == "bottom"
    assert mpl.rcParams["scalebar.location"] == "upper right"
    assert mpl.rcParams["scalebar.box_alpha"] == 1
    assert mpl.rcParams["scalebar.label_loc"] == "top"


def test_rc_assertion():
    with pytest.raises(AssertionError):
        rc = ["axes.labelweight", "normal"]
        isns.set_context(rc=rc)

        rc = ["color", "black"]
        isns.set_scalebar(rc=rc)


def test_reset_defaults():
    isns.reset_defaults()

    for k, v in mpl.rcParams.items():
        # ignore scalebar keys since they are not in
        # matplotlib defaults
        if "scalebar" in k:
            pass
        else:
            assert mpl.rcParams[k] == mpl.rcParamsDefault[k]
