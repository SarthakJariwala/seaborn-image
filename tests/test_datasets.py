import pytest

import numpy as np
import pooch
from skimage import color, data, io

import seaborn_image as isns


def test_load_image():
    # test polymer
    img = isns.load_image("polymer")
    fname = pooch.retrieve(
        url="https://raw.githubusercontent.com/SarthakJariwala/seaborn-image/master/data/PolymerImage.txt",
        known_hash="7b6798865080adf3ecf11e342f3d86d7b52ea0700020a1f062544ee825fb8a0e",
    )
    test_img = np.loadtxt(fname, skiprows=1) * 1e9
    np.testing.assert_array_equal(img, test_img)

    # test polymer
    img = isns.load_image("polymer outliers")
    fname = pooch.retrieve(
        url="https://raw.githubusercontent.com/SarthakJariwala/seaborn-image/master/data/PolymerImage.txt",
        known_hash="7b6798865080adf3ecf11e342f3d86d7b52ea0700020a1f062544ee825fb8a0e",
    )
    test_img = np.loadtxt(fname, skiprows=1) * 1e9
    test_img[25, 25] = 80
    np.testing.assert_array_equal(img, test_img)

    img = isns.load_image("fluorescence")
    fname = pooch.retrieve(
        url="https://raw.githubusercontent.com/SarthakJariwala/seaborn-image/master/data/Perovskite.txt",
        known_hash="3228eeade5afec3c2b1ed116b2d4fe35877224d2d9bf7b4a17e04a432e6135c5",
    )
    test_img = np.loadtxt(fname)
    np.testing.assert_array_equal(img, test_img)


def test_load_image_from_skimage():
    img = isns.load_image("cells")

    fname = pooch.retrieve(
        url="https://github.com/SarthakJariwala/seaborn-image/raw/master/data/cells.tif",
        known_hash="2120cfe08e0396324793a10a905c9bbcb64b117215eb63b2c24b643e1600c8c9",
    )
    test_img = io.imread(fname).T
    np.testing.assert_array_equal(img, test_img)

    img = isns.load_image("retina-gray")
    test_img = color.rgb2gray(data.retina())[300:700, 700:900]
    np.testing.assert_array_equal(img, test_img)


def test_load_image_cifar10():
    img = isns.load_image("cifar10")

    fname = pooch.retrieve(
        url="https://github.com/eugenioLR/seaborn-image/raw/multiformat-images/data/cifar10.npy",
        known_hash="c0a12085b3b82f4a6d1f95e609a40701648a137eb9ff1fb5751071f54cc8e05c",
    )
    test_img = np.load(fname)
    np.testing.assert_array_equal(img, test_img)


def test_load_image_cifar10_list():
    img_list = isns.load_image("cifar10 list")

    fname = pooch.retrieve(
        url="https://github.com/eugenioLR/seaborn-image/raw/multiformat-images/data/cifar10.npy",
        known_hash="c0a12085b3b82f4a6d1f95e609a40701648a137eb9ff1fb5751071f54cc8e05c",
    )
    test_img = np.load(fname)
    for idx, img in enumerate(img_list):
        np.testing.assert_array_equal(img, test_img[idx])


def test_load_image_error():
    with pytest.raises(ValueError):
        isns.load_image("coins")
