#!/usr/bin/env python

"""Tests for `cloud_filler` package."""

import pytest
import numpy as np


from cloud_filler import cloud_filler  # TODO: import should be relative?


@pytest.fixture
def input_image_center_pixel_missing():
    """5x5 with pixel at 2,2 missing."""
    img = np.ones((1, 5, 5)).astype(np.float32)
    img[0, 2, 2] = np.nan
    return img


@pytest.fixture
def target_image():
    img = np.ones((1, 5, 5)).astype(np.float32)
    img[0, 2, 2] = np.nan
    img[0, 1, 1] = 0.5
    return img


@pytest.fixture
def input_image():
    img = np.ones((1, 5, 5)).astype(np.float32)
    img[0, 2, 2] = 0.3
    img[0, 1, 1] = 0.32
    return img


@pytest.fixture
def target_multiband_image():
    img = np.ones((2, 5, 5)).astype(np.float32)
    img[:, 2, 2] = np.nan
    img[0, 1, 1] = 0.5
    img[1, 1, 1] = 0.6
    return img


@pytest.fixture
def input_multiband_image():
    img = np.ones((2, 5, 5)).astype(np.float32)
    # Target pixel
    img[0, 2, 2] = 0.3
    img[1, 2, 2] = 0.4
    # Similar pixel
    img[0, 1, 1] = 0.32
    img[1, 1, 1] = 0.36
    return img


@pytest.fixture
def input_image_distant_similar_pixel():
    img = np.ones((1, 5, 5)).astype(np.float32)
    img[0, 2, 2] = 0.3
    img[0, 0, 0] = 0.32
    return img


@pytest.fixture
def target_image_distant_similar_pixel():
    img = np.ones((1, 5, 5)).astype(np.float32)
    img[0, 2, 2] = np.nan
    img[0, 0, 0] = 0.5
    return img


@pytest.fixture
def input_null():
    img = np.ones((1, 5, 5)).astype(np.float32)
    img[:] = np.nan
    return img


###############################################################################


def test_no_fill_when_both_missing(input_image_center_pixel_missing):
    output = cloud_filler.nspi(
        input_image_center_pixel_missing,
        input_image_center_pixel_missing,
        similarity_threshold=0.2,
        window_sizes=[3],
        min_num_similar_pix=2,
    )

    assert np.allclose(output, input_image_center_pixel_missing, equal_nan=True)


def test_no_fill_when_input_missing(input_null, target_image):
    output = cloud_filler.nspi(
        target_image,
        input_null,
        similarity_threshold=0.2,
        window_sizes=[3],
        min_num_similar_pix=2,
    )

    assert np.allclose(output, target_image, equal_nan=True)


def test_no_fill_when_no_similar_pixels(input_image, target_image):
    output = cloud_filler.nspi(
        target_image,
        input_image,
        similarity_threshold=0.0001,  # So small there will be no similar pix.
        window_sizes=[3],
        min_num_similar_pix=1,
    )

    assert np.allclose(output, target_image, equal_nan=True)


def test_singleband_spatial_filling(input_image, target_image):
    output = cloud_filler.nspi(
        target_image,
        input_image,
        similarity_threshold=0.1,
        window_sizes=[3],
        min_num_similar_pix=1,
        prediction_method="spatial",
    )

    assert output[0, 2, 2] == pytest.approx(0.5, abs=0.00001)


def test_singleband_temporal_filling(input_image, target_image):
    output = cloud_filler.nspi(
        target_image,
        input_image,
        similarity_threshold=0.1,
        window_sizes=[3],
        min_num_similar_pix=1,
        prediction_method="temporal",
    )

    assert output[0, 2, 2] == pytest.approx(0.48, abs=0.00001)


def test_singleband_combined_filling(input_image, target_image):
    output = cloud_filler.nspi(
        target_image,
        input_image,
        similarity_threshold=0.1,
        window_sizes=[3],
        min_num_similar_pix=1,
        prediction_method="combined",
    )

    assert output[0, 2, 2] == pytest.approx(0.48079, abs=0.00001)


def test_expanding_window_filling(
    input_image_distant_similar_pixel, target_image_distant_similar_pixel
):
    output = cloud_filler.nspi(
        target_image_distant_similar_pixel,
        input_image_distant_similar_pixel,
        similarity_threshold=0.1,
        window_sizes=[3, 5],
        min_num_similar_pix=1,
        prediction_method="spatial",
    )

    assert output[0, 2, 2] == pytest.approx(0.5, abs=0.00001)


def test_fewer_than_min_sim_pix_filling(input_image, target_image):
    output = cloud_filler.nspi(
        target_image,
        input_image,
        similarity_threshold=0.1,
        window_sizes=[3],
        min_num_similar_pix=10,
        prediction_method="combined",
    )

    assert output[0, 2, 2] == pytest.approx(0.48079, abs=0.00001)


def test_multiband_filling(input_multiband_image, target_multiband_image):
    output = cloud_filler.nspi(
        target_multiband_image,
        input_multiband_image,
        similarity_threshold=0.035,  # The similarity of the pixel is ~0.0316
        window_sizes=[3],
        min_num_similar_pix=10,
        prediction_method="spatial",
    )

    assert output[0, 2, 2] == pytest.approx(0.5, abs=0.00001)
    assert output[1, 2, 2] == pytest.approx(0.6, abs=0.00001)


def test_multiband_filling_similarity_threshold(
    input_multiband_image, target_multiband_image
):
    output = cloud_filler.nspi(
        target_multiband_image,
        input_multiband_image,
        similarity_threshold=0.03,  # The similarity of the pixel is ~0.0316
        window_sizes=[3],
        min_num_similar_pix=1,
        prediction_method="spatial",
    )

    assert np.allclose(output, target_multiband_image, equal_nan=True)


###############################################################################
# Tests for the helper functions
###############################################################################


def test_similarity_threshold(target_multiband_image):
    assert cloud_filler.similarity_threshold(
        target_multiband_image, 3
    ) == pytest.approx(0.0599, abs=0.0001)


def test_find_init_window_size():
    assert cloud_filler.find_init_window_size(4) == 3


def test__pad_arrays(target_multiband_image):
    padded = np.array(
        [
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan],
                [np.nan, 1.0, 0.5, 1.0, 1.0, 1.0, np.nan],
                [np.nan, 1.0, 1.0, np.nan, 1.0, 1.0, np.nan],
                [np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan],
                [np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan],
                [np.nan, 1.0, 0.6, 1.0, 1.0, 1.0, np.nan],
                [np.nan, 1.0, 1.0, np.nan, 1.0, 1.0, np.nan],
                [np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan],
                [np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
        ],
        dtype=np.float32,
    )
    assert np.allclose(
        cloud_filler._pad_array(target_multiband_image, 1), padded, equal_nan=True
    )


def test__square_window_decimated(target_multiband_image, input_multiband_image):
    # target_multiband_image = cloud_filler._pad_array(target_multiband_image, 2)
    # input_multiband_image = cloud_filler._pad_array(input_multiband_image, 2)
    t, i = cloud_filler._square_window_decimated(
        target_multiband_image,
        input_multiband_image,
        index=(2, 2),
        window_size=5,
        step=2,
    )

    assert np.allclose(
        i,
        np.array(
            [
                [[1.0, 1.0, 1.0], [1.0, 0.3, 1.0], [1.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0], [1.0, 0.4, 1.0], [1.0, 1.0, 1.0]],
            ]
        ),
        equal_nan=True,
    )
    assert np.allclose(
        t,
        np.array(
            [
                [[1.0, 1.0, 1.0], [1.0, np.nan, 1.0], [1.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0], [1.0, np.nan, 1.0], [1.0, 1.0, 1.0]],
            ]
        ),
        equal_nan=True,
    )


def test__window_rmsd(input_multiband_image):
    oracle = np.array(
        [
            [6.5192026e-01, 6.5192026e-01, 6.5192026e-01, 6.5192026e-01, 6.5192026e-01],
            [6.5192026e-01, 3.1622767e-02, 6.5192026e-01, 6.5192026e-01, 6.5192026e-01],
            [6.5192026e-01, 6.5192026e-01, 1.0000000e-10, 6.5192026e-01, 6.5192026e-01],
            [6.5192026e-01, 6.5192026e-01, 6.5192026e-01, 6.5192026e-01, 6.5192026e-01],
            [6.5192026e-01, 6.5192026e-01, 6.5192026e-01, 6.5192026e-01, 6.5192026e-01],
        ],
        dtype=np.float32,
    )
    assert np.allclose(
        cloud_filler._window_rmsd(input_multiband_image), oracle, equal_nan=True
    )


def test__window_distance():
    oracle = np.array(
        [
            [
                [1.4142135623730951, 1.0, 1.4142135623730951],
                [1.0, 0.0, 1.0],
                [1.4142135623730951, 1.0, 1.4142135623730951],
            ]
        ]
    )
    assert np.allclose(cloud_filler._window_distance(3), oracle)


def test__can_downsample():
    assert cloud_filler._can_downsample(5, 2) == True
    assert cloud_filler._can_downsample(9, 3) == False


def test__find_step_size():
    assert cloud_filler._find_step_size(101, 30) == 5
