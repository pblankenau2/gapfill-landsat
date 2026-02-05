"""Module for filling gaps in Landsat 7 SLC-off images using the NSPI algorithm."""

import numpy as np
import numba
from numba import types, prange
from numba.typed import Dict

# TODO: make jit and regular versions of functions for debugging.


########################################################################################################################
# Helper functions
########################################################################################################################
def _propgate_nan_through_bands(array):
    """If any value is np.nan along axis 0 then np.nan is propogated to the rest of the indices along axis."""
    # TODO: It would be nice to generalize this to work on any axis.
    array_nans = np.sum(array, axis=0)  # Propogate nans
    array = np.where(np.isnan(array_nans), np.nan, array)
    return array


def find_similarity_threshold(image, nclasses):
    """Estimate the similarity threshold.

    :param image: A 3D array where the first dim is the band.  Usually this is
    the input image, i.e. the image used to find similar pixels.
    :param nclasses: Number of land cover classes present in image. Empirical.

    """
    nbands = image.shape[0]
    return np.nansum(np.nanstd(image, axis=(1, 2)) * 2.0 / nclasses, axis=0) / nbands


def find_init_window_size(min_similar_pix):
    """Find an initial window size where the min_similar_pix can be found."""
    return int((np.sqrt(min_similar_pix) + 1) / 2) * 2 + 1


def _pad_array(array, pad_width):
    """Pad a 3d array along its 1st and 2nd axes with np.nan.

    This kind of padding is intended for multiband images data.

    :param array: A 3D array to pad.
    :type array: np.array
    :param padding: Distance to pad.
    :type padding: int

    """
    return np.stack(
        [
            np.pad(
                array[i].copy(),
                pad_width=pad_width,
                mode="constant",
                constant_values=np.nan,
            )
            for i in range(array.shape[0])
        ]
    )


@numba.jit(nopython=True)
def _square_window_decimated(target_image, input_image, index, window_size, step):
    i, j = index

    # can_be_decimated = window_size//2%step # TODO only allow these window sizes?

    window_center_idx = window_size // 2

    # TODO: smallest decimated window should have a size of 3

    return (
        target_image[
            :,
            (i - window_center_idx) : (i + window_center_idx + 1) : step,
            (j - window_center_idx) : (j + window_center_idx + 1) : step,
        ],
        input_image[
            :,
            (i - window_center_idx) : (i + window_center_idx + 1) : step,
            (j - window_center_idx) : (j + window_center_idx + 1) : step,
        ],
    )


@numba.jit(nopython=True)
def _window_center(window_size):
    """Assumes a square window."""
    return window_size // 2, window_size // 2


@numba.jit(nopython=True)
def _window_rmsd(input_window):
    bands, x, y = input_window.shape
    window_size = x  # Assumes a square window
    window_center_x, window_center_y = _window_center(window_size)

    center_vals = (
        input_window[:, window_center_x, window_center_y].copy().reshape(bands, 1, 1)
    )
    rmsd = np.sqrt(np.sum(np.square(center_vals - input_window), axis=0) / bands)

    # Make rmsd with 0 a tiny number to avoid zero division issues
    rmsd = rmsd + 1e-10

    return rmsd


@numba.jit(nopython=True)
def _window_gaussian(input_window):
    """Not multiband."""
    bands, x, y = input_window.shape
    window_size = x  # Assumes a square window
    window_center_x, window_center_y = _window_center(window_size)

    center_vals = (
        input_window[:, window_center_x, window_center_y].copy().reshape(bands, 1, 1)
    )
    # rmsd = np.sqrt(np.sum(np.square(center_vals - input_window), axis=0) / bands)
    sig = 0.3

    gaussian = (
        np.exp(-np.power(input_window - center_vals, 2.0) / (2 * np.power(sig, 2.0)))
        * -1
        + 1.0001
    )

    # Make rmsd with 0 a tiny number to avoid zero division issues
    gaussian = gaussian + 1e-10

    return gaussian


@numba.jit(nopython=True)
def _absolute_difference(input_window):
    """Not multiband."""
    bands, x, y = input_window.shape
    window_size = x  # Assumes a square window
    window_center_x, window_center_y = _window_center(window_size)

    center_vals = (
        input_window[:, window_center_x, window_center_y].copy().reshape(bands, 1, 1)
    )
    abs_diff = np.absolute(center_vals - input_window)

    # Make abs_diff with 0 a tiny number to avoid zero division issues
    abs_diff = abs_diff + 1e-10

    return abs_diff


@numba.jit(nopython=True)
def _window_distance(window_size):
    window_center_x, window_center_y = _window_center(window_size)

    # Find distance to the center pixel for the window
    idx = np.stack(
        (
            np.arange(0, window_size).reshape(window_size, 1)
            + np.zeros((window_size, window_size)),
            np.arange(0, window_size) + np.zeros((window_size, window_size)),
        ),
        axis=0,
    )

    center_idx = np.array([[[window_center_x]], [[window_center_y]]])
    dist = np.sqrt(np.sum(np.square(idx - center_idx), axis=0))
    return dist


@numba.jit(nopython=True)
def _can_downsample(window_size, step):
    if window_size // 2 % step == 0:
        return True
    else:
        return False


@numba.jit(nopython=True)
def _find_step_size(window_size, max_effective_size):
    step_estimate = round(window_size / max_effective_size)

    for step in range(max(step_estimate, 1), window_size, 1):
        if _can_downsample(window_size, step):
            return step
    else:
        raise ValueError("No step size found!")


########################################################################################################################


@numba.jit(nopython=True)
def _interpolator(
    target_window,
    input_window,
    similarity_threshold,
    similarity,
    prediction_method="combined",
):
    """target_window must have the same dimensions as the input_window.

    :param similarity: An array the shape of the window where the values
    reflect degree of similarity to the target pixel.
    :param prediction_method: One of 'spatial', 'temporal', 'combined'.  'spatial'
    is the prediction derived only from similar pixels in the target_window.
    'temporal' is the prediction derived from the difference between the similar
    pixels in the target and the input windows.  'combined' is a weighted average
    of the two predictions.
    """
    bands, window_size = input_window.shape[0], input_window.shape[1]
    window_center_x, window_center_y = _window_center(window_size)

    # Find distance to the center pixel for the window
    dist = np.square(_window_distance(window_size))

    # Mask the input window where target window is masked
    similarity_masked = np.where(
        np.isnan(target_window)[0], np.nan, similarity
    )  # TODO: write a union_nulls function

    # Find pixels that are similar enough
    similar_pixels = np.where(
        (similarity_masked <= similarity_threshold), similarity_masked, np.nan
    )

    # TODO: the two lines below are from the nNSPI algorithm not the original paper.
    dist = (dist - np.nanmin(dist)) / (np.nanmax(dist) - np.nanmin(dist) + 1e-10) + 1
    similar_pixels = (similar_pixels - np.nanmin(similar_pixels)) / (
        np.nanmax(similar_pixels) - np.nanmin(similar_pixels) + 1e-10
    ) + 1

    # First prediction weights
    weights = 1.0 / (dist * similar_pixels)
    weights = weights / np.nansum(weights)
    weights = np.where(np.isnan(weights), 0.0, weights)

    # Prepare the target window
    target_window_no_nan = np.where(np.isnan(target_window), 0.0, target_window)

    # Compute first prediction value
    # nansum doesn't work with axis arg in numba
    value1 = np.sum(weights * target_window_no_nan, axis=1)
    value1 = np.sum(value1, axis=1)

    # Compute second prediction value
    center_vals = input_window[:, window_center_x, window_center_y].copy()
    change = target_window - input_window
    change_no_nan = np.where(np.isnan(change), 0.0, change)
    value2 = np.sum(weights * change_no_nan, axis=1)
    value2 = center_vals + np.sum(value2, axis=1)

    # Find the window's homogenaity
    homogenaity = np.nanmean(
        similarity
    )  # TODO: the center value in similarity is still present.

    # Degree of change between input and target windows
    radiometric_change = np.nanmean(np.sqrt(np.sum(np.square(change), axis=0) / bands))
    # Avoid zero division issues.
    radiometric_change = radiometric_change + 1e-10

    # Compute weights for value1 and value2
    weight1 = (1.0 / homogenaity) / ((1.0 / radiometric_change) + (1.0 / homogenaity))
    weight2 = (1.0 / radiometric_change) / (
        (1.0 / radiometric_change) + (1.0 / homogenaity)
    )

    # Compute the final value
    value = weight1 * value1 + weight2 * value2

    if prediction_method == "spatial":
        return value1
    elif prediction_method == "temporal":
        return value2
    else:
        return value


@numba.jit(nopython=True, parallel=True)
def _interpolate(
    padded_target_image,
    padded_input_image,
    image_width,
    image_height,
    similarity_threshold,
    min_num_similar_pix,
    window_sizes,
    max_effective_window_size,
    prediction_method,
):

    # Create an image that will collect the fill values.
    filled_image = np.zeros(shape=padded_target_image.shape) - 99

    window_size_to_step = Dict.empty(  # TODO: Change to 16 or 32bit int
        key_type=types.int64, value_type=types.int64
    )

    for ws in window_sizes:
        window_size_to_step[ws] = _find_step_size(ws, max_effective_window_size)
    # TODO: What not just run this function within the loop?

    max_window_center_idx = max(window_sizes) // 2

    # The indices we want are those where the whole window fits within the image
    for i in prange(max_window_center_idx, image_height + max_window_center_idx):
        for j in range(max_window_center_idx, image_width + max_window_center_idx):

            # Only operate on windows where the target pixel is nan and the input pixel is valid
            # Only check the first band
            if np.isnan(padded_target_image[0, i, j]) and not np.isnan(
                padded_input_image[0, i, j]
            ):

                min_num_similar_pix_found = False
                num_similar_pix = 0
                # Expand the window size from min to max until enough similar pixels are found.
                for window_size in window_size_to_step.keys():

                    target_window, input_window = _square_window_decimated(
                        padded_target_image,
                        padded_input_image,
                        (i, j),
                        window_size=window_size,
                        step=window_size_to_step[window_size],
                    )

                    # Similarity
                    rmsd = _window_rmsd(input_window)

                    # Mask the input window where target window is masked
                    input_window_mask = np.isnan(target_window)[0]
                    rmsd_masked = np.where(input_window_mask, np.nan, rmsd)

                    # Check if min number of similar pixels are present
                    num_similar_pix = np.sum(rmsd_masked <= similarity_threshold)
                    if num_similar_pix >= min_num_similar_pix:
                        min_num_similar_pix_found = True
                        # Add the predicted pixel value to the fill_image
                        filled_image[:, i, j] = _interpolator(
                            target_window,  # TODO: i and j in correct order?
                            input_window,
                            similarity_threshold,
                            rmsd,
                            prediction_method,
                        )
                        break  # No need to look farther out with bigger windows.
                    else:  # Contine the loop with an expanded window size
                        continue
                if (
                    not min_num_similar_pix_found
                ):  # If the min number of similar pixels was never available.
                    if num_similar_pix > 0:
                        filled_image[:, i, j] = _interpolator(
                            target_window,
                            input_window,
                            similarity_threshold,
                            rmsd,
                            prediction_method,
                        )
                    else:
                        # Compute change from common pixel and use as fill value.
                        filled_image[:, i, j] = np.nan
    return filled_image


def nspi(
    target_image,
    input_image,
    similarity_threshold=None,
    window_sizes=range(5, 89, 2),
    max_effective_window_size=30,
    min_num_similar_pix=20,
    prediction_method="combined",
):
    """Fill the target image using the nearest similar pixel interpolator (NSPI).

    This function fills np.nan values in the target image.  Any pixels you don't
    want filled should be assigned to another value.  Pixels that you don't
    want to fill and you also don't want to be considered as similar pixels
    should be assigned a value far outside the range of values in the image.

    - Note: Apply the cloud mask Nans from the target image to the input image,
        but not Nans you want filled (e.g., SLC-off gaps).  This will ensure that
        clouds in the target image will not be filled.
    - Note: -99 has a special meaning in this algorithm and images that contain
      values of -99 will cause issues.

    .. topic:: References
        - `"A simple and effective method for filling gaps in Landsat ETM+ SLC-off images"
          <https://doi.org/10.1016/j.rse.2010.12.010>`_
          Chen et al., (2011)

    :param target_image: An image (bands x width x height) that needs to be filled.
    :type target_image: np.array
    :param input_image: An image used to locate similar pixels.
    :type input_image: np.array
    :param similarity_threshold: How similar (RMSD) pixels should be to be
        used as predictors of the missing pixel.  If None, the threshold is
        automatically estimated with 5 classes.
    :type similarity_threshold: int
    :param window_sizes: A collection of window sizes to iterate through while
        searching for the minimum number of similar pixels.
    :type window_sizes: list or range
    :param max_effective_window_size: The window will be downsampled so that
        it never exceeds this size.  Keeping this small hinders accuracy but
        improves the speed of the algorithm.
    :type max_effective_window_size: int
    :param min_num_similar_pix: The minimum number of similar pixels to find
        before predicting the missing pixel.
    :param prediction_method: One of 'spatial', 'temporal', 'combined'.  'spatial'
    is the prediction derived only from similar pixels in the target_window.
    'temporal' is the prediction derived from the difference between the similar
    pixels in the target and the input windows.  'combined' is a weighted average
    of the two predictions.
    :type prediction_method: string

    """
    if not similarity_threshold:
        similarity_threshold = find_similarity_threshold(input_image, 5)

    # Casting inputs
    window_sizes = list(window_sizes)
    target_image = target_image.astype(np.float32)
    input_image = input_image.astype(np.float32)

    # Unify nodata mask, all bands should share a mask
    target_image = _propgate_nan_through_bands(target_image)
    input_image = _propgate_nan_through_bands(input_image)

    # Original image dimensions
    bands, x, y = target_image.shape

    max_window_center_idx = (
        max(window_sizes) // 2
    )  # TODO: this is recomputed multiple times.

    # Pad input images with the max window size
    padded_target_image = _pad_array(target_image, max_window_center_idx)
    padded_input_image = _pad_array(input_image, max_window_center_idx)

    filled_image = _interpolate(
        padded_target_image,
        padded_input_image,
        y,
        x,
        similarity_threshold,
        min_num_similar_pix,
        window_sizes,
        max_effective_window_size,
        prediction_method,
    )

    filled_image = filled_image[
        :,
        max_window_center_idx : x + max_window_center_idx,
        max_window_center_idx : y + max_window_center_idx,
    ]

    filled_image = np.where(
        filled_image == -99.0, target_image, filled_image
    )  # TODO: is using -99 as a sentinel an issue?
    return filled_image
