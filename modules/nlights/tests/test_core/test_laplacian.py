# TEST INUTILE!!
import numpy as np
import numba as nb
import pytest

# Assuming build_laplacian_indices is already defined in the same file or imported

@nb.njit(parallel=True)
def build_laplacian_indices(height, width, mask, pixel_indices):
    row_indices = []
    col_indices = []
    values = []

    for i in nb.prange(height):
        for j in range(width):
            if not mask[i, j]:
                continue

            idx = pixel_indices[i, j]
            neighbors = 0

            if i > 0 and mask[i - 1, j]:
                row_indices.append(idx)
                col_indices.append(pixel_indices[i - 1, j])
                values.append(-1)
                neighbors += 1
            if i < height - 1 and mask[i + 1, j]:
                row_indices.append(idx)
                col_indices.append(pixel_indices[i + 1, j])
                values.append(-1)
                neighbors += 1
            if j > 0 and mask[i, j - 1]:
                row_indices.append(idx)
                col_indices.append(pixel_indices[i, j - 1])
                values.append(-1)
                neighbors += 1
            if j < width - 1 and mask[i, j + 1]:
                row_indices.append(idx)
                col_indices.append(pixel_indices[i, j + 1])
                values.append(-1)
                neighbors += 1

            row_indices.append(idx)
            col_indices.append(idx)
            values.append(neighbors)

    return np.array(row_indices, dtype=np.int32), \
           np.array(col_indices, dtype=np.int32), \
           np.array(values, dtype=np.float32)

import numpy as np
import pytest

@pytest.mark.parametrize("height, width", [(5, 5), (10, 10)])
def test_build_laplacian_indices(height, width):
    mask = np.array([[1, 1, 0, 1, 1],
                     [1, 1, 1, 0, 1],
                     [0, 1, 1, 1, 0],
                     [1, 0, 1, 1, 1],
                     [1, 1, 1, 0, 1]], dtype=bool)

    pixel_indices = -np.ones((height, width), dtype=np.int32)
    valid_pixels = np.where(mask)
    num_valid_pixels = len(valid_pixels[0])
    pixel_indices[valid_pixels] = np.arange(num_valid_pixels, dtype=np.int32)

    row_indices, col_indices, values = build_laplacian_indices(height, width, mask, pixel_indices)

    # Check the output shapes
    assert len(row_indices) == len(col_indices) == len(values)

    # Define the expected output based on the mask and connectivity
    expected_row_indices = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4])
    expected_col_indices = np.array([1, 4, 0, 2, 4, 1, 1, 2, 3, 1, 0, 2, 3, 3, 4, 1])
    expected_values = np.array([-1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3])

    # Adjust expected values based on the actual connectivity
    # The last value should be the degree of the node (number of neighbors)
    expected_values = np.array([-1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3])

    # Check that the output matches the expected values
    assert np.array_equal(row_indices[:len(expected_row_indices)], expected_row_indices)
    assert np.array_equal(col_indices[:len(expected_col_indices)], expected_col_indices)
    assert np.array_equal(values[:len(expected_values)], expected_values)

    # Additional check for the degree of each valid pixel
    degree_counts = np.bincount(row_indices, weights=values)
    assert degree_counts[0] == 2  # Pixel (0,0) has 2 neighbors
    assert degree_counts[1] == 4  # Pixel (1,1) has 4 neighbors
    assert degree_counts[2] == 3  # Pixel (2,2) has 3 neighbors
    assert degree_counts[3] == 4  # Pixel (3,3) has 4 neighbors
    assert degree_counts[4] == 2  # Pixel (4,4) has 2 neighbors

# To run the test, use the command: pytest -v <name_of_this_file>.py
