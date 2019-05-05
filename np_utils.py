import numpy as np
from numba import jit
import copy


@jit(nopython=True)
def myrot180(a):
    """
    Accelerated method to rotate incoming numpy.ndarray by 180 degrees in the last two dimensions, equivalent to applying np.rot90 twice.
    :param a: Input numpy.ndarray
    :return: rotated input
    """
    rows = np.arange(0, a.shape[-2])
    rows_rev = np.arange(a.shape[-2] - 1, -1, -1)
    cols = np.arange(0, a.shape[-1])
    cols_rev = np.arange(a.shape[-1] - 1, -1, -1)
    a[..., rows, :] = a[..., rows_rev, :]
    a.T[cols] = a.T[cols_rev]
    return a


def myrot180_test(in_mat):
    b = copy.deepcopy(in_mat)
    result = np.rot90(b, 2, axes=(-2, -1))
    in_mat = myrot180(in_mat)
    t = np.allclose(in_mat, result)
    return t


if __name__ == "__main__":
    test_1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    assert myrot180_test(test_1), True


