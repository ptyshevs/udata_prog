from Matrix import Matrix
import collections


def eye(n):
    """
    Create nxn Identity matrix
    :param n: side
    :return:
    """
    return Matrix([[1 if i == j else 0 for i in range(n)] for j in range(n)])


def flipud(A):
    """
    Flip lower-upper triangular matrix
    :param A: Matrix
    :return:
    """
    return A[::-1]


def zeros(shape):
    """
    Create Matrix of size <shape>, filled with zeros.
    If shape is integer, create nxn zero matrix
    :param shape:
    :return:
    """
    if type(shape) is int:
        return Matrix([[0 for _ in range(shape)] for _ in range(shape)])
    elif isinstance(shape, collections.Sequence) and len(shape) == 2:
        return Matrix([[0 for _ in range(shape[1])] for _ in range(shape[0])])
    else:
        raise ValueError("Don't understand input shape:", shape)


def argmax(A, axis=0):
    """
    Find index of maximum value in A
    :param A: Matrix
    :param axis: 0 for row index, 1 for column index, 2 for (row, col) tuple
    :return: index (-1 in case of error)
    """
    max_row, max_col, max_val = -1, -1, None
    if type(A) in (list, tuple):  # instead of failing miserably, find proper index
        for i, v in enumerate(A):
            if max_val is None:
                max_val = v
                max_row = i
            elif v > max_val:
                max_val = v
                max_row = i
        return max_row
    if A.shape == (0, 0):
        return max_row
    if A.shape[0] == 1 or A.shape[1] == 1:
        for i, val in enumerate(A):
            if max_val is None:
                max_val = val
                max_row = i
            elif val > max_val:
                max_val = val
                max_row = i
        return max_row
    for i, row in enumerate(A):
        for j, col in enumerate(row):
            if max_val is None:
                max_val = col
                max_row, max_col = i, j
            elif col > max_val:
                max_val = col
                max_row, max_col = i, j
    if axis == 0:
        return max_row
    elif axis == 1:
        return max_col
    else:
        return max_row, max_col


if __name__ == '__main__':
    A = Matrix([[1, 2, 20, 3, -5, 0],
                [0, 0, 3, -1, 2, 55]])
    print(argmax(A, axis=2))
    print(A[argmax(A, axis=2)])
