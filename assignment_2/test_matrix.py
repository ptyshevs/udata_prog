from matrix_tools import eye, zeros
from Matrix import Matrix


def test_zeros():
    """
    Test matrix filled with zeros of different shapes
    :return:
    """
    A = zeros(3)
    assert A == 0 and A.shape == (3, 3)
    A = zeros((10, 1))
    assert A == 0 and A.shape == (10, 1)
    A = zeros(((1, 5)))
    assert A == 0 and A.shape == (1, 5)


def test_eye():
    """
    Test 5x5 Identity matrix
    :return:
    """
    A = eye(5)
    for i, row in enumerate(A):
        for j, col in enumerate(row):
            assert col == 1 if i == j else col == 0


def test_transpose_square():
    """
    Check transpose of square 3x3 matrix
    :return:
    """
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    A_transpose = Matrix([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    assert A.T == A_transpose


def test_transpose_empty():
    """
    Transpose of empty matrix equality
    :return:
    """
    A = Matrix([])
    assert A.T == A


def test_transpose_rectangular():
    """
    Check that dimensions are correctly switched when transposing rectangular
    3x2 matrix.
    :return:
    """
    A = Matrix([[1, 2],
                [3, 4],
                [5, 6]])

    A_transpose = Matrix([[1, 3, 5],
                          [2, 4, 6]])
    assert A.T == A_transpose
