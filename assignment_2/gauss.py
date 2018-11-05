import numpy as np
from Matrix import Matrix
from matrix_tools import eye


def solve_gauss(A, b):
    """
    Solve SOLE using Gaussian elimination

    Available operations:
    1) Swapping two rows
    2) Multiplying a row by a nonzero number
    3) Adding a multiple of one row another row
    :param A:
    :return: A
    """
    nrow, ncol = A.shape if len(A.shape) == 2 else (A.shape, None)
    X = A[:, :]  # Make a copy (needed for np.array)
    b = b[:, :]  # Make a copy
    for i in range(min((nrow, ncol))):
        if X[i, i] == 0:  # find row with non-zero on the pivot place, swap
            for j in range(i + 1, nrow):
                if X[j, i] != 0:
                    X[i, :], X[j, :] = X[j, :], X[i, :]
                    b[i, :], b[j, :] = b[j, :], b[i, :]
                    break
        if X[i, i] != 1 and X[i, i] != 0:
            k = X[i, i]  # scale factor
            X[i, :] /= k  # scale coefficients
            b[i, :] /= k  # scale free term

        for j in range(i + 1, nrow):  # Remove corresponding coef. in other equations
            if X[j, i] != 0:
                k = X[j, i]  # scale factor
                X[j, :] -= X[i, :] * k
                b[j, :] -= b[i, :] * k
        for j in range(i - 1, -1, -1):  # remove coef. above the main diagonal
            if X[j, i] != 0:
                k = X[j, i]
                X[j, :] -= X[i, :] * k
                b[j, :] -= b[i, :] * k
    return X, b


def gauss_inv(A):
    """
    Calculate inverse of matrix using Gaussian elimination on matrix A,
    expanded with identity matrix
    :param A:
    :return:
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Inverse of non-square matrix")
    X, A_inv = solve_gauss(A, np.eye(A.shape[0]))
    return A_inv


def parse_assignment():
    """
    Parse format from the assignment (reading input from user, etc.)
    :return: Matrix
    """
    i = int(input())
    parsed = []
    for _ in range(i):
        parsed.append(list(map(float, input().split())))
    return Matrix(parsed)


def split_input(M):
    """
    Split input matrix into matrix of coefficients X and vector of constant
    terms b
    :param M:
    :return: X, b
    """
    b = M[:, -1]  # last column
    X = M[:, :-1]  # all except the last column
    return X, b


def output_result(X, b):
    """
    Check the result of gauss solve and output the result in proper format
    :param X:
    :param b:
    :return:
    """
    # if system has unique solution, we have Identity matrix in X
    e = eye(X.shape[0])
    if np.all(X.round(2) == e):
        print(" ".join([str(round(_, 5)) for _ in b]))
    else:
        # No solution or infinitely many solutions? Who cares?
        print(-1)


if __name__ == '__main__':
    X, b = split_input(parse_assignment())
    A, b_solved = solve_gauss(X, b.T)
    output_result(A, b_solved)
