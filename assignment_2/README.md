# Solve system of linear equation using Gaussian elimination

## How to use

`python3 gauss.py`

## How to run unit-tests

Elementary unit-tests are written in `test_gauss.py` and `test_matrix.py` to
ensure adherence to the specification. To run them use command:

`nosetests`

## Under the hood

In order to solve system of linear equation, class `Matrix` was written. Based on
nested lists, it provides similar indexing capabilities to `np.ndarray`. Basic
mathematical and comparison operations are also implemented, facilitating search
of the solution. Using it as an underlying structure, `solve_gauss` and `gauss_inv`
functions are provided to (1) solve system of linear equations and (2) find an
inverse of a matrix, respectfully.

`parse_assignment` function allows to get input as specified in the assignment,
`split_input` further divides input matrix into coefficient matrix `X` and the
corresponding vector of free terms `b`.
`output_result` takes the output from `solve_gauss` and formats it according to
the requirement in the assignment.

Using this class, some of the basic methods are implemented
in `matrix_tools.py`: Identity matrix (`eye`), `np.zeros` and `np.argmax` analogues.

Moreover, `MatrixParser.py` class is configured to enable string-to-Matrix conversion.