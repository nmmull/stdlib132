from sympy import Matrix
from stdlib132 import latex

def test_lin_comb_basic(file_regression):
    coeffs = Matrix([1, 2, 3, 4, 5])
    result = latex.lin_comb(coeffs)
    file_regression.check(result)

def test_lin_comb_basic_list(file_regression):
    coeffs = [1, 2, 3, 4, 5]
    result = latex.lin_comb(coeffs)
    file_regression.check(result)

def test_lin_comb_basic_row(file_regression):
    coeffs = Matrix([[1, 2, 3, 4, 5]])
    result = latex.lin_comb(coeffs)
    file_regression.check(result)

def test_lin_comb_zero_case(file_regression):
    coeffs = [0 for _ in range(100)]
    result = latex.lin_comb(coeffs)
    file_regression.check(result)

def test_lin_comb_neg_one_first(file_regression):
    coeffs = [-1, 2, 3, 4, 5]
    result = latex.lin_comb(coeffs)
    file_regression.check(result)
