from sympy import *
from stdlib132 import latex

def test_lin_comb_basic(file_regression):
    coeffs = Matrix([1, 2, 3, 4, 5])
    result = latex.lin_comb(coeffs)
    file_regression.check(result)

