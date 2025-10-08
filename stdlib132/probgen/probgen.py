from .. import latex, utils
from pathlib import Path
import numpy as np
import inspect


# silly but workable trick recommended by TerrierGPT
class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


here = Path(__file__).parent


def prob_text(**kwargs):
    ident = inspect.stack()[1].function
    with open(here / (ident + ".txt"), "r") as f:
        return f.read().format(ident=ident, **kwargs)


def standalone(problems: list[str]) -> str:
    """Standalone latex file with problems, primarily for testing.

    Parameters
    ----------
    problems : list[str]
        The collection of problems that will be included in the output
        string.

    Returns
    -------
    str
        String for a standalong latex file with given problems.
    """
    out = ""
    for problem in problems:
        out += f"\\item {problem}\n"
    return prob_text(
        problems=out,
    )


def dom_cod_mat(a: np.ndarray, seed: int) -> str:
    """Determine domain, codomain, and underyling matrix.

    Parameters
    ----------
    a : np.array
        Matrix defining the linear transformation.
    seed : int
        Seed used to generate `a`.

    Returns
    -------
    str
        Problem statement as defined in `dom_cod_mat.txt`

    """
    return prob_text(
        seed=seed,
        lin_trans=latex.lin_transform(a),
    )

def in_range(aug: np.ndarray, seed: int) -> str:
    """Determine if vector is in range.

    Parameters
    ----------
    aug : np.ndarray
        Augmented matrix used for the problem.  We take the vector to
        be `aug[:,-1]` and the matrix to be `aug[:,:-1]`.
    seed: int
        Seed used to generate `aug`.

    Returns
    -------
    str
        Problem statement as defined in `in_range.txt`.

    """
    mat_vec = latex.mat_set(
        [
            ("A", aug[:, :-1]),
            ("\\mathbf v", aug[:, -1]),
        ]
    )
    return prob_text(
        seed=seed,
        mat_vec=mat_vec
    )


def compute_lin_trans(vecs: np.ndarray, coeffs : np.ndarray, seed: int) -> str:
    """Compute linear transformation on given input.

    Parameters
    ----------
    vecs : np.ndarray
        Vectors used as inputs to the linear transformation.
    coeffs : np.ndarray
        Values used in linear combination.  `coeffs` should have
        `vecs.shape[1]` entries.

    Returns
    -------
    str
        Problem statement as defined in `compute_lin_trans.txt`.

    """
    assert len(vecs.shape) == 2
    assert len(coeffs.shape) == 1
    assert vecs.shape[1] == coeffs.shape[0]
    var_names = [f"\\mathbf v_{{{i + 1}}}" for i in range(vecs.shape[1])]
    apps = list(map(lambda s: f"T({s})", var_names))
    vecs = [vecs[:,i] for i in range(vecs.shape[1])]
    out = latex.mat_set(list(zip(apps, vecs)))
    lin_comb = latex.lin_comb(
        coeffs,
        var_names,
        "\\mathbf 0"
    )
    return prob_text(
        seed=seed,
        lin_comb=lin_comb,
        out=out,
    )

def lin_trans_out_from_in(mat: np.ndarray, aug: np.ndarray, seed: int) -> str:
    """Determine the image given a collection of images.

    Parameters
    ----------
    mat : np.ndarray
        Matrix for the linear transformation.
    aug : np.ndarray
        Augmented matrix used to generate inputs to the function.
    seed : int
        Seed used to generate `aug`.

    Returns
    -------
    str
        Problem statement as defined in lin_trans_out_from_in.txt.

    """
    assert len(mat.shape) == 2
    assert len(aug.shape) == 2
    assert mat.shape[1] == aug.shape[0]
    vecs = aug[:,:-1]
    rhs = aug[:,-1]

    in_strs = [f"T\\left({latex.bmatrix(vecs[i,:])}\\right)" for i in range(vecs.shape[1])]
    outs = [mat @ vecs[i,:] for i in range(vecs.shape[1])]
    in_outs = list(zip(in_strs, outs))
    return prob_text(
        seed=seed,
        outputs=latex.mat_set(in_outs),
        vec=latex.bmatrix(rhs),
    )

def matrix_from_lin_trans(mat: np.ndarray, aug: np.ndarray, seed: int) -> str:
    """Determine the matrix of a linear transformation given a
    collection of images.

    Parameters
    ----------
    mat : np.ndarray
        Matrix for the linear transformation.
    aug : np.ndarray
        Augmented matrix used to generate inputs to the function.
    seed : int
        Seed used to generate `aug`.

    Returns
    -------
    str
        Problem statement as defined in matrix_from_lin_trans.txt.

    """
    assert len(mat.shape) == 2
    assert len(aug.shape) == 2
    assert mat.shape[1] == aug.shape[0]
    vecs = aug[:,:-1]
    rhs = aug[:,-1]

    in_strs = [f"T\\left({latex.bmatrix(vecs[i,:])}\\right)" for i in range(vecs.shape[1])]
    outs = [mat @ vecs[i,:] for i in range(vecs.shape[1])]
    in_outs = list(zip(in_strs, outs))
    return prob_text(
        seed=seed,
        outputs=latex.mat_set(in_outs),
    )


def one_to_one_onto_matrix_trans(mat: np.ndarray, seed: int) -> str:
    """Determine if matrix tranformation is 1-1/onto.

    Parameters
    ----------
    mat : numpy.npdarray
        Matrix for the transformation.
    seed: int
        Seed used to generate `mat`.

    Returns
    -------
    str
        Problem statement as defined in one_to_one_onto_matrix_trans.txt.

    """
    return prob_text(
        seed=seed,
        mat=latex.bmatrix(mat),
    )


def draw_unit_square(mat: np.ndarray, seed: int) -> str:
    """Determine if matrix tranformation is 1-1/onto.

    Parameters
    ----------
    mat : numpy.ndarray
        Matrix for the transformation.  It must be a 2 by 2 matrix.
    seed : int
        Seed used to generate `mat`.

    Returns
    -------
    str
        Problem statement as defined in draw_unit_square.txt.

    """
    return prob_text(
        seed=seed,
        mat=latex.bmatrix(mat),
    )


def compute_lin_comb_mat(coeffs : np.ndarray, mats : list[np.ndarray], seed : int) -> str:
    """Compute linear combination of matrices

    Parameters
    ----------
    mats : numpy.ndarray
        Matrices in linear combination.
    seed : int
        Seed use to generate mats.

    Returns
    -------
    str
        Problem statement as defined in compute_lin_comb_mat.txt.
    """
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWZYZ")
    mats = [(letters[i], mats[i]) for i in range(len(mats))]
    return prob_text(
        seed=seed,
        mat_lin_comb=latex.lin_comb(coeffs, letters, "\\mathbf{0}"),
        mats=latex.mat_set(mats),
    )

def compute_mat_mul(mat1: np.ndarray, mat2: np.ndarray, seed: int) -> str:
    """Compute matrix multiplication

    Parameters
    ----------
    mat1 : numpy.ndarray
        First matrix in multiplication.
    mat2 : numpy.ndarray
        Second matrix in multiplication.
    seed :
        Seed used for generating matrices.

    Returns
    -------
    str
        Problem statement as defined in compute_mat_mul.txt.

    """
    return prob_text(
        seed=seed,
        mats=latex.mat_set([("A", mat1), ("B", mat2)]),
    )

def use_inverse(inv_mat: np.ndarray, vecs: list[np.ndarray], seed: int) -> str:
    """Use the inverse to solve several matrix equations.

    Parameters
    ----------
    inv_mat : numpy.ndarray
        Inverse matrix use to solve each system.
    vecs : list[numpy.ndarray]
        The collections of right-hand sides to the matrix equations.

    Returns
    -------
    str
        Problem statement as defined in compute_mat_mul.txt.

    """
    vec_names = [f"\\mathbf b_{{{i + 1}}}" for i in range(len(vecs))]
    mat_vecs = [("A^{-1}", inv_mat)] + list(zip(vec_names, vecs))
    return prob_text(
        seed=seed,
        mat_vecs=latex.mat_set(mat_vecs),
    )

def determine_inv(mat: np.ndarray, seed: int) -> str:
    """Determine the inverse of a matrix.

    Parameters
    ----------
    mat : numpy.ndarray
       Matrix to invert.

    Returns
    -------
    str
        Problem statement as defined in determin_inv.txt

    """
    return prob_text(
        seed=seed,
        mat=latex.bmatrix(mat),
    )


def determine_coefficient_augmented_matrix(aug, seed):
    return prob_text(
        seed=seed,
        lin_sys=latex.lin_sys(aug),
    )


def determine_linear_system(aug, seed):
    return prob_text(
        seed=seed,
        aug_matrix=latex.bmatrix(aug),
    )


def determine_unique_solution_linear_system(aug, seed):
    assert np.linalg.matrix_rank(aug) == aug.shape[0]
    return prob_text(
        seed=seed,
        lin_sys=latex.int_lin_sys(aug),
    )


def verify_solution_linear_system(sol, aug, seed):
    return prob_text(
        seed=seed,
        sol=latex.solution(sol),
        lin_sys=latex.int_lin_sys(aug),
    )


def apply_row_ops(row_ops, a, ops_seed, mat_seed):
    return prob_text(
        ops_seed=ops_seed,
        mat_seed=mat_seed,
        row_ops=latex.row_ops(row_ops),
        matrix=latex.bmatrix(a),
    )


def row_ops_pair_transform(ops, mat, seed):
    b = np.copy(mat)
    utils.apply_row_ops(ops, b)
    return prob_text(
        seed=seed,
        matrix1=latex.bmatrix(mat),
        matrix2=latex.bmatrix(b),
    )


def gen_form_sol_rref(rref, seed):
    return prob_text(
        seed=seed,
        rref=latex.bmatrix(rref),
    )


def gen_form_sol_lin_sys(aug, seed):
    return prob_text(
        seed=seed,
        lin_sys=latex.int_lin_sys(aug),
    )


def gen_form_sol_mat_eq(aug, seed):
    mat_vec = latex.mat_set(
        [
            ("A", aug[:, :-1]),
            ("\\mathbf b", aug[:, -1]),
        ]
    )
    return prob_text(
        seed=seed,
        mat_vec=mat_vec,
    )


def determine_rref(matrix, seed):
    return prob_text(
        seed=seed,
        matrix=latex.bmatrix(matrix),
    )


def alt_gen_form(rref, seed):
    return prob_text(seed=seed, gen_form=latex.gen_form_sol(rref))


def particular_sol(rref, seed):
    return prob_text(
        seed=seed,
        rref=latex.bmatrix(rref),
    )


def compute_lin_comb_vec(coeffs, vecs, seed):
    return prob_text(seed=seed, lin_comb_vec=latex.lin_comb_vec(coeffs, vecs))


def equiv_vector_eq(aug, seed):
    return prob_text(
        seed=seed,
        lin_sys=latex.int_lin_sys(aug),
    )


def in_span_of_two(matrix, seed):
    assert matrix.shape[1] == 2
    return prob_text(
        seed=seed,
        vec_set=latex.vec_set(matrix),
    )


def gen_form_sol_vec_eq(aug, seed):
    return prob_text(
        seed=seed,
        vec_eq=latex.vec_eq(aug),
    )


def vec_in_span(matrix, seed):
    return prob_text(seed=seed, vec_set=latex.vec_set(matrix))


def span_pair_vec(vecs, seed):
    assert vecs.shape == (3, 2)
    return prob_text(seed=seed, vec_set=latex.vec_set(vecs))


def compute_mat_vec_mul(mat, vec, mat_seed, vec_seed):
    mat_vec = latex.mat_set(
        [
            ("A", mat),
            ("\\mathbf v", vec),
        ]
    )
    return prob_text(
        mat_seed=mat_seed,
        vec_seed=vec_seed,
        mat_vec=mat_vec,
    )


def col_full_span(mat, seed):
    return prob_text(
        seed=seed,
        n=mat.shape[0],
        mat=latex.bmatrix(mat),
    )


def determine_lin_dep(vecs, seed):
    return prob_text(
        seed=seed,
        vecs=latex.vec_set(vecs),
    )


def determine_all_lin_ind(vecs, seed):
    return prob_text(
        seed=seed,
        vecs=latex.vec_set(vecs),
    )
