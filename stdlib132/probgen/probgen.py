from .. import latex, utils, random
from pathlib import Path
import numpy as np
import inspect
import sympy
from sympy import Matrix, Symbol

# silly but workable trick recommended by TerrierGPT
class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


_here = Path(__file__).parent


def prob_text(**kwargs):
    ident = inspect.stack()[1].function
    with open(_here / "problems" / (ident + ".txt"), "r") as f:
        return f.read().format(ident=ident, **kwargs)

def least_squares_solve(
        rows,
        cols,
        rank,
        seed=None,
        mat_low=None,
        mat_high=None,
        unit_scaling=False,
        consistent=False,
):
    seed = random.mk_seed(seed)
    rng = random.mk_rng(seed)
    if not consistent:
        a = random.rref(
            rows=rows,
            cols=cols,
            rank=rank,
            rng=rng,
            low=mat_low,
            high=mat_high,
        )
        random.scramble(
            a,
            rng=rng,
            unit_scaling=unit_scaling
        )
        b = random.int_matrix(
            rows=rows,
            cols=1,
            rng=rng
        )
    else:
        _a = random.rref(
            rows=rows,
            cols=cols + 1,
            rank=rank,
            force_consistent=True,
            rng=rng,
            low=mat_low,
            high=mat_high,
        )
        random.scramble(
            _a,
            rng=rng,
            unit_scaling=unit_scaling,
        )
        a = _a[:, :-1]
        b = _a[:, -1]
    print(a.T @ a)
    print(a.T @ b)
    print(Matrix.hstack(a.T @ a, a.T @ b).rref())
    return prob_text(
        seed=seed,
        mat=latex.matrix(a),
        vec=latex.matrix(b),
    )

def project_onto_orthonormal(
        dim,
        num,
        seed=None,
        vec_low=None,
        vec_high=None,
        vec_set_low=None,
        vec_set_high=None,
):
    seed = random.mk_seed(seed)
    rng = random.mk_rng(seed)
    vec = random.int_matrix(
        rows=dim,
        cols=1,
        seed=seed,
        low=vec_low,
        high=vec_high,
    )
    vec_set = random.orthogonal_set(
        num=num,
        dim=dim,
        rng=rng,
        low=vec_set_low,
        high=vec_set_high,
    )
    vec_set = [v.normalized() for v in vec_set]

    return prob_text(
        seed=seed,
        vec=latex.matrix(vec),
        vec_set=latex.matrix(vec_set),
    )


def project_onto_vector(
        dim,
        seed=None,
):
    seed = random.mk_seed(seed)
    vecs = random.int_matrix(
        rows=dim,
        cols=2,
        seed=seed,
    )
    return prob_text(
        seed=seed,
        u=latex.matrix(vecs[:,0]),
        v=latex.matrix(vecs[:,1]),
    )


def in_terms_of_orthonormal(
        dim,
        seed=None,
        basis_low=None,
        basis_high=None,
        vec_low=None,
        vec_high=None,
):
    seed = random.mk_seed(seed)
    rng = random.mk_rng(seed)
    basis = random.orthogonal_set(
        num=dim,
        dim=dim,
        rng=rng,
        low=basis_low,
        high=basis_high,
    )
    basis = [v.normalized() for v in basis]
    vec = random.int_matrix(
        rows=dim,
        cols=1,
        rng=rng,
        low=vec_low,
        high=vec_high,
    )
    return prob_text(
        seed=seed,
        vec=latex.matrix(vec),
        basis=latex.matrix(basis),
    )

def determine_orthogonal(
        num,
        dim,
        answer,
        seed=None,
        low=None,
        high=None,
):
    seed = random.mk_seed(seed)
    rng = random.mk_rng(seed)
    vecs = random.orthogonal_set(
        num,
        dim,
        seed=seed,
        low=low,
        high=high
    )
    if answer == False: # TODO: Fix this
        while utils.is_orthogonal(*vecs):
            i = rng.randint(0, num - 1)
            j = rng.randint(0, dim - 1)
            k = rng.randint(low, high)
            vecs[i][j] = k
    return prob_text(
        seed=seed,
        vecs=latex.vector_set(vecs),
    )


def col_null_given_rref(
        rows,
        cols,
        rank=None,
        seed=None,
        rref_low=None,
        rref_high=None,
):
    seed = random.mk_seed(seed)
    rref = random.rref(
        rows,
        cols,
        rank=rank,
        seed=seed,
        low=rref_low,
        high=rref_high,
    )
    mat = Matrix([Symbol(f'\\mathbf a_{{{i + 1}}}') for i in range(cols)]).T
    return prob_text(
        seed=seed,
        mat=latex.matrix(mat),
        rref=latex.matrix(rref),
    )


def coord_vec(
        num_vecs,
        dim,
        seed=None,
        rref_low=None,
        rref_high=None,
        scramble_low=None,
        scramble_high=None,
):
    seed = random.mk_seed(seed)
    rng = random.mk_rng(seed)
    mat = random.rref(
        rows=dim,
        cols=num_vecs + 1,
        rank=num_vecs,
        force_consistent=True,
        rng=rng,
        low=rref_low,
        high=rref_high,
    )
    random.scramble(
        mat,
        rng=rng,
        low=scramble_low,
        high=scramble_high,
    )
    vec = mat[:,-1]
    vec_latex = latex.matrix(vec)
    basis = [mat[:,i] for i in range(num_vecs)]
    basis_latex = latex.set([latex.matrix(vec) for vec in basis])
    return prob_text(
        seed=seed,
        vec=vec_latex,
        vec_set=basis_latex,
    )


def eigenvals(
        dim,
        seed=None,
        diag_low=None,
        diag_high=None,
        scramble_low=None,
        scramble_high=None,
):
    seed = random.mk_seed(seed)
    rng = random.mk_rng(seed)
    d = random.int_matrix(
        rows=dim,
        cols=dim,
        low=diag_low,
        high=diag_high,
        kind='diag',
        rng=rng,
    )
    l = Matrix.eye(2)
    random.scramble(
        l,
        unit_scaling=True,
        low=scramble_low,
        high=scramble_high,
        rng=rng,
    )
    mat = l @ d @ l.inv()
    return prob_text(
        seed=seed,
        mat=latex.matrix(mat),
    )


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

def change_of_basis(
        num_vecs,
        dim,
        seed=None
):
    seed = random.mk_seed(seed)
    rng = random.mk_rng(seed)
    mat = random.rref(
        rows=dim,
        cols=num_vecs,
        rank=num_vecs,
        rng=rng
    )
    random.scramble(
        mat,
        rng=rng,
        unit_scaling=True,
    )
    basis_latex = latex.set([latex.matrix(mat[:,i]) for i in range(num_vecs)])
    return prob_text(
        seed=seed,
        vecs=basis_latex,
    )

def diag_by_poly(mat, poly, seed):
    return prob_text(
        seed=seed,
        mat=sympy.latex(mat),
        poly=sympy.latex(poly),
    )

def pair_len_angle_dist(vecs, seed):
    return prob_text(
        seed=seed,
        vecs=latex.vec_set(vecs),
    )


def normalization(
        dim,
        low=None,
        high=None,
        seed=None,

):
    seed = random.mk_seed(seed)
    v = random.int_matrix(
        rows=dim,
        cols=1,
        low=low,
        high=high,
        seed=seed
    )
    return prob_text(
        seed=seed,
        vec=latex.matrix(v),
    )


def diagonalization(
        dim,
        low=None,
        high=None,
        seed=None,
):
    seed = random.mk_seed(seed)
    rng = random.mk_rng(seed)
    d = random.int_matrix(
        rows=dim,
        cols=dim,
        low=low,
        high=high,
        kind='diag',
        rng=rng,
    )
    p = random.rref(
        rows=dim,
        cols=dim,
        rank=dim,
        low=low,
        high=high,
        rng=rng,
    )
    random.scramble(
        p,
        low=low,
        high=high,
        rng=rng,
        unit_scaling=True,
    )
    mat_latex = latex.matrix(p @ d @ p.inv())
    return prob_text(
        seed=seed,
        mat=mat_latex,
    )

def diagonalization_quiz(
        dim,
        low=None,
        high=None,
        seed=None,
):
    seed = random.mk_seed(seed)
    rng = random.mk_rng(seed)
    d = random.int_matrix(
        rows=dim,
        cols=dim,
        low=low,
        high=high,
        kind='diag',
        rng=rng,
    )
    p = random.rref(
        rows=dim,
        cols=dim,
        rank=dim,
        low=low,
        high=high,
        rng=rng,
    )
    random.scramble(
        p,
        low=low,
        high=high,
        rng=rng,
        unit_scaling=True,
    )
    mat_latex = latex.matrix(p @ d @ p.inv())
    return prob_text(
        seed=seed,
        mat=mat_latex,
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

def matrix_from_lin_trans(mat: np.ndarray, vecs: np.ndarray, seed: int) -> str:
    """Determine the matrix of a linear transformation given a
    collection of images.

    Parameters
    ----------
    mat : np.ndarray
        Matrix for the linear transformation.
    vecs : np.ndarray
        matrix used to generate inputs to the function.
    seed : int
        Seed used to generate `aug`.

    Returns
    -------
    str
        Problem statement as defined in matrix_from_lin_trans.txt.

    """
    assert len(mat.shape) == 2
    assert len(vecs.shape) == 2
    assert mat.shape[1] == vecs.shape[0]

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
    seed :
        Seed used for generating matrices.

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
    seed :
        Seed used for generating the given matrix.

    Returns
    -------
    str
        Problem statement as defined in determine_inv.txt

    """
    return prob_text(
        seed=seed,
        mat=latex.bmatrix(mat),
    )

def matrix_impl_row_ops(shape: tuple[int, int], ops: list[tuple[str, int, int]], seed: int) -> str:
    """Determine the matrix implementing a sequence of row operations.

    Parameters
    ----------
    shape : tuple(int, int)
        Shape of the expected matrix.
    ops : list[tuple[str, int, int]]
        List of row operations.
    seed :
        Seed used for generating row operations.

    Returns
    -------
    str
        Problem statement as defined in matrix_impl_row_ops.txt

    """
    return prob_text(
        seed=seed,
        m=shape[0],
        n=shape[1],
        row_ops=latex.row_ops(ops)
    )

def determine_transform_inv(mat: np.ndarray, seed: int) -> str:
    """Determine the inverse of a linear transformation.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix for the given transformation.
    seed :
        Seed used for generating the `matrix`.

    Returns
    -------
    str
        Problem statement as defined in determine_transform_inv.txt

    """
    return prob_text(
        seed=seed,
        transform=latex.lin_transform(mat),
    )

def determine_lu(mat: np.ndarray, seed: int) -> str:
    """Determine an LU factorization.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix used in the problem.
    seed :
        Seed used for generating the `matrix`.

    Returns
    -------
    str
        Problem statement as defined in determine_lu.txt

    """
    return prob_text(
        seed=seed,
        mat=latex.bmatrix(mat),
    )

def inv_from_row_ops(shape, ops, seed):
    """Determine the inverse based on row operations.

    Parameters
    ----------
    shape : tuple(int, int)
        Shape of the expected matrix.
    ops : list[tuple[str, int, int]]
        List of row operations.
    seed :
        Seed used for generating row operations.

    Returns
    -------
    str
        Problem statement as defined in inv_from_row_ops.txt

    """
    return prob_text(
        seed=seed,
        m=shape[0],
        n=shape[1],
        row_ops=latex.row_ops(ops)
    )

def one_to_one_onto_lin_trans(mat, seed):
    """Determine if linear tranformation is 1-1/onto.

    Parameters
    ----------
    mat : numpy.ndarray
        Matrix for the transformation.
    seed : int
        Seed used to generate `mat`.

    Returns
    -------
    str
        Problem statement as defined in one_to_one_onto_lin_trans.txt.

    """
    return prob_text(
        seed=seed,
        lin_trans=latex.lin_transform(mat),
    )


def col_null(mat, seed):
    """Determine column/null space of matrix.

    Parameters
    ----------
    mat : numpy.ndarray
        Matrix for problem.
    seed : int
        Seed used to generate `mat`.

    Returns
    -------
    str
        Problem statement as defined in col_null.txt.
    """
    return prob_text(
        seed=seed,
        mat=latex.bmatrix(mat),
    )


def find_basis(vecs, seed):
    """Find basis of subspace.

    Parameters
    ----------
    vecs : numpy.ndarray
        Matrix containing vectors
    seed : int
        Seed used to generate `vecs`.

    Returns
    -------
    str
        Problem statement as defined in find_basis.txt.
    """
    return prob_text(
        seed=seed,
        span=latex.span(vecs),
    )

def lu_fact(mat, seed):
    """Determine the LU factorization.

    Parameters
    ----------
    mat : numpy.ndarray
        matrix to factor.
    seed : int
        Seed used to generate `mat`

    Returns
    -------
    str
        Problem statement as defined in lu_fact.txt.
    """
    return prob_text(
        seed=seed,
        mat=latex.bmatrix(mat),
    )


def state_diag(mat, seed):
    """Draw a state diagram

    Parameters
    ----------
    mat : numpy.ndarray
        Transition matrix.
    seed : int
        Seed used to generate `mat`

    Returns
    -------
    str
        Problem statement as defined in state_diag.txt.
    """
    return prob_text(
        seed=seed,
        mat=latex.bmatrix(mat),
    )


def steady_state(mat, seed):
    """Determine steady state.

    Parameters
    ----------
    mat : numpy.ndarray
        Transition matrix.
    seed : int
        Seed used to generate `mat`

    Returns
    -------
    str
        Problem statement as defined in steady_state.txt.
    """
    return prob_text(
        seed=seed,
        mat=latex.bmatrix(mat),
    )

def eigen(mat, vec, seed):
    return prob_text(
        seed=seed,
        mat=latex.bmatrix(mat),
        vec=latex.bmatrix(vec),
    )

def eigenspace(mat, lamb, seed):
    return prob_text(
        seed=seed,
        mat=latex.bmatrix(mat),
        lamb=lamb,
    )


def det(mat, seed):
    return prob_text(
        seed=seed,
        mat=latex.bmatrix(mat),
    )

def det_inv(mat, seed):
    return prob_text(
        seed=seed,
        mat=latex.bmatrix(mat),
    )


def determine_coefficient_augmented_matrix(
        num_eqs,
        num_vars,
        seed,
):
    seed = random.mk_seed(seed)
    aug = random.int_matrix(
        rows=num_eqs,
        cols=num_vars + 1,
        seed=seed,
    )
    return prob_text(
        seed=seed,
        lin_sys=latex.lin_sys(aug),
    )


def determine_linear_system(aug, seed):
    return prob_text(
        seed=seed,
        aug_matrix=latex.bmatrix(aug),
    )


def determine_unique_solution_linear_system(
        num_eqs,
        seed=None
):
    seed = random.mk_seed(seed)
    rng = random.mk_rng(seed)
    aug = random.rref(
        rows=num_eqs,
        cols=num_eqs + 1,
        force_consistent=True,
        rank=num_eqs,
        rng=rng,
    )
    random.scramble(
        aug,
        rng=rng
    )
    return prob_text(
        seed=seed,
        lin_sys=latex.lin_sys(aug),
    )


def verify_solution_linear_system(
        num_eqs,
        num_vars,
        seed=None,
):
    seed = random.mk_seed(seed)
    rng = random.mk_rng(seed)
    aug = random.rref(
        rows=num_eqs,
        cols=num_vars + 1,
        force_consistent=True,
        rng=rng,
    )
    print(sympy.pretty(aug))
    random.scramble(
        aug,
        rng=rng,
    )
    sol = (1, 2, 3)
    return prob_text(
        seed=seed,
        sol=str(sol),
        lin_sys=latex.lin_sys(aug),
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


def particular_sol(
        num_rows,
        num_cols,
        rref, seed):
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
