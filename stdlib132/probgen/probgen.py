from .. import latex, utils
from pathlib import Path
import numpy as np
import inspect

here = Path(__file__).parent

# silly but workable trick recommended by TerrierGPT
class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"

def prob_text(**kwargs):
    ident = inspect.stack()[1].function
    with open (here / (ident + '.txt'), 'r') as f:
        return f.read().format(ident=ident, **kwargs)

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
        lin_sys=latex.lin_sys(aug),
    )

def verify_solution_linear_system(sol, aug, seed):
    return prob_text(
        seed=seed,
        sol=latex.solution(sol),
        lin_sys=latex.lin_sys(aug),
    )

def apply_row_ops(row_ops, a, ops_seed, mat_seed):
    return prob_text(
        ops_seed=ops_seed,
        mat_seed=mat_seed,
        row_ops=latex.row_ops(row_ops),
        matrix=latex.bmatrix(a),
    )

def row_ops_pair_transform(ops, mat, ops_seed, mat_seed):
    b = np.copy(mat)
    utils.apply_row_ops(ops, b)
    return prob_text(
        ops_seed=ops_seed,
        mat_seed=mat_seed,
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
        lin_sys=latex.lin_sys(aug),
    )

def gen_form_sol_mat_eq(aug, seed):
    mat_vec = latex.mat_set([
        ("A", aug[:,:-1]),
        ("\\mathbf b", aug[:,-1]),
    ])
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
    return prob_text(
        seed=seed,
        gen_form=latex.gen_form_sol(rref)
    )

def particular_sol(rref, seed):
    return prob_text(
        seed=seed,
        rref=latex.bmatrix(rref),
    )

def compute_lin_comb_vec(coeffs, vecs, seed):
    return prob_text(
        seed=seed,
        lin_comb_vec=latex.lin_comb_vec(coeffs, vecs)
    )

def equiv_vector_eq(aug, seed):
    return prob_text(
        seed=seed,
        lin_sys=latex.lin_sys(aug),
    )

def in_span_of_two(matrix, seed):
    assert(matrix.shape[1] == 2)
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
    return prob_text(
        seed=seed,
        vec_set=latex.vec_set(matrix)
    )

def span_pair_vec(vecs, seed):
    assert vecs.shape == (3, 2)
    return prob_text(
        seed=seed,
        vec_set=latex.vec_set(vecs)
    )

def compute_mat_vec_mul(mat, vec, mat_seed, vec_seed):
    mat_vec = latex.mat_set([
        ("A", mat),
        ("\\mathbf v", vec),
    ])
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
