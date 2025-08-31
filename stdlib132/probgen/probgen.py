from .. import latex, utils
from pathlib import Path
import numpy as np

here = Path(__file__).parent

def prob_text(ident):
    with open (here / (ident + '.txt'), 'r') as f:
        return f.read()

def determine_coefficient_augmented_matrix(aug, seed):
    ident = "determine_coefficient_augmented_matrix"
    return prob_text(ident).format(
        ident=ident,
        seed=seed,
        lin_sys=latex.lin_sys(aug),
    )

def determine_linear_system(aug, seed):
    ident = "determine_linear_system"
    return prob_text(ident).format(
        ident=ident,
        seed=seed,
        aug_matrix=latex.bmatrix(aug),
    )

def determine_unique_solution_linear_system(aug, seed):
    assert np.linalg.matrix_rank(aug) == aug.shape[0]
    ident = "determine_unique_solution_linear_system"
    return prob_text(ident).format(
        ident=ident,
        seed=seed,
        lin_sys=latex.lin_sys(aug),
    )

def verify_solution_linear_system(sol, aug, seed):
    ident = "verify_solution_linear_system"
    return prob_text(ident).format(
        ident=ident,
        seed=seed,
        sol=latex.solution(sol),
        lin_sys=latex.lin_sys(aug),
    )

def apply_row_ops(row_ops, a, ops_seed, mat_seed):
    ident = "apply_row_ops"
    return prob_text(ident).format(
        ident=ident,
        ops_seed=ops_seed,
        mat_seed=mat_seed,
        row_ops=latex.row_ops(row_ops),
        matrix=latex.bmatrix(a),
    )

def row_ops_pair_transform(ops, mat, ops_seed, mat_seed):
    ident = "row_ops_pair_transform"
    b = np.copy(mat)
    utils.apply_row_ops(ops, b)
    return prob_text(ident).format(
        ident=ident,
        ops_seed=ops_seed,
        mat_seed=mat_seed,
        matrix1=latex.bmatrix(mat),
        matrix2=latex.bmatrix(b),
    )

def gen_form_sol_rref(rref, seed):
    ident = "gen_form_sol_rref"
    return prob_text(ident).format(
        ident=ident,
        seed=seed,
        rref=latex.bmatrix(rref),
    )

def gen_form_sol_lin_sys(aug, seed):
    ident = "gen_form_sol_lin_sys"
    return prob_text(ident).format(
        ident=ident,
        seed=seed,
        lin_sys=latex.lin_sys(aug),
    )

def determine_rref(matrix, seed):
    ident = "determine_rref"
    return prob_text(ident).format(
        ident=ident,
        seed=seed,
        matrix=latex.bmatrix(matrix),
    )

def alt_gen_form(rref, seed):
    ident = "alt_gen_form"
    return prob_text(ident).format(
        ident=ident,
        seed=seed,
        gen_form=latex.gen_form_sol(rref)
    )
