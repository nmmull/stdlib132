import math
import secrets
import random
import numpy as np
import sympy
from sympy import Matrix, MatrixBase


def seed_rng(seed=None):
    if seed is None:
        seed = secrets.randbits(32)
        print(seed)
    return (seed, random.Random(seed))


def rng(seed=None):
    return seed_rng(seed)[1]


def nz_int(
        low=None,
        high=None,
        rng=None,
):
    rng = rng() if rng is None else rng
    low = -5 if low is None else low
    high = 5 if high is None else high
    n = rng.randint(low, high - 1)
    return n + 1 if n >= 0 else n


def int_matrix(
        rows,
        cols,
        rng=None,
        low=None,
        high=None,
):
    rng = rng() if rng is None else rng
    low = -10 if low is None else low
    high = 10 if high is None else high
    return sympy.randMatrix(
        rows,
        cols,
        min=low,
        max=high,
        prng=rng,
    )


def rref(
        rows,
        cols,
        rng=None,
        low=None,
        high=None,
        rank=None,
        force_consistent=None,
        force_first_column=None,
):
    rng = rng() if rng is None else rng
    low = -6 if low is None else low
    high = 6 if high is None else high
    force_consistent = False if force_consistent is None else force_consistent
    force_first_column = True if force_first_column is None else force_first_column
    if rank is None:
        cap = min(rows, cols) + 1
        if force_consistent:
            cap = min(rows + 1, cols)
        if force_first_column:
            cap -= 1
        rank = rng.randint(0, cap - 1)
        if force_first_column:
            rank += 1
    a = Matrix.zeros(rows, cols)
    max_pivot = cols
    num_pivots = rank
    if force_consistent:
        max_pivot -= 1
    if force_first_column:
        max_pivot -= 1
        num_pivots -= 1
    pivot_cols = rng.sample([i for i in range(max_pivot)], num_pivots)
    if force_first_column:
        pivot_cols = [0] + list(map(lambda x: x + 1, pivot_cols))
    pivot_cols.sort()
    for i in range(rank):
        pivot_pos = pivot_cols[i]
        zeros = [0 for _ in range(pivot_pos + 1)]
        rands = [rng.randint(low, high) for _ in range(cols - pivot_pos - 1)]
        a[i,:] = Matrix(zeros + rands).T
    eye = Matrix.eye(rows, rank)
    for i in range(rank):
        a[:, pivot_cols[i]] = eye[:, i]
    return a


def scramble(
        a,
        rng=None,
        low=None,
        high=None,
        force_unit_scaling=None,
):
    rng = seed_rng() if rng is None else rng
    low = -3 if low is None else low
    high = 3 if high is None else high
    force_unit_scaling = False if force_unit_scaling is None else force_unit_scaling
    assert low <= high
    def scale_factor():
        negate = 2 * rng.randint(0, 1) - 1
        if unit_scaling:
            return negate
        return negate * (abs(rng.randint(low + 1, high - 1)) + 1)
    for i in range(a.rows):
        for prev_i in range(i):
            a[prev_i,:] += rng.randint(low, high) * a[i,:]
    for i in range(a.rows - 1, -1, -1):
        for next_i in range(i + 1, a.rows):
            a[next_i,:] *= scale_factor()
            a[next_i,:] += rng.randint(low, high) * a[i,:]
    a[0,:] *= scale_factor()


def scrambled(
        rows,
        cols,
        rng=None,
        low_rref=None,
        high_rref=None,
        low_scramble=None,
        high_scramble=None,
        rank=None,
        force_consistent=None,
        force_first_column=None,
        force_unit_scaling=None,
):
    a = rref(
        rows=rows,
        cols=cols,
        rng=rng,
        low=low_rref,
        high=high_rref,
        rank=rank,
        force_consistent=force_consistent,
        force_first_column=force_first_column,
    )
    scramble(
        a,
        low=low_scramble,
        high=high_scramble,
        force_unit_scaling=force_unit_scaling,
    )
    return a


def swap_row_op(
        rows,
        rng=None,
):
    rng = rng() if rng is None else rng
    [i, j] = rng.sample([i + 1 for i in range(rows)])
    return ("swap", i, j)


def scale_row_op(
        rows,
        rng=None,
        low=None,
        high=None,
):
    rng = rng() if rng is None else rng
    c = nz_int(
        low=low,
        high=high,
        rng=rng,
    )
    return ("scale", rng.randint(1, rows), c)


def scale_replace_op(
        rows,
        rng=None,
        low=None,
        high=None,
):
    rng = rng() if rng is None else rng
    [i, j] = rng.sample([i + 1 for i in range(rows)])
    c = nz_int(
        low=low,
        high=high,
        rng=rng
    )
    return ("scale", i, c, j)


def row_op(
        rows,
        rng=None,
        low=None,
        high=None,
):
    rng = rng() if rng is None else rng
    kind = rnd.randint(0, 2)
    if kind == 0:
        swap_row_op(rows, rng=rng)
    elif kind == 1:
        scale_row_op(rows, low=low, high=high, rng=rng)
    elif kind == 2:
        replace_row_op(rows, low=low, high=high, rng=rng)


def row_ops(
        rows,
        num,
        rng=None,
        low=None,
        high=None,
):
    rng = rng() if rng is None else rng
    return [row_op(rows=rows, rng=rng, seed=seed, low=low, high=high) for _ in range(num)]


# TODO: Rewrite
# def orthogonal_set(
#         num: int,
#         dim: int,
#         rng: random.Random | None = None,
#         seed: int | None = None,
#         low: int = None,
#         high: int = None,
# ) -> list[MatrixBase]:
#     assert num <= dim
#     A = int_matrix(
#         rows=dim,
#         cols=num,
#         rng=rng,
#         seed=seed,
#         low=low,
#         high=high,
#     )
#     vs = sympy.Matrix.orthogonalize(*[A[:,i] for i in range(A.cols)])
#     entries = []
#     for v in vs:
#         entries += map(lambda x: x.as_numer_denom()[1], [x for x in v])
#     return [v * math.lcm(*entries) for v in vs]
