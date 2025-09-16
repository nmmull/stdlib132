import secrets
import numpy as np

def matrix(
        shape,
        rng=None,
        seed=None,
        low=-10,
        high=10,
):
    assert len(shape) == 2 and shape[0] >= 1 and shape[1] >= 1
    assert low <= high
    seed = secrets.randbits(32) if seed is None else seed
    rng = np.random.default_rng(seed) if rng is None else rng
    out = rng.integers(size=shape, low=low, high=high)
    return out, seed

def vector(
        num,
        rng=None,
        seed=None,
        low=-10,
        high=10,
):
    assert num  >= 1
    assert low <= high
    seed = secrets.randbits(32) if seed is None else seed
    rng = np.random.default_rng(seed) if rng is None else rng
    out = rng.integers(size=num, low=low, high=high)
    return out, seed




def lin_comb_vec(
        shape,
        rng=None,
        seed=None,
        low=-10,
        high=10,
):
    assert len(shape) == 2 and shape[0] >= 1 and shape[1] >= 1
    assert low <= high
    seed = secrets.randbits(32) if seed is None else seed
    rng = np.random.default_rng(seed) if rng is None else rng
    coeffs = rng.integers(size=shape[1], low=low, high=high)
    vecs = rng.integers(size=shape, low=low, high=high)
    return coeffs, vecs, seed

def rref(
        shape,
        rank=None,
        force_consistent=False,
        force_first_column=True,
        rng=None,
        seed=None,
        low=-6,
        high=6,
):
    assert len(shape) == 2 and shape[0] >= 1 and shape[1] >= 1
    if rank is not None:
        assert rank <= min(shape[0], shape[1])
    assert low <= high
    num_rows = shape[0]
    num_cols = shape[1]
    seed = secrets.randbits(32) if seed is None else seed
    rng = np.random.default_rng(seed) if rng is None else rng
    a = np.zeros(shape, dtype=np.int64)
    if rank is None:
        cap = min(num_rows, num_cols) + 1
        if force_consistent:
            cap = min(num_rows + 1, num_cols)
        if force_first_column:
            cap -= 1
        rank = rng.integers(cap)
        if force_first_column:
            rank += 1
    max_pivot = num_cols
    num_pivots = rank
    if force_consistent:
        max_pivot -= 1
    if force_first_column:
        max_pivot -= 1
        num_pivots -= 1
    pivot_cols = rng.choice(max_pivot, num_pivots, replace=False)
    if force_first_column:
        pivot_cols += 1
        pivot_cols = np.hstack((np.array([0]), pivot_cols))
    pivot_cols.sort()
    for i in range(rank):
        pivot_pos = pivot_cols[i]
        num_zeros = pivot_pos
        a[i] = np.hstack([
            np.zeros(pivot_pos + 1),
            rng.integers(low=low, high=high, size=num_cols - pivot_pos - 1),
        ])
    eye = np.eye(num_rows, rank)
    for i in range(rank):
        a[:,pivot_cols[i]] = eye[:,i]
    return a, seed

def scrambled(
        a,
        seed=None,
        rng=None,
        low=-3,
        high=3,
):
    assert low <= high
    assert len(a.shape) >= 2 and a.shape[0] >= 1 and a.shape[1] >= 1
    a = np.copy(a)
    num_rows = a.shape[0]
    num_cols = a.shape[1]
    seed = secrets.randbits(32) if seed is None else seed
    rng = np.random.default_rng(seed) if rng is None else rng
    for i in range(num_rows):
        for prev_i in range(i):
            a[prev_i] += rng.integers(low=low, high=high) * a[i]
    for i in range(num_rows - 1, -1, -1):
        for next_i in range(i + 1, num_rows):
            a[next_i] += rng.integers(low=low, high=high) * a[i]
    return a, seed

def simple_matrix(
        shape,
        rank=None,
        force_consistent=False,
        force_first_column=True,
        rng=None,
        seed=None,
        scramble_low=-3,
        scramble_high=3,
        rref_low=-6,
        rref_high=6,
):
    assert len(shape) == 2 and shape[0] >= 1 and shape[1] >= 1
    if rank is not None:
        assert rank <= min(shape[0], shape[1])
    assert scramble_low <= scramble_high
    assert rref_low <= rref_high
    seed = secrets.randbits(32) if seed is None else seed
    rng = np.random.default_rng(seed) if rng is None else rng
    a_rref, _ = rref(
        shape,
        rank=rank,
        force_consistent=force_consistent,
        force_first_column=force_first_column,
        rng=rng,
        low=rref_low,
        high=rref_high,
    )
    a, _ = scrambled(
        a_rref,
        rng=rng,
        low=scramble_low,
        high=scramble_high,
    )
    return a_rref, a, seed

def row_op(
        shape,
        rng=None,
        seed=None,
        low=-5,
        high=-5,
):
    assert len(shape) == 2 and shape[0] >= 1 and shape[1] >= 1
    assert low <= high
    seed = secrets.randbits(32) if seed is None else seed
    rng = np.random.default_rng(seed) if rng is None else rng
    op_kind = rng.choice(['swap', 'scale', 'replace'])
    def subscript():
        return rng.integers(shape[0]) + 1
    def coeff():
        n = rng.integers(low=low, high=high - 1)
        return n + 1 if n >= 0 else n
    if op_kind == 'swap':
        return (op_kind, subscript(), subscript())
    if op_kind == 'scale':
        return (op_kind, subscript(), coeff())
    if op_kind == 'replace':
        i = subscript()
        j = rng.integers(shape[0] - 1) + 1
        if j >= i:
            j += 1
        return (op_kind, i, coeff(), j)

def row_ops(
        shape,
        num,
        rng=None,
        seed=None,
        low=-5,
        high=5,
):
    assert len(shape) == 2 and shape[0] >= 1 and shape[1] >= 1
    assert low <= high
    seed = secrets.randbits(32) if seed is None else seed
    rng = np.random.default_rng(seed) if rng is None else rng
    return [row_op(shape, rng=rng, seed=seed, low=low, high=high) for _ in range(num)], seed
