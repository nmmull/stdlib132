import numpy as np

def lin_comb(coeffs, elem_strs, zero_str):
    if not np.any(coeffs):
        return zero_str
    i = np.nonzero(coeffs)[0][0]
    coeff = coeffs[i]
    out = ''
    if coeff == 1:
        coeff_str = ''
    elif coeff == -1:
        coeff_str = '-'
    else:
        coeff_str = f'{coeff}'
    out += f'{coeff_str}{elem_strs[i]}'
    for i in range(i + 1, coeffs.shape[0]):
        coeff=coeffs[i]
        if coeff != 0:
            op = '+' if coeff > 0 else '-'
            coeff = f'{abs(coeff)}' if abs(coeff) > 1 else ''
            out += f' {op} {coeff}{elem_strs[i]}'
    return out

def lin_eq(coeffs, rhs):
    lhs = lin_comb(
        coeffs,
        [f'x_{{{i + 1}}}' for i in range(len(coeffs))],
        '0'
    )
    return f'{lhs} &= {rhs}'

def lin_sys(aug):
    assert len(aug.shape) == 2 and aug.shape[0] >= 1 and aug.shape[1] >= 2
    num_rows = aug.shape[0]
    num_cols = aug.shape[1]
    out = '\\begin{align*}\n'
    for i in range(num_rows - 1):
        next_line = lin_eq(aug[i,:-1], aug[i,-1])
        if next_line != '0 = 0':
            out += next_line + ' \\\\\n'
    last_line = lin_eq(aug[-1,:-1], aug[-1, -1])
    if last_line != '0 = 0':
        out += last_line + '\n'
    out += '\\end{align*}'
    return out

def bmatrix(a):
    assert len(a.shape) == 2 and a.shape[0] > 0 and a.shape[1] > 0
    num_rows = a.shape[0]
    num_cols = a.shape[1]
    def row_latex(row):
        out = f'{row[0]}'
        for elem in row[1:]:
            out += f' & {elem}'
        return out
    out = '\\begin{bmatrix}\n'
    for row in a[:-1]:
        out += row_latex(row) + ' \\\\\n'
    return out + row_latex(a[-1]) + '\n\\end{bmatrix}'

def bvector(v):
    num_entries = v.shape[0]
    out = '\\begin{bmatrix}\n'
    for entry in v[:-1]:
        out += f'{entry} \\\\\n'
    return f'{out}{v[-1]}\n\\end{{bmatrix}}'

def solution(v):
    return f'{tuple(v)}'

def row_op(op):
    def scalar(x):
        if x == 1:
            return ''
        if x == -1:
            return '-'
        return f'{x}'
    if op[0] == 'swap':
        return f'R_{{{op[1]}}} &\\leftrightarrow R_{{{op[2]}}}'
    if op[0] == 'scale':
        return f'R_{{{op[1]}}} &\\gets {scalar(op[2])}R_{{{op[1]}}}'
    if op[0] == 'replace':
        scalar = op[2]
        scalar_str = f'{abs(scalar)}' if abs(scalar) > 1 else ''
        op_str = '+' if scalar >= 1 else '-'
        return f'R_{{{op[1]}}} &\\gets R_{{{op[1]}}} {op_str} {scalar_str}R_{{{op[3]}}}'

def row_ops(ops):
    out = '\\begin{align*}\n'
    for op in ops[:-1]:
        out += row_op(op) + ' \\\\\n'
    return out + row_op(ops[-1]) + '\n\\end{align*}'

def gen_form_sol(rref):
    def row(r):
        nonzeros = np.nonzero(r)[0]
        if nonzeros.size:
            leading_index = nonzeros[0]
        else:
            return None
        rhs = f'{r[-1]}'
        for i in range(leading_index + 1, len(r) - 1):
            scalar = r[i]
            if scalar > 0:
                rhs += f' - {str(scalar) if scalar > 1 else ""}x_{{{i + 1}}}'
            elif scalar < 0:
                rhs += f' + {str(abs(scalar)) if scalar < 1 else ""}x_{{{i + 1}}}'
        if rhs[:4] == '0 + ':
            rhs = rhs[4:]
        elif rhs[:4] == '0 - ':
            rhs = '-' + rhs[4:]
        return f'x_{{{leading_index + 1}}} &= {rhs} \\\\\n'
    out = '\\begin{align*}\n'
    count = 0
    for i in range(rref.shape[0]):
        r = rref[i]
        nonzeros = np.nonzero(r)[0]
        if nonzeros.size:
            leading_index = nonzeros[0]
            while count < leading_index:
                out += f'x_{{{count + 1}}} &\\text{{ is free}} \\\\\n'
                count += 1
            out += row(r)
            count += 1
        else:
            while count < len(r) - 1:
                out += f'x_{{{count + 1}}} &\\text{{ is free}} \\\\\n'
                count += 1
            return out[:-4] + '\n' + '\\end{align*}'
    return out[:-4] + '\n' + '\\end{align*}'

def lin_comb_vec(coeffs, vecs):
    vec_strs = []
    for i in range(vecs.shape[1]):
        vec_strs.append(bvector(vecs[i,:]))
    return lin_comb(coeffs, vec_strs, '\\mathbf{0}')

def vec_set(vecs, names=None):
    if names is None:
        names = []
        for i in range(vecs.shape[1]):
            names.append(f'\\mathbf{{v}}_{{{i + 1}}}')
    out = '\\begin{align*}\n'
    out += f'{names[0]} = {bvector(vecs[:,0])}'
    for i in range(1, vecs.shape[1]):
        out += f' \\quad {names[i]} = {bvector(vecs[:,i])}'
    out += '\n\\end{align*}'
    return out

def vec_eq(aug):
    out = '\\begin{align*}\n'
    out += f'x_1{bvector(aug[:,0])}'
    for i in range(1, aug.shape[1] - 1):
        out += f' + x_{{{i + 1}}}{bvector(aug[:,i])}'
    out += f'= {bvector(aug[:,-1])}\n'
    out += '\\end{align*}'
    return out
