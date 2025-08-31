import numpy as np

def lin_eq(coeffs, rhs):
    if not np.any(coeffs):
        return f'0 = {rhs}'
    out = ''
    i = np.nonzero(coeffs)[0][0]
    coeff = coeffs[i]
    if coeff == 1:
        coeff_str = ''
    elif coeff == -1:
        coeff_str = '-'
    else:
        coeff_str = f'{coeff}'
    out += f'{coeff_str}x_{{{i + 1}}}'
    for i in range(i + 1, coeffs.shape[0]):
        coeff = coeffs[i]
        if coeff != 0:
            if coeff < 0:
                op = '-'
            else:
                op = '+'
            out += f' {op} {str(abs(coeff)) if abs(coeff) > 1 else ""}x_{{{i + 1}}}'
    return f'{out} &= {rhs}'

def lin_sys(aug):
    num_rows = aug.shape[0]
    num_cols = aug.shape[1]
    assert(num_rows >= 1 and num_cols >= 2)
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
    num_rows = a.shape[0]
    num_cols = a.shape[1]
    assert num_rows > 0 and num_cols > 0
    def row_latex(row):
        out = f'{row[0]}'
        for elem in row[1:]:
            out += f' & {elem}'
        return out
    out = '\\begin{bmatrix}\n'
    for row in a[:-1]:
        out += row_latex(row) + ' \\\\\n'
    return out + row_latex(a[-1]) + '\n\\end{bmatrix}'

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
        return f'R_{{{op[1]}}} \\leftrightarrow R_{{{op[2]}}}'
    if op[0] == 'scale':
        return f'R_{{{op[1]}}} \\gets {scalar(op[2])}R_{{{op[1]}}}'
    if op[0] == 'replace':
        return f'R_{{{op[1]}}} \\gets R_{{{op[1]}}} + {scalar(op[2])}R_{{{op[3]}}}'

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
