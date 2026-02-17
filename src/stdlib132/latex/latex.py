import sympy
import numpy as np


def lined_env(env, lines):
    out = ""
    for line in lines[:-1]:
        out += line + " \\\\"
    out += lines[-1]
    return f"\\begin{{{env}}}" + out + f"\\end{{{env}}}"


def align_env(lines, star=True) -> str:
    env = "align" + ("*" if star else "")
    return lined_env(env, lines)


def lin_comb(coeffs, elem_strs=None, zero_str=None):
    if elem_strs is None:
        elem_strs = [f'x_{i + 1}' for i in range(len(coeffs))]
    zero_str = '0' if zero_str is None else zero_str
    if not any(coeffs):
        return zero_str
    i = list(coeffs.iter_items())[0][0]
    coeff = coeffs[i]
    out = ""
    if coeff == 1:
        coeff_str = ""
    elif coeff == -1:
        coeff_str = "-"
    else:
        coeff_str = f"{coeff}"
    out += f"{coeff_str}{elem_strs[i]}"
    for i in range(i + 1, len(coeffs)):
        coeff = coeffs[i]
        if coeff != 0:
            op = "+" if coeff > 0 else "-"
            coeff = f"{abs(coeff)}" if abs(coeff) > 1 else ""
            out += f" {op} {coeff}{elem_strs[i]}"
    return out


def lin_eq(
        coeffs,
        elem_strs=None,
        zero_str=None,
        rhs_str=None,
        aligned=None,
):
    lhs = lin_comb(
        coeffs,
        elem_strs=elem_strs,
        zero_str=zero_str,
    )
    zero_str = '0' if zero_str is None else zero_str
    rhs_str = zero_str if rhs is None else rhs_str
    aligned = False if aligned is None else aligned
    eq = "&=" if aligned else "="
    return f"{lhs} {eq} {rhs}"


def lin_sys(aug):
    lines = []
    for i in range(aug.rows):
        if not all(entry.is_zero for entry in aug.row(i)):
            row = list(aug.row(i))
            lines.append(lin_eq(row[:-1], row[-1], aligned=True))
    return align_env(lines)


def matrix(aug):
    return sympy.latex(aug)


def lin_transform(a, var_strs=None):
    if var_strs is None:
        var_strs = [f"x_{i + 1}" for i in range(a.cols)]
    lin_combs = []
    for i in range(a.rows):
        coeffs = a.row(i)
        lin_combs.append(lin_comb(coeffs, elem_strs=var_strs)) # ? TODO
    return f"{matrix(sympy.Matrix(var_strs))} \\mapsto {bmatrix(lin_combs)}"


def swap_row_op(i, j, align=None):
    align = False if align is None else align
    arrow = (if align then "&" else "") + "\leftrightarrow"
    return f"R_{{{i}}} {arrow} R_{{{j}}}"


def scale_row_op(i, c, align=None):
    scalar_str = f"{c}"
    if c == 1:
        scalar_str = ""
    if c == -1:
        scalar_str = "-"
    align = False if align is None else align
    arrow = (if align then "&" else "") + "\gets"
    return f"R_{{{i}}} {arrow} {scalar_str}R_{{{i}}}"


def replace_row_op(i, c, j, align=None):
    scalar_str = f"{abs(c)}" if abs(c) > 1 else ""
    op_str = "+" if c >= 1 else "-"
    align = False if align is None else align
    arrow = (if align then "&" else "") + "\gets"
    return f"R_{{{i}}} {arrow} R_{{{i}}} {op_str} {scalar_str}R_{{{j}}}"


def row_op(op, align=None):
    if op[0] == "swap":
        return swap_row_op(op[1], op[2], align=align)
    if op[0] == "scale":
        return scale_row_op(op[1], op[2], align=align)
    if op[0] == "replace":
        return replace_row_op(op[1], op[2], op[3], align=align)


def row_ops(ops):
    out = "\\begin{align*}\n"
    for op in ops[:-1]:
        out += row_op(op, align=True) + " \\\\\n"
    return out + row_op(ops[-1]) + "\n\\end{align*}"


def gen_form_sol(rref):
    def row(r):
        nonzeros = np.nonzero(r)[0]
        if nonzeros.size:
            leading_index = nonzeros[0]
        else:
            return None
        rhs = f"{r[-1]}"
        for i in range(leading_index + 1, len(r) - 1):
            scalar = r[i]
            if scalar > 0:
                rhs += f" - {str(scalar) if scalar > 1 else ''}x_{{{i + 1}}}"
            elif scalar < 0:
                rhs += f" + {str(abs(scalar)) if scalar < 1 else ''}x_{{{i + 1}}}"
        if rhs[:4] == "0 + ":
            rhs = rhs[4:]
        elif rhs[:4] == "0 - ":
            rhs = "-" + rhs[4:]
        return f"x_{{{leading_index + 1}}} &= {rhs} \\\\\n"

    out = "\\begin{align*}\n"
    count = 0
    for i in range(rref.shape[0]):
        r = rref[i]
        nonzeros = np.nonzero(r)[0]
        if nonzeros.size:
            leading_index = nonzeros[0]
            while count < leading_index:
                out += f"x_{{{count + 1}}} &\\text{{ is free}} \\\\\n"
                count += 1
            out += row(r)
            count += 1
        else:
            while count < len(r) - 1:
                out += f"x_{{{count + 1}}} &\\text{{ is free}} \\\\\n"
                count += 1
            return out[:-4] + "\n" + "\\end{align*}"
    return out[:-4] + "\n" + "\\end{align*}"


def lin_comb_vec(coeffs, vecs):
    vec_strs = []
    for vec in vecs:
        vec_strs.append(matrix(vec))
    return lin_comb(coeffs, vec_strs, "\\mathbf{0}")


def mat_set(mats):
    out = "\\begin{align*}\n"
    out += f"{mats[0][0]} = {matrix(mats[0][1])}"
    for name, mat in mats[1:]:
        out += f" \\quad {name} = {matrix(mat)}"
    out += "\n\\end{align*}"
    return out


def matrix_collection(mats, names=None):
    assert len(mats) >= 1
    if names is None:
        names = [f"\\mathbf{{v}}_{{{i + 1}}}" for i in range(len(mats))]
    out = "\\begin{align*}"
    out += f"{names[0]} = {matrix(mats[0])}"
    for i in range(1, len(mats)):
        out += f" \\quad {names[i]} = {matrix(mats[i])}"
    out += "\\end{align*}"
    return out


def vec_set(vecs, names=None):
    if names is None:
        names = []
        for i in range(vecs.shape[1]):
            names.append(f"\\mathbf{{v}}_{{{i + 1}}}")
    out = "\\begin{align*}"
    out += f"{names[0]} = {bmatrix(vecs[:, 0])}"
    for i in range(1, vecs.shape[1]):
        out += f" \\quad {names[i]} = {bmatrix(vecs[:, i])}"
    out += "\\end{align*}"
    return out


def set(elems: list[str]) -> str:
    out = "\\left\\{"
    out += f"{elems[0]}"
    for i in range(1, len(elems)):
        out += f", {elems[i]}"
    out += "\\right\\}"
    return out


def vector_set(vecs: list[sympy.MatrixBase]) -> str:
    return set([matrix(vec) for vec in vecs])


def span(vecs):
    return "\\mathrm{span}\\!" + vector_set(vecs)


def vec_eq(aug):
    out = "\\begin{align*}\n"
    out += f"x_1{matrix(aug[:, 0])}"
    for i in range(1, aug.cols - 1):
        out += f" + x_{{{i + 1}}}{matrix(aug[:, i])}"
    out += f"= {matrix(aug[:, -1])}\n"
    out += "\\end{align*}"
    return out
