import numpy as np


def lined_env(env: str, lines: list[str]) -> str:
    """Latex environment with lines.

    Parameters
    ----------
    env : str
        Name of the environment.
    lines : list[str]
        Lines to be put in the environment.

    Returns
    -------
    str
        Environment with `lines` (not indented).

    """
    assert len(lines) > 0
    out = ""
    for line in lines[:-1]:
        out += line + " \\\\\n"
    out += lines[-1]
    return f"\\begin{{{env}}}\n" + out + f"\n\\end{{{env}}}"


def align_env(lines: list[str], star: bool = True) -> str:
    """`align*` environment.

    Parameters
    ----------
    lines : list[str]
        Lines to be put in an `align` environment.
    star : bool, default=True
        Determines if number labels should be removed from the lines
        in the environment.

    Returns
    -------
    str
        `align` environment with `lines` (not indented)

    """
    env = "align" + ("*" if star else "")
    return lined_env(env, lines)


def lin_comb(coeffs: np.ndarray, elem_strs: list[str], zero_str: str) -> str:
    """Latex for a linear combination.

    Parameters
    ----------
    coeff : numpy.ndarray
        Coefficients used in the linear combination.  This is required
        to be a 1D array.  We only support integer coefficents as of
        now.
    elem_strs : list[str]
        The strings used for the elements of the linear combination.
    zero_str : str
        The string used in the case that all coefficients are `0`.

    Returns
    -------
    str
        Latex for a linear combination.

    """
    assert len(coeffs.shape) == 1
    if not np.any(coeffs):
        return zero_str
    i = np.nonzero(coeffs)[0][0]
    coeff = coeffs[i]
    out = ""
    if coeff == 1:
        coeff_str = ""
    elif coeff == -1:
        coeff_str = "-"
    else:
        coeff_str = f"{coeff}"
    out += f"{coeff_str}{elem_strs[i]}"
    for i in range(i + 1, coeffs.shape[0]):
        coeff = coeffs[i]
        if coeff != 0:
            op = "+" if coeff > 0 else "-"
            coeff = f"{abs(coeff)}" if abs(coeff) > 1 else ""
            out += f" {op} {coeff}{elem_strs[i]}"
    return out


def lin_eq(coeffs: np.ndarray, rhs: int, aligned: bool = False) -> str:
    """Latex for a linear equation.

    Parameters
    ----------
    coeffs : np.ndarray
        The coefficents used for the left side of the equation.  We
        only support integer coefficents as of now.
    rhs: int
        The value used for the right side of the equation
    aligned : bool, default=False
        Determined whether or not to include `&` for the `align*`
        environment

    Return
    ------
    str
        Latex for a linear equation.

    """
    lhs = lin_comb(coeffs, [f"x_{{{i + 1}}}" for i in range(len(coeffs))], "0")
    eq = "&=" if aligned else "="
    return f"{lhs} {eq} {rhs}"


def lin_sys(aug: np.ndarray) -> str:
    """Latex for a linear system.

    Parameters
    ----------
    aug : numpy.ndarray
        The augmented matrix of a linear system.

    Returns
    -------
    str
        Latex for the linear system with augmented matrix `aug`

    """
    assert len(aug.shape) == 2
    assert aug.shape[0] >= 1 and aug.shape[1] >= 2
    lines = []
    for i in range(aug.shape[0]):
        if not np.all(aug[i] == 0):
            lines.append(lin_eq(aug[i, :-1], aug[i, -1], aligned=True))
    return align_env(lines)


def bmatrix_env(lines: list[str]):
    """`bmatrix` environment.

    Parameters
    ----------
    lines: list[str]

    Returns
    -------
    str
        Lines put in a `bmatrix` environment.

    """
    return lined_env("bmatrix", lines)


def bmatrix(a) -> str:
    """Latex for matrices and vectors using the `bmatrix` environment.

    Parameters
    ----------
    a
        Array used to construct a vector or a matrix

    Returns
    str
        Latex for `a` in a `bmatrix` environment

    """

    def row_latex(row):
        if isinstance(row, str):
            return row
        try:
            out = f"{row[0]}"
            for elem in row[1:]:
                out += f" & {elem}"
            return out
        except (TypeError, IndexError):
            return f"{row}"

    lines = []
    for row in a:
        lines.append(row_latex(row))
    return bmatrix_env(lines)


def point(v) -> str:
    """Latex for a point.

    Parameters
    ----------
    v
        A collection of values for the entries of the point.

    Returns
    -------
    str
        Latex for a point.

    """
    return f"{tuple(v)}"


def lin_transform(a: np.ndarray) -> str:
    """Latex for a linear transformation

    Parameters
    ----------
    a : numpy.ndarray
        The coefficients used in the output entires of the linear
        transformation.

    Returns
    -------
    str
        Latex for the linear transformation defined by `a`

    """
    assert len(a.shape) == 2
    assert a.shape[0] >= 1 and a.shape[1] >= 1
    var_names = [f"x_{{{i + 1}}}" for i in range(a.shape[1])]
    lin_combs = []
    for coeffs in a:
        lin_combs.append(lin_comb(coeffs, var_names, "0"))
    return f"{bmatrix(var_names)} \\mapsto {bmatrix(lin_combs)}"

def row_op(op):
    def scalar(x):
        if x == 1:
            return ""
        if x == -1:
            return "-"
        return f"{x}"

    if op[0] == "swap":
        return f"R_{{{op[1]}}} &\\leftrightarrow R_{{{op[2]}}}"
    if op[0] == "scale":
        return f"R_{{{op[1]}}} &\\gets {scalar(op[2])}R_{{{op[1]}}}"
    if op[0] == "replace":
        scalar = op[2]
        scalar_str = f"{abs(scalar)}" if abs(scalar) > 1 else ""
        op_str = "+" if scalar >= 1 else "-"
        return f"R_{{{op[1]}}} &\\gets R_{{{op[1]}}} {op_str} {scalar_str}R_{{{op[3]}}}"


def row_ops(ops):
    out = "\\begin{align*}\n"
    for op in ops[:-1]:
        out += row_op(op) + " \\\\\n"
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
    for i in range(vecs.shape[1]):
        vec_strs.append(bmatrix(vecs[i, :]))
    return lin_comb(coeffs, vec_strs, "\\mathbf{0}")


def mat_set(mats):
    out = "\\begin{align*}\n"
    out += f"{mats[0][0]} = {bmatrix(mats[0][1])}"
    for name, mat in mats[1:]:
        out += f" \\quad {name} = {bmatrix(mat)}"
    out += "\n\\end{align*}"
    return out


def vec_set(vecs, names=None):
    if names is None:
        names = []
        for i in range(vecs.shape[1]):
            names.append(f"\\mathbf{{v}}_{{{i + 1}}}")
    out = "\\begin{align*}\n"
    out += f"{names[0]} = {bmatrix(vecs[:, 0])}"
    for i in range(1, vecs.shape[1]):
        out += f" \\quad {names[i]} = {bmatrix(vecs[:, i])}"
    out += "\n\\end{align*}"
    return out


def vec_eq(aug):
    out = "\\begin{align*}\n"
    out += f"x_1{bmatrix(aug[:, 0])}"
    for i in range(1, aug.shape[1] - 1):
        out += f" + x_{{{i + 1}}}{bmatrix(aug[:, i])}"
    out += f"= {bmatrix(aug[:, -1])}\n"
    out += "\\end{align*}"
    return out
