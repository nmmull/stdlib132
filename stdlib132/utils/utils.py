
def apply_row_op(op, a):
    if op[0] == 'swap':
        i = op[1] - 1
        j = op[2] - 1
        a[(i, j),:] = a[(j, i),:]
    if op[0] == 'scale':
        i = op[1] - 1
        n = op[2]
        a[i] *= n
    if op[0] == 'replace':
        i = op[1] - 1
        n = op[2]
        j = op[3] - 1
        a[i] += n * a[j]

def apply_row_ops(ops, a):
    for op in ops:
        apply_row_op(op, a)
