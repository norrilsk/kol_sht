def fast_power(x, y, mod = None):
    stry = bin(y)

    k = x
    mul = 1
    i = 0
    for c in stry[::-1]:
        if c == 'b':
            break
        elif c == '1':
            mul *= k
        k *= k
        i += 1
        if mod:
            k %= mod
            mul %= mod
    return mul