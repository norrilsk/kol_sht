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


def bytes_to_int (bytes):
    a = 0
    for s in bytes:
        a = a*256 + int(s)
    return a


def int_to_bytes(c:int):
    b = [0]*64
    for i in range(64):
        b[63-i] = c%256
        c=c//256
    return bytearray(b)