import numpy as np


# Global variables
# -----------------------------------------------------------------------
def get_local_fft_shape(nr, direct=True, full=False, **kwargs):
    s = []
    for item in nr:
        s.append(slice(None))
    s = tuple(s)
    shape = np.array(nr)
    if not full and not direct:
        shape[-1] = shape[-1] // 2 + 1
    offsets = np.zeros_like(nr, dtype=np.int32)
    return (s, shape, offsets)


def best_fft_size(N, max_prime=13, scale=0.99, even=True, prime_factors=None, **kwargs):
    """
    http ://www.fftw.org/fftw3_doc/Complex-DFTs.html#Complex-DFTs
    "FFTW is best at handling sizes of the form 2^a 3^b 5^c 7^d 11^e 13^f,  where e+f is either 0 or 1,  and the other exponents are arbitrary."
    """
    prime_factors_multi = [2, 3, 5, 7]
    prime_factors_one = [
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
    ]

    if prime_factors is None:
        prime_factors = prime_factors_multi
    else:
        for item in prime_factors:
            if item not in prime_factors_multi:
                prime_factors_one.insert(0, item)

    a = int(np.ceil(np.log2(N))) + 1
    b = int(np.ceil(np.log(N) / np.log(3))) + 1
    c = int(np.ceil(np.log(N) / np.log(5))) + 1
    d = int(np.ceil(np.log(N) / np.log(7))) + 1

    if 3 not in prime_factors:
        b = 1
    if 5 not in prime_factors:
        c = 1
    if 7 not in prime_factors:
        d = 1

    if even:
        istart = 1
    else:
        istart = 0

    if max_prime == 2:
        mgrid = np.arange(istart, a).reshape(1, -1)
        arr0 = 2 ** mgrid[0]
    elif max_prime == 3:
        mgrid = np.mgrid[istart:a, :b].reshape(2, -1)
        arr0 = 2 ** mgrid[0] * 3 ** mgrid[1]
    elif max_prime == 5:
        mgrid = np.mgrid[istart:a, :b, :c].reshape(3, -1)
        arr0 = 2 ** mgrid[0] * 3 ** mgrid[1] * 5 ** mgrid[2]
    elif max_prime > 5:
        mgrid = np.mgrid[istart:a, :b, :c, :d].reshape(4, -1)
        arr0 = 2 ** mgrid[0] * 3 ** mgrid[1] * 5 ** mgrid[2] * 7 ** mgrid[3]

    if N < 100:
        arr1 = arr0[np.logical_and(arr0 > N / (max_prime + 1), arr0 < 2 * N)]
    else:
        arr1 = arr0[np.logical_and(arr0 > N / (max_prime + 1), arr0 < 1.2 * N)]

    arrAll = []
    arrAll.extend(arr1)
    for item in prime_factors_one:
        if max_prime < item:
            continue
        else:
            arrAll.extend(arr1 * item)

    arrAll = np.asarray(arrAll)
    if scale is None:
        bestN = np.min(arrAll[arrAll > N - 1])
    else:
        bestN = np.min(arrAll[arrAll > np.ceil(scale * N) - 1])
    return bestN
