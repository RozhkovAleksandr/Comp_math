from math import sqrt
import numpy as np
from PIL import Image
import argparse
import time
from random import normalvariate

# --------------- SIMPLE --------------

EPS = 1e-10

def svd_1d(A):
    n, m = A.shape
    unnormalized = [normalvariate(0, 1) for _ in range(m)]
    theNorm = sqrt(sum(x * x for x in unnormalized))
    x =  [x / theNorm for x in unnormalized]
    lastV = None
    currentV = x
    B = np.dot(A.T, A)

    iterations = 0
    while True:
        iterations += 1
        lastV = currentV
        currentV = np.dot(B, lastV)
        currentV = currentV / np.linalg.norm(currentV)

        if abs(np.dot(currentV, lastV)) > 1 - EPS:
            return currentV


def simple(A):
    n, m = A.shape
    svdSoFar = []

    for i in range(m):
        matrixFor1D = A.copy()

        for singularValue, u, v in svdSoFar[:i]:
            matrixFor1D = matrixFor1D - singularValue * np.outer(u, v)

        v = svd_1d(matrixFor1D)
        u_unnormalized = np.dot(A, v)
        sigma = np.linalg.norm(u_unnormalized)
        u = u_unnormalized / sigma

        svdSoFar.append((sigma, u, v))

    singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]

    return us.T, singularValues, vs

# --------------- ADVANCED ---------------

def testfsplit(U, W, e, k):
    test_f_convergence = False

    for l in np.arange(k + 1)[::-1]:
        if abs(e[l]) <= 1.e-2:
            test_f_convergence = True
            break  # break out of l loop
        if abs(W[l - 1]) <= 1.e-2:
            break

    if test_f_convergence: 
        return l

    c, s, l1 = 0.0, 1.0, l - 1

    for i in range(l, k + 1):
        f, e[i] = s * e[i], c * e[i]

        if abs(f) <= 1.e-2: 
            break

        g = W[i]
        f1, g1 = np.fabs(f), np.fabs(g)
        if f1 > g1:
            x = g1 / f1
            h = f1 * np.sqrt(1.0 + x * x)
        else:
            x = f1 / g1
            h = g1 * np.sqrt(1.0 + x * x)
        W[i], c, s = h, g / h, -f / h

        Y, Z = U[:, l1].copy(), U[:, i].copy()
        U[:, l1] = Y * c + Z * s
        U[:, i] = -Y * s + Z * c
    return l

def advanced_method(U, W, e):
    m, n = U.shape
    g = 0.0
    scale = 0.0

    for i in range(n):
        e[i], l = scale * g, i + 1

        if i < m:
            scale = U[i:, i].dot(U[i:, i])

            if scale <= 1.e-2:
                g = 0.0
            else:
                U[i:, i] = U[i:, i] / scale
                s = U[i:, i].dot(U[i:, i])
                f = U[i, i].copy()
                if f >= 0:
                    g = -np.fabs(np.sqrt(s))
                else:
                    g = np.fabs(np.sqrt(s))
                h = f * g - s
                U[i, i] = f - g

                for j in range(l, n):
                    f = U[i:, i].dot(U[i:, j]) / h
                    U[i:, j] = U[i:, j] + f * U[i:, i]

                U[i:, i] *= scale
        else:
            g = 0.0

        W[i] = scale * g

        if (i < m) and (i != n - 1):
            scale = U[i, l:].dot(U[i, l:])

            if scale <= 1.e-2:
                g = 0.0
            else:
                U[i, l:] = U[i, l:] / scale
                s = U[i, l:].dot(U[i, l:])
                f = U[i, l].copy()
                if f >= 0:
                    g = - np.fabs(np.sqrt(s))
                else:
                    g = np.fabs(np.sqrt(s))
                h = f * g - s
                U[i, l] = f - g
                e[l:] = U[i, l:] / h

                for j in range(l, m):
                    s = U[j, l:].dot(U[i, l:])
                    U[j, l:] = U[j, l:] + s * e[l:]

                U[i, l:] *= scale
        else:
            g = 0.0

def convergence_method(U, W, V, e, k):
    for t in range(1000):
        l = testfsplit(U, W, e, k)

        if l == k:
            if W[k] < 0.0:
                W[k] *= -1
                V[:, k] *= -1
            break

        x, y, z = W[l], W[k - 1], W[k]
        g, h = e[k - 1], e[k]

        f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y)

        f1, g1 = np.fabs(f), np.fabs(1.0)
        if f1 > g1:
            g = f1 * np.sqrt(1.0 + pow(g1 / f1, 2))
        else:
            g = g1 * np.sqrt(1.0 + pow(f1 / g1, 2))

        t = np.fabs(g) if f >= 0 else - np.fabs(g)
        f = ((x - z) * (x + z) + h * ((y / (f + t)) - h)) / x
        c = 1.0
        s = 1.0
        for i in range(l + 1, k + 1):
            g, y = e[i], W[i]
            h, g = s * g, c * g
            f1, g1 = np.fabs(f), np.fabs(h)
            if f1 > g1:
                z = f1 * np.sqrt(1.0 + pow(g1 / f1, 2))
            else:
                z = g1 * np.sqrt(1.0 + pow(f1 / g1, 2))
            e[i - 1] = z

            c, s = f / z, h / z
            f, g = x * c + g * s, g * c - x * s
            h, y = y * s, y * c

            X, Z = V[:, i - 1].copy(), V[:, i].copy()
            V[:, i - 1] = c * X + s * Z
            V[:, i] = c * Z - s * X
            f1, g1 = np.fabs(f), np.fabs(h)
            if f1 > g1:
                z = f1 * np.sqrt(1.0 + pow(g1 / f1, 2))
            else:
                z = g1 * np.sqrt(1.0 + pow(f1 / g1, 2))
            W[i - 1] = z

            if z >= 1.e-2:
                c, s = f / z, h / z

            f, x = c * g + s * y, c * y - s * g

            Y, Z = U[:, i - 1].copy(), U[:, i].copy()
            U[:, i - 1] = c * Y + s * Z
            U[:, i] = c * Z - s * Y

        e[l], e[k], W[k] = 0.0, f, x


def bidiagonalize(A):
    U = np.asarray(A).copy()
    m, n = U.shape

    W = np.zeros(n)
    V = np.zeros((n, n))
    e = np.zeros(n)

    advanced_method(U, W, e)
    m, n = U.shape
    g, l = 0.0, 0

    for i in np.arange(n)[::-1]:
        if i < n - 1:
            if g != 0.0:
                V[l:, i] = (U[i, l:] / U[i, l]) / g
                for j in range(l, n):
                    s = U[i, l:].dot(V[l:, j])
                    V[l:, j] += s * V[l:, i]

            V[i, l:] = V[l:, i] = 0.0

        V[i, i], g, l = 1.0, e[i], i

    for i in np.arange(min([m, n]))[::-1]:
        l = i + 1
        g = W[i]
        U[i, l:] = 0.0

        if g != 0.0:
            g = 1.0 / g
            for j in range(l, n):
                f = (U[l:, i].dot(U[l:, j]) / U[i, i]) * g
                U[i:, j] = U[i:, j] + f * U[i:, i]

            U[i:, i] = U[i:, i] * g
        else:
            U[i:, i] = 0.0

        U[i, i] += 1.0

    return U, W, V, e


def advanced(A):
    U, W, V, e = bidiagonalize(A)

    for k in np.arange(U.shape[1])[::-1]:
        convergence_method(U, W, V, e, k)

    m, n = U.shape
    idsorted = np.argsort(-W)

    U = U[:, idsorted]
    S = W[idsorted]
    V = V[:, idsorted]

    return U, S, np.transpose(V)


# --------------- NUMPY ---------------

def compress(input_file, temp_file, N, svd):
    img = Image.open(f"/task3/images/{input_file}.bmp")

    img_array = np.array(img, dtype='float32' )

    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]

    if (svd == "numpy") :
        U1, Sigma1, V1 = np.linalg.svd(r)
        U2, Sigma2, V2 = np.linalg.svd(g)
        U3, Sigma3, V3 = np.linalg.svd(b)
    elif (svd == "simple"):
        U1, Sigma1, V1 = simple(r)
        U2, Sigma2, V2 = simple(g)
        U3, Sigma3, V3 = simple(b)
    else:
        U1, Sigma1, V1 = advanced(r)
        U2, Sigma2, V2 = advanced(g)
        U3, Sigma3, V3 = advanced(b)

    m, n = img.size
    print(m, n)

    k = int(max(1, ((-(m+n) + sqrt((m+n)**2 + 4 * m * n / 9 / N)) / 2)))
    if k <= 0:
        print("Can't compress image")
        exit(1)

    new_r = np.diag(Sigma1[:k])
    new_g = np.diag(Sigma2[:k])
    new_b = np.diag(Sigma3[:k])

    np.savez(f"/task3/images/{temp_file}.npz", 
             U1=U1[:, :k], Sigma1=new_r, V1=V1[:k, :],
             U2=U2[:, :k], Sigma2=new_g, V2=V2[:k, :],
             U3=U3[:, :k], Sigma3=new_b, V3=V3[:k, :])

def decompress(temp_file, result):
    data = np.load(f"/task3/images/{temp_file}.npz")
    U1 = data["U1"]
    Sigma1 = data["Sigma1"]
    V1 = data["V1"]
    U2 = data["U2"]
    Sigma2 = data["Sigma2"]
    V2 = data["V2"]
    U3 = data["U3"]
    Sigma3 = data["Sigma3"]
    V3 = data["V3"]

    r = np.dot(U1, np.dot(Sigma1, V1))
    g = np.dot(U2, np.dot(Sigma2, V2))
    b = np.dot(U3, np.dot(Sigma3, V3))

    compressed_img_array = np.stack((r, g, b), axis=-1)
    image = Image.fromarray(np.uint8(compressed_img_array))
    image.save(f"/task3/images/{result}.bmp")

# --------------- PARSE ---------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    start_time = time.time()
    parser.add_argument("--mode", choices=["compress", "decompress"], required=True, help="Mode of operation: compress "
                                                                                          "or decompress")
    parser.add_argument("--method", choices=["numpy", "simple", "advanced"], help="Method of singular value "
                                                                                  "decomposition (only required if "
                                                                                  "mode is compress)")
    parser.add_argument("--compression", type=int, help="Compression factor (only required if mode is compress)")
    parser.add_argument("--in_file", required=True, help="Input file path")
    parser.add_argument("--out_file", required=True, help="Output file path")
    args = parser.parse_args()
    method = args.method
    compression = args.compression
    in_file = args.in_file
    out_file = args.out_file

    if args.mode == "compress":
        if method is None or compression is None:
            ValueError("Error: --method and --compression are required for compression mode.")
        if method == "numpy":
            compress(in_file, out_file, compression, method)
        elif method == "simple":
            compress(in_file, out_file, compression, method)
        elif method == "advanced":
            compress(in_file, out_file, compression, method)

    elif args.mode == "decompress":
        decompress(in_file, out_file)
    else:
        ValueError("Error: Invalid mode. Mode must be 'compress' or 'decompress'.")
    print("--- %s seconds ---" % (time.time() - start_time))
