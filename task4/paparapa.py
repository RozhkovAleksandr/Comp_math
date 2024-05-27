import random
import numpy as np
from math import cos, sqrt, sin
from scipy.integrate import quad

# Алгоритм взят с параграфа 8. Стр. 92 http://www.ict.nsc.ru/matmod/files/textbooks/KhakimzyanovCherny-2.pdf

def phi_j(j, xj, x, N, h):
    cases = {
        0: lambda: (x[1] - xj) / h if x[0] <= xj <= x[1] else 0,
        N: lambda: (xj - x[N - 1]) / h if x[N - 1] <= xj <= x[N] else 0
    }
    
    def default_case():
        if x[j - 1] <= xj <= x[j]:
            return (xj - x[j - 1]) / h
        elif x[j] <= xj <= x[j + 1]:
            return (x[j + 1] - xj) / h
        else:
            return 0

    return cases.get(j, default_case)()

def border(lam, j, k, x, h):
    if j > k:
        j, k = k, j
    
    if j == k:
        return ((x[j+1] + lam * x[j]**2 * x[j+1] - lam * x[j] * x[j+1]**2 + (lam * x[j+1]**3) / 3 - 
                x[j-1] - lam * x[j]**2 * x[j-1] + lam * x[j] * x[j-1]**2 - (lam * x[j-1]**3) / 3) / (h**2))
    elif j + 1 == k:
        return (-1 / 6.0) * (-6 + lam * (x[j] - x[j-1])**2) * (x[j] - x[j+1]) / (h**2)
    else:
        return 0

def tomas(a, b, c, d):
    n = d.size
    for i in range(1, n):
        tmp = a[i] / b[i-1]
        b[i] = b[i] - tmp * c[i-1]
        d[i] = d[i] - tmp * d[i-1]
    
    arr = np.zeros(n+1)
    arr[n-1] = d[n-1] / b[n-1]
    for i in range(n-2, -1, -1):
        arr[i] = (d[i] - c[i] * arr[i+1]) / b[i]
    return arr

def solve(lam, x, N):
    # matrix
    
    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    d = np.zeros(N)
    l_sqrt = sqrt(lam)
    h = x[1] - x[0]

    for i in range(1, N + 1):
        j = i - 1
        h = x[1] - x[0]
        cos_l = cos(l_sqrt * x[j])
        sin_l = sin(l_sqrt * x[j])
        
        if i >= 1:
            a[j] = border(lam, i - 1, i, x, h)
        if i < N:
            c[j] = border(lam, i + 1, i, x, h)
        b[j] = border(lam, j, j, x, h)

        if j == 0:
            a_term = 2 * (-l_sqrt * (x[j] - x[j + 1]) * cos_l + sin_l - sin(l_sqrt * x[j + 1]))
            b_term = 0
        elif j == N - 1:
            a_term = 0
            b_term = 2 * (-l_sqrt * (x[j] - x[j - 1]) * cos_l + sin_l - sin(l_sqrt * x[j - 1]))
        else:
            a_term = 2 * (-l_sqrt * (x[j] - x[j + 1]) * cos_l + sin_l - sin(l_sqrt * x[j + 1]))
            b_term = 2 * (-l_sqrt * (x[j] - x[j - 1]) * cos_l + sin_l - sin(l_sqrt * x[j - 1]))

        d[j] = (a_term + b_term) / h

    arr = tomas(a, b, c, d)
    arr = np.concatenate(([0], arr, [0]))

    return arr

def f(lam, L):
    def integrand(x):
        return (-2 * lam * np.sin(np.sqrt(lam) * x))**2

    norm, _ = quad(integrand, 0, L)

    return np.sqrt(norm)

def calc(y, xj, x, N):
    l = 0
    r = N - 1
    h = x[1] - x[0]
    while r - l > 1:
        m = (l + r) // 2
        if xj > x[m]:
            l = m
        else:
            r = m
    return y[l] * phi_j(l, xj, x, N, h) + y[r] * phi_j(r, xj, x, N, h)


def check_error(lam, N, h, y, x):
    arr = np.array([sin(sqrt(lam) * (i * h)) for i in range(N)])
    app = np.array([calc(y, i * h, x, N) for i in range(N)])

    border = np.sqrt(np.sum((arr - app) ** 2) * h)

    return border

for N in range(100, 1001, 100):
    A = 0.0
    B = np.pi
    n = random.randint(1, 4)
    lam = (np.pi * n / B) ** 2
    x = np.linspace(A, B, N+1)
    h = x[1] - x[0]
    y = solve(lam, x, N)
    err = check_error(lam, N, h, y, x)
    print(f"Error: \t \t {err}  \t \t h: {h} \t \t N: {N}")
