import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

REPS = 1
THRESHOLD = 32


# ============================================================
# Standard (NumPy baseline)
# ============================================================

def standard_np(A, B):
    return A @ B


# ============================================================
# Helpers
# ============================================================

def add(A, B):
    return A + B


def sub(A, B):
    return A - B


def split(A):
    n = A.shape[0]
    mid = n // 2

    return (
        A[:mid, :mid],
        A[:mid, mid:],
        A[mid:, :mid],
        A[mid:, mid:]
    )


def combine(C11, C12, C21, C22):
    top = np.hstack((C11, C12))
    bottom = np.hstack((C21, C22))
    return np.vstack((top, bottom))


# ============================================================
# Sequential Strassen (correct)
# ============================================================

def strassen(A, B):
    n = A.shape[0]

    if n <= THRESHOLD:
        return A @ B

    A11, A12, A21, A22 = split(A)
    B11, B12, B21, B22 = split(B)

    M1 = strassen(add(A11, A22), add(B11, B22))
    M2 = strassen(add(A21, A22), B11)
    M3 = strassen(A11, sub(B12, B22))
    M4 = strassen(A22, sub(B21, B11))
    M5 = strassen(add(A11, A12), B22)
    M6 = strassen(sub(A21, A11), add(B11, B12))
    M7 = strassen(sub(A12, A22), add(B21, B22))

    C11 = add(sub(add(M1, M4), M5), M7)
    C12 = add(M3, M5)
    C21 = add(M2, M4)
    C22 = add(sub(add(M1, M3), M2), M6)

    return combine(C11, C12, C21, C22)


# ============================================================
# Parallel part (ONLY top-level M1..M7)
# ============================================================

def compute_M(task):
    func, args = task
    return func(*args)


def strassen_parallel(A, B):
    n = A.shape[0]

    if n <= THRESHOLD:
        return A @ B

    A11, A12, A21, A22 = split(A)
    B11, B12, B21, B22 = split(B)

    tasks = [
        (strassen, (add(A11, A22), add(B11, B22))),  # M1
        (strassen, (add(A21, A22), B11)),            # M2
        (strassen, (A11, sub(B12, B22))),            # M3
        (strassen, (A22, sub(B21, B11))),            # M4
        (strassen, (add(A11, A12), B22)),            # M5
        (strassen, (sub(A21, A11), add(B11, B12))),  # M6
        (strassen, (sub(A12, A22), add(B21, B22)))   # M7
    ]

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as pool:
        M1, M2, M3, M4, M5, M6, M7 = pool.map(compute_M, tasks)

    C11 = add(sub(add(M1, M4), M5), M7)
    C12 = add(M3, M5)
    C21 = add(M2, M4)
    C22 = add(sub(add(M1, M3), M2), M6)

    return combine(C11, C12, C21, C22)


# ============================================================
# Padding to power of 2
# ============================================================

def next_pow2(n):
    return 1 << (n - 1).bit_length()


def pad(A, size):
    B = np.zeros((size, size), dtype=A.dtype)
    B[:A.shape[0], :A.shape[1]] = A
    return B


def unpad(A, n):
    return A[:n, :n]


# ============================================================
# Wrappers
# ============================================================

def strassen_seq(A, B):
    n = A.shape[0]
    size = next_pow2(n)

    A_pad = pad(A, size)
    B_pad = pad(B, size)

    C = strassen(A_pad, B_pad)
    return unpad(C, n)


def strassen_par(A, B):
    n = A.shape[0]
    size = next_pow2(n)

    A_pad = pad(A, size)
    B_pad = pad(B, size)

    C = strassen_parallel(A_pad, B_pad)
    return unpad(C, n)


# ============================================================
# Benchmark
# ============================================================

def benchmark(N):
    A = np.random.randint(0, 10, (N, N))
    B = np.random.randint(0, 10, (N, N))

    # standard
    t0 = time.perf_counter()
    for _ in range(REPS):
        standard_np(A, B)
    t_std = (time.perf_counter() - t0) / REPS

    # strassen seq
    t0 = time.perf_counter()
    for _ in range(REPS):
        strassen_seq(A, B)
    t_str = (time.perf_counter() - t0) / REPS

    # strassen parallel
    t0 = time.perf_counter()
    for _ in range(REPS):
        strassen_par(A, B)
    t_par = (time.perf_counter() - t0) / REPS

    return t_std * 1000, t_str * 1000, t_par * 1000


# ============================================================
# Main
# ============================================================

def main():
    sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

    print(f"CPU cores: {multiprocessing.cpu_count()}")
    print(f"THRESHOLD: {THRESHOLD}\n")

    print(f"{'N':>6} | {'Std':>10} | {'Strassen':>10} | {'Par Str':>10}")
    print("-" * 45)

    for N in sizes:
        t_std, t_str, t_par = benchmark(N)

        print(
            f"{N:6d} | "
            f"{t_std:8.2f} ms | "
            f"{t_str:8.2f} ms | "
            f"{t_par:8.2f} ms"
        )


if __name__ == "__main__":
    main()