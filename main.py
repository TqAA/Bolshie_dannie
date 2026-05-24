"""Алгоритм Штрассена умножения матриц имеет асимптотическую сложность Theta(Nlog 7),
где N – размер перемножаемых квадратных матриц. При каких N время выполнения
реализации алгоритма Штрассена на языках C и Python меньше, чем для алгоритма
стандартного перемножения квадратных матриц?
"""

import numpy as np
import time
from mpi4py import MPI
import psutil
import math

# mpiexec -n 6 python main.py

THRESHOLD = 32

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# =========================
# MATRIX OPS
# =========================

def standard_np(A, B):
    return A @ B

def add(A, B):
    return A + B

def sub(A, B):
    return A - B

def split(A):
    n = A.shape[0]
    m = n // 2
    return A[:m,:m], A[:m,m:], A[m:,:m], A[m:,m:]

def combine(C11, C12, C21, C22):
    return np.vstack((np.hstack((C11, C12)),
                      np.hstack((C21, C22))))

# =========================
# STRASSEN (SEQ)
# =========================

def strassen(A, B):
    n = A.shape[0]

    if n <= THRESHOLD:
        return A @ B

    A11, A12, A21, A22 = split(A)
    B11, B12, B21, B22 = split(B)

    M1 = strassen(add(A11,A22), add(B11,B22))
    M2 = strassen(add(A21,A22), B11)
    M3 = strassen(A11, sub(B12,B22))
    M4 = strassen(A22, sub(B21,B11))
    M5 = strassen(add(A11,A12), B22)
    M6 = strassen(sub(A21,A11), add(B11,B12))
    M7 = strassen(sub(A12,A22), add(B21,B22))

    C11 = add(sub(add(M1,M4),M5),M7)
    C12 = add(M3,M5)
    C21 = add(M2,M4)
    C22 = add(sub(add(M1,M3),M2),M6)

    return combine(C11,C12,C21,C22)

# =========================
# MPI STRASSEN
# =========================

def strassen_mpi(A, B):

    n = A.shape[0]

    if n <= THRESHOLD:
        return A @ B

    A11,A12,A21,A22 = split(A)
    B11,B12,B21,B22 = split(B)

    tasks = [
        (add(A11,A22), add(B11,B22)),
        (add(A21,A22), B11),
        (A11, sub(B12,B22)),
        (A22, sub(B21,B11)),
        (add(A11,A12), B22),
        (sub(A21,A11), add(B11,B12)),
        (sub(A12,A22), add(B21,B22))
    ]

    if rank < 7:
        X,Y = tasks[rank]
        local = strassen(X,Y)
    else:
        local = None

    results = comm.gather(local, root=0)

    if rank == 0:
        M1,M2,M3,M4,M5,M6,M7 = results

        C11 = add(sub(add(M1,M4),M5),M7)
        C12 = add(M3,M5)
        C21 = add(M2,M4)
        C22 = add(sub(add(M1,M3),M2),M6)

        return combine(C11,C12,C21,C22)

    return None

# =========================
# HELPERS (COMPRESSION MODEL)
# =========================

def next_pow2(n):
    return 1 << (n - 1).bit_length()

def pad(A, s):
    B = np.zeros((s,s), dtype=A.dtype)
    B[:A.shape[0], :A.shape[1]] = A
    return B

def unpad(A, n):
    return A[:n,:n]

# =========================
# PSNR
# =========================

def psnr(original, restored):
    mse = np.mean((original - restored) ** 2)
    if mse == 0:
        return 99.0
    return 10 * np.log10((255**2)/mse)

# =========================
# BENCHMARK
# =========================

def benchmark(N):

    if rank == 0:
        A = np.random.randint(0,10,(N,N))
        B = np.random.randint(0,10,(N,N))
    else:
        A = None
        B = None

    A = comm.bcast(A, root=0)
    B = comm.bcast(B, root=0)

    if rank == 0:
        t0 = time.perf_counter()
        standard_np(A,B)
        t_std = (time.perf_counter()-t0)*1000

        t0 = time.perf_counter()
        strassen(A,B)
        t_seq = (time.perf_counter()-t0)*1000
    else:
        t_std = t_seq = None

    comm.Barrier()

    t0 = MPI.Wtime()
    C = strassen_mpi(A,B)
    t_par = (MPI.Wtime()-t0)*1000

    return t_std, t_seq, t_par

# =========================
# COMPRESSION METRICS (FAKE MODEL SIZE)
# =========================

def model_size_bytes(N):

    # количество операций Strassen ~ 7 * log2(N)
    ops = 7 * math.log2(N)

    # 7 матриц + padding + float64 params
    params = ops * N * 2

    return params * 8  # bytes

# =========================
# MAIN
# =========================

def main():

    sizes = [4,8,16,32,64,128,256,512,1024]

    if rank == 0:
        phys = psutil.cpu_count(logical=False)
        logi = psutil.cpu_count(logical=True)

        print(f"Physical cores: {phys}")
        print(f"Logical cores : {logi}")
        print(f"MPI processes : {size}\n")

        print(f"{'N':>6} | {'Std':>10} | {'Seq':>10} | {'MPI':>10} | {'PSNR':>8} | {'CR':>8}")
        print("-"*70)

    for N in sizes:

        t_std, t_seq, t_par = benchmark(N)

        if rank == 0:

            A = np.random.randint(0,10,(N,N))
            B = np.random.randint(0,10,(N,N))

            A2 = A.copy()
            B2 = B.copy()

            C1 = strassen(A2,B2)
            C2 = strassen_mpi(A2,B2)

            ps = psnr(C1, C2 if C2 is not None else C1)

            orig_bits = N*N*64
            comp_bits = model_size_bytes(N)*8

            cr = orig_bits / comp_bits

            print(
                f"{N:6d} | "
                f"{t_std:8.2f} | "
                f"{t_seq:8.2f} | "
                f"{t_par:8.2f} | "
                f"{ps:8.2f} | "
                f"{cr:8.2f}"
            )

if __name__ == "__main__":
    main()