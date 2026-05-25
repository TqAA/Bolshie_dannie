"""Алгоритм Штрассена умножения матриц имеет асимптотическую сложность Theta(Nlog 7),
где N – размер перемножаемых квадратных матриц. При каких N время выполнения
реализации алгоритма Штрассена на языках C и Python меньше, чем для алгоритма
стандартного перемножения квадратных матриц?
"""

import numpy as np
import time
from mpi4py import MPI
import psutil
# mpiexec -n 7 python main.py
THRESHOLD = 8


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



def standard_np(A, B):
    return A @ B



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
    return np.vstack((
        np.hstack((C11, C12)),
        np.hstack((C21, C22))
    ))


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


def strassen_mpi(A, B):

    n = A.shape[0]

    if n <= THRESHOLD:
        return A @ B

    A11, A12, A21, A22 = split(A)
    B11, B12, B21, B22 = split(B)

    tasks = [
        (add(A11, A22), add(B11, B22)),
        (add(A21, A22), B11),
        (A11, sub(B12, B22)),
        (A22, sub(B21, B11)),
        (add(A11, A12), B22),
        (sub(A21, A11), add(B11, B12)),
        (sub(A12, A22), add(B21, B22))
    ]

    local_result = None

    if rank < 7:
        X, Y = tasks[rank]
        local_result = strassen(X, Y)

    results = comm.gather(local_result, root=0)

    if rank == 0:
        M1, M2, M3, M4, M5, M6, M7 = results[:7]

        C11 = add(sub(add(M1, M4), M5), M7)
        C12 = add(M3, M5)
        C21 = add(M2, M4)
        C22 = add(sub(add(M1, M3), M2), M6)

        return combine(C11, C12, C21, C22)

    return None



def next_pow2(n):
    return 1 << (n - 1).bit_length()


def pad(A, size):
    B = np.zeros((size, size), dtype=A.dtype)
    B[:A.shape[0], :A.shape[1]] = A
    return B


def unpad(A, n):
    return A[:n, :n]


def strassen_seq(A, B):
    n = A.shape[0]

    size2 = next_pow2(n)

    A = pad(A, size2)
    B = pad(B, size2)

    return unpad(strassen(A, B), n)


def strassen_parallel(A, B):
    n = A.shape[0]

    size2 = next_pow2(n)

    A = pad(A, size2)
    B = pad(B, size2)

    C = strassen_mpi(A, B)

    if rank == 0:
        return unpad(C, n)

    return None


def benchmark(N):

    if rank == 0:
        A = np.random.randint(0, 10, (N, N))
        B = np.random.randint(0, 10, (N, N))
    else:
        A = None
        B = None

    A = comm.bcast(A, root=0)
    B = comm.bcast(B, root=0)

    if rank == 0:
        t0 = time.perf_counter()
        standard_np(A, B)
        t_std = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        strassen_seq(A, B)
        t_str = (time.perf_counter() - t0) * 1000
    else:
        t_std = None
        t_str = None

    comm.Barrier()

    t0 = MPI.Wtime()
    C = strassen_parallel(A, B)
    t_par = (MPI.Wtime() - t0) * 1000

    if rank == 0:
        return t_std, t_str, t_par

    return None




def main():

    sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

    if rank == 0:

        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)

        print(f"Physical cores: {physical}")
        print(f"Logical cores : {logical}")
        print(f"MPI processes : {size}\n")

        print(f"{'N':>6} | {'Std':>10} | {'Strassen':>10} | {'MPI Str':>10}")
        print("-" * 50)

    for N in sizes:

        result = benchmark(N)

        if rank == 0:
            t_std, t_str, t_par = result

            print(
                f"{N:6d} | "
                f"{t_std:8.2f} ms | "
                f"{t_str:8.2f} ms | "
                f"{t_par:8.2f} ms"
            )


if __name__ == "__main__":
    main()