"""
2.7. Фрактальное сжатие.
Реализовать фрактальные алгоритмы сжатия на OpenMP и MPI.
Вход алгоритма в input.jpg: данные для сжатия.
Выход алгоритма в output.jpg: сжатый файл.
"""

import os
import time
import struct
import numpy as np
from PIL import Image
from mpi4py import MPI
#  mpiexec -n 6 python .\fisher_fractal\fractal.py

RANGE_SIZE   = 8
DOMAIN_SIZE  = 16
DOMAIN_STEP  = 16
DECODE_ITERS = 20

INPUT_PATH  = "fisher_fractal/input.png"
OUTPUT_MPI  = "output_mpi.png"
FRACTAL_DAT = "fractal.dat"

# Формат одной записи: ry, rx, dy, dx (uint16) + tid (uint8) + s, o (float32)
RECORD_FMT  = ">HHHHBff"
RECORD_SIZE = struct.calcsize(RECORD_FMT)



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def load_image(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float32)

def save_image(img, path):
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)


def downsample(block):
    f = DOMAIN_SIZE // RANGE_SIZE
    return block.reshape(RANGE_SIZE, f, RANGE_SIZE, f).mean(axis=(1, 3))

def get_transforms(block):
    return [
        block,
        np.fliplr(block),
        np.flipud(block),
        np.rot90(block, 1),
        np.rot90(block, 2),
        np.rot90(block, 3),
        np.fliplr(np.rot90(block)),
        np.flipud(np.rot90(block)),
    ]



def fit(domain, rng):
    """
     МНК-подбор коэффициентов s и o
    """
    d = domain.reshape(-1).astype(np.float64)
    r = rng.reshape(-1).astype(np.float64)
    n = len(d)
    dd = np.dot(d, d)
    dr = np.dot(d, r)
    sd = d.sum()
    sr = r.sum()
    denom = n * dd - sd * sd
    if abs(denom) < 1e-10:
        s = 0.0
        o = r.mean()
    else:
        s = (n * dr - sd * sr) / denom
        o = (sr - s * sd) / n
    mse = np.mean((s * d + o - r) ** 2)
    return s, o, mse



def psnr(original, restored):
    mse = np.mean((original - restored) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((255 ** 2) / mse)



def build_domain_pool(img):
    H, W = img.shape
    pool = []
    for y in range(0, H - DOMAIN_SIZE + 1, DOMAIN_STEP):
        for x in range(0, W - DOMAIN_SIZE + 1, DOMAIN_STEP):
            block = img[y:y + DOMAIN_SIZE, x:x + DOMAIN_SIZE]
            pool.append((y, x, downsample(block)))
    return pool



def encode_chunk(img, ranges, domain_pool):
    records = []
    for i, (ry, rx) in enumerate(ranges):
        r_block  = img[ry:ry + RANGE_SIZE, rx:rx + RANGE_SIZE]
        best     = None
        best_err = float("inf")
        for dy, dx, d_small in domain_pool:
            for tid, t in enumerate(get_transforms(d_small)):
                s, o, err = fit(t, r_block)
                if err < best_err:
                    best_err = err
                    best = (ry, rx, dy, dx, tid, float(s), float(o))
        records.append(best)
        if i % 50 == 0:
            print(f"[rank {rank}] {i}/{len(ranges)}", flush=True)
    return records



def save_fractal(records, path):
    """
    Сохраняет записи в бинарнике
    Формат: ry(u16) rx(u16) dy(u16) dx(u16) tid(u8) s(f32) o(f32)
    """
    with open(path, "wb") as f:
        # Заголовок: число записей (uint32)
        f.write(struct.pack(">I", len(records)))
        for ry, rx, dy, dx, tid, s, o in records:
            f.write(struct.pack(RECORD_FMT, ry, rx, dy, dx, tid, s, o))

def load_fractal(path):
    """Читает бинарный файл и возвращает список записей"""
    records = []
    with open(path, "rb") as f:
        n = struct.unpack(">I", f.read(4))[0]
        for _ in range(n):
            rec = struct.unpack(RECORD_FMT, f.read(RECORD_SIZE))
            records.append(rec)
    return records



def decode(records, shape):
    H, W = shape
    img = np.full((H, W), 128.0, dtype=np.float32)
    for _ in range(DECODE_ITERS):
        new_img = img.copy()
        for ry, rx, dy, dx, tid, s, o in records:
            block   = img[dy:dy + DOMAIN_SIZE, dx:dx + DOMAIN_SIZE]
            d_small = downsample(block)
            t       = get_transforms(d_small)[tid]
            new_img[ry:ry + RANGE_SIZE, rx:rx + RANGE_SIZE] = s * t + o
        img = np.clip(new_img, 0, 255)
    return img



def run_mpi(img, ranges, domain_pool):

    if rank == 0:
        chunks = [[] for _ in range(size)]
        for i, r in enumerate(ranges):
            chunks[i % size].append(r)
    else:
        chunks = None

    my_ranges = comm.scatter(chunks, root=0)

    comm.Barrier()
    encode_start = MPI.Wtime()

    my_records = encode_chunk(img, my_ranges, domain_pool)

    comm.Barrier()
    encode_time = MPI.Wtime() - encode_start

    all_records = comm.gather(my_records, root=0)

    if rank == 0:
        records = [item for chunk in all_records for item in chunk]

        # Бинарное сохранение
        save_fractal(records, FRACTAL_DAT)

        decode_start = time.perf_counter()
        restored     = decode(records, img.shape)
        decode_time  = time.perf_counter() - decode_start

        save_image(restored, OUTPUT_MPI)


        quality    = psnr(img, restored)

        original_size = os.path.getsize(INPUT_PATH)
        fractal_size  = os.path.getsize(FRACTAL_DAT)
        ratio         = original_size / fractal_size


        print("\n==============================")
        print("DONE")
        print("==============================")
        print(f"Encoding time  : {encode_time:.2f} sec")
        print(f"Decoding time  : {decode_time:.2f} sec")
        print(f"PSNR           : {quality:.2f} dB")
        print()
        print(f"Original size  : {original_size / 1024:.2f} KB")
        print(f"Fractal size   : {fractal_size / 1024:.2f} KB")
        print(f"Compression    : {ratio:.2f}x")




def main():
    img  = load_image(INPUT_PATH)
    H, W = img.shape

    ranges = [
        (y, x)
        for y in range(0, H - RANGE_SIZE + 1, RANGE_SIZE)
        for x in range(0, W - RANGE_SIZE + 1, RANGE_SIZE)
    ]

    domain_pool = build_domain_pool(img)

    if rank == 0:
        print("==============================")
        print("Fractal Compression MPI")
        print("==============================")
        print(f"Image size     : {W}x{H}")
        print(f"Range blocks   : {len(ranges)}")
        print(f"Domain blocks  : {len(domain_pool)}")
        print(f"MPI processes  : {size}")
        print()

    comm.Barrier()
    run_mpi(img, ranges, domain_pool)

if __name__ == "__main__":
    main()