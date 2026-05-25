"""
Задача 3.9 -- Частотный анализ текстов для проверки гипотезы об авторстве
    mpiexec -n 6 python .\frequency\frequency.py
    chcp 65001
"""

import re
import html
import math
import time
import unicodedata
import collections
import numpy as np
from mpi4py import MPI
import sys
import nltk
nltk.download('stopwords', quiet=True)
sys.stdout.reconfigure(encoding='utf-8')


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



TEXT1_PATH = "frequency/text1.txt"
TEXT2_PATH = "frequency/text2.txt"

TOP_N = 50


def load_stopwords():
    from nltk.corpus import stopwords
    return set(stopwords.words("russian"))

STOP_WORDS = load_stopwords()


def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[/?[a-zA-Z][^\]]*\]", " ", text)
    text = re.sub(r"[*_]{1,3}(.+?)[*_]{1,3}", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u00a0\u200b\u200c\u200d\u2009\u202f\u3000\t\r]", " ", text)
    text = re.sub(r"[\u00ad\u200e\u200f\ufeff\u034f]", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\b[IVXLCDM]{2,}\b", " ", text)
    text = re.sub(
        r"^\s*(страница|глава|часть|раздел|chapter|page)[\s\d]*$",
        " ", text, flags=re.MULTILINE | re.IGNORECASE
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^а-яё\s]", " ", text)
    words = text.split()
    return [w for w in words if w not in STOP_WORDS and len(w) >= 3]

def count_words(words):
    return collections.Counter(words)

def merge_counters(cs):
    total = collections.Counter()
    for c in cs:
        total.update(c)
    return total



def get_top_vocab(f1, f2, top_n):
    top1 = set(w for w, _ in f1.most_common(top_n))
    top2 = set(w for w, _ in f2.most_common(top_n))
    return sorted(top1 | top2)

def build_vector(freq, vocab, total):
    return np.array([freq.get(w, 0) / total for w in vocab], dtype=float)

def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return dot / (n1 * n2) if n1 and n2 else 0.0


def run_sequential():
    if rank != 0:
        return None, None

    t0 = time.perf_counter()

    with open(TEXT1_PATH, encoding="utf-8") as f:
        t1 = f.read()
    with open(TEXT2_PATH, encoding="utf-8") as f:
        t2 = f.read()

    t1 = clean_text(t1)
    t2 = clean_text(t2)

    w1 = preprocess(t1)
    w2 = preprocess(t2)

    f1 = count_words(w1)
    f2 = count_words(w2)

    vocab  = get_top_vocab(f1, f2, TOP_N)
    v1     = build_vector(f1, vocab, len(w1))
    v2     = build_vector(f2, vocab, len(w2))
    cosine = cosine_similarity(v1, v2)

    t_seq = time.perf_counter() - t0

    print(f"Топ-10 слов (текст 1) : {[w for w, _ in f1.most_common(15)]}")
    print(f"Топ-10 слов (текст 2) : {[w for w, _ in f2.most_common(15)]}")

    return t_seq, cosine


def run_parallel():
    comm.Barrier()
    t_start = MPI.Wtime()

    with open(TEXT1_PATH, encoding="utf-8") as f:
        l1 = f.readlines()
    with open(TEXT2_PATH, encoding="utf-8") as f:
        l2 = f.readlines()

    def chunk(lines):
        n = math.ceil(len(lines) / size)
        return lines[rank*n:(rank+1)*n]

    t1 = clean_text(" ".join(chunk(l1)))
    t2 = clean_text(" ".join(chunk(l2)))

    w1 = preprocess(t1)
    w2 = preprocess(t2)

    f1 = count_words(w1)
    f2 = count_words(w2)

    g1     = comm.gather(f1, root=0)
    g2     = comm.gather(f2, root=0)
    total1 = comm.reduce(len(w1), MPI.SUM, root=0)
    total2 = comm.reduce(len(w2), MPI.SUM, root=0)

    comm.Barrier()
    t_parallel = MPI.Wtime() - t_start

    if rank == 0:
        f1 = merge_counters(g1)
        f2 = merge_counters(g2)

        vocab  = get_top_vocab(f1, f2, TOP_N)
        v1     = build_vector(f1, vocab, total1)
        v2     = build_vector(f2, vocab, total2)
        cosine = cosine_similarity(v1, v2)

        return t_parallel, cosine

    return None, None



def main():


    if rank == 0:
        t_seq, cos_seq = run_sequential()

    comm.Barrier()


    t_par, cos_par = run_parallel()

    if rank == 0:
        speedup = t_seq / t_par if t_par and t_par > 0 else 0

        print(f"\nВремя последовательно        : {t_seq:.3f} сек")
        print(f"Время параллельно             : {t_par:.3f} сек")
        print(f"Ускорение                     : {speedup:.2f}x  ({size} процессов)")
        print(f"Косинусное сходство (посл/пар): {cos_seq:.6f} / {cos_par:.6f}")

if __name__ == "__main__":
    main()