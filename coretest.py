import os
import time
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
import psutil

# --- Detect core counts ---
physical: int = psutil.cpu_count(logical=False) # type: ignore
logical: int = psutil.cpu_count(logical=True) # type: ignore

print(f"Detected physical cores: {physical}")
print(f"Detected logical cores:  {logical}")

# --- Thread candidates to test ---
# Always test around physical cores + some above/below
candidates = [1, 2, physical//2, physical, physical+2, logical]

# Remove duplicates and invalid values
candidates = sorted({int(t) for t in candidates if t >= 1})

print("\nTesting thread counts:", candidates, "\n")

# --- Heavy workload function ---
def heavy_job():
    # English comments
    a = np.random.rand(3000, 3000)
    b = np.dot(a, a.T)
    return b

results: dict[int, float] = {}

for t in candidates:
    with threadpool_limits(limits=t):
        start = time.perf_counter()
        heavy_job()
        elapsed = time.perf_counter() - start
        results[t] = elapsed
        print(f"threads={t}: {elapsed:.4f} seconds")

# --- Summary ---
print("\nThreadpool info after tests:")
print(threadpool_info())

print("\nSorted results (best first):")
for k in sorted(results, key=results.get):  # type: ignore
    print(f"{k} threads -> {results[k]:.4f} sec")
