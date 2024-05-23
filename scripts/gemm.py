#!/usr/bin/env python
import time
import numpy as np

N = 512
if __name__ == '__main__':
  
  A = np.random.randn(N,N).astype(np.float32)
  B = np.random.randn(N,N).astype(np.float32)

  # Multiplication [N^2 * N = N^3] 
  # Addition [N^3]
  flop = 2*N*N*N 

  for _ in range(10):
    st = time.monotonic()
    C = A@B.T
    et = time.monotonic()
    s = et-st
    print(f"{flop/s * 1e-9:6.2f} GFLOP/S, {s*1e3:6.2f} ms")