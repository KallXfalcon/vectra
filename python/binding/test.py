import sys
sys.path.append("./")

import tensor_ops as to

if to.SSE:
    t_sse = to.SSE.TenFloat32_SSE([3, 3], 1.5)
    print("SSE Tensor:\n", t_sse)

if to.AVX:
    t_avx = to.AVX.TenFloat32_AVX([3, 3], 1.5)
    print("AVX Tensor:\n", t_avx)