#include "iir2d_core.h"

#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cmath>
#include <cstdio>
#include <cstring>

#define CUDA_CHECK_RET(call) do { \
    int _err = (int)(call); \
    if (_err != 0) return (_err < 0) ? _err : IIR2D_STATUS_CUDA_ERROR; \
} while (0)

struct WorkspaceCache {
    int device = -1;
    size_t bytes = 0;
    void* buf = nullptr;
};

static inline int get_workspace_bytes(size_t bytes, void** out) {
    thread_local WorkspaceCache cache;
    if (!out) return IIR2D_STATUS_NULL_POINTER;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return IIR2D_STATUS_CUDA_ERROR;
    if (cache.device != dev || cache.bytes < bytes) {
        if (cache.buf && cudaFree(cache.buf) != cudaSuccess) return IIR2D_STATUS_WORKSPACE_ERROR;
        if (cudaMalloc(&cache.buf, bytes) != cudaSuccess) return IIR2D_STATUS_WORKSPACE_ERROR;
        cache.device = dev;
        cache.bytes = bytes;
    }
    *out = cache.buf;
    return IIR2D_STATUS_OK;
}

static inline size_t align_up(size_t v, size_t a) {
    return (v + (a - 1)) & ~(a - 1);
}

// ---- border sampling ----
__device__ __forceinline__ float border_sample_f(const float* row, int n, int idx, int mode, float constant_val) {
    if (idx >= 0 && idx < n) return row[idx];
    if (mode == IIR2D_BORDER_CONSTANT) return constant_val;
    if (mode == IIR2D_BORDER_CLAMP) {
        int ci = idx < 0 ? 0 : (n - 1);
        return row[ci];
    }
    if (mode == IIR2D_BORDER_WRAP) {
        int m = idx % n;
        if (m < 0) m += n;
        return row[m];
    }
    int period = n * 2;
    int m = idx % period;
    if (m < 0) m += period;
    if (m >= n) m = period - 1 - m;
    return row[m];
}

__device__ __forceinline__ double border_sample_d(const double* row, int n, int idx, int mode, double constant_val) {
    if (idx >= 0 && idx < n) return row[idx];
    if (mode == IIR2D_BORDER_CONSTANT) return constant_val;
    if (mode == IIR2D_BORDER_CLAMP) {
        int ci = idx < 0 ? 0 : (n - 1);
        return row[ci];
    }
    if (mode == IIR2D_BORDER_WRAP) {
        int m = idx % n;
        if (m < 0) m += n;
        return row[m];
    }
    int period = n * 2;
    int m = idx % period;
    if (m < 0) m += period;
    if (m >= n) m = period - 1 - m;
    return row[m];
}

// ---- scan-based helpers (f32) ----
struct Affine {
    float a;
    float p;
};

struct AffineOp {
    __device__ __forceinline__ Affine operator()(const Affine& left, const Affine& right) const {
        return {right.a * left.a, right.a * left.p + right.p};
    }
};

struct AffineD {
    double a;
    double p;
};

struct AffineOpD {
    __device__ __forceinline__ AffineD operator()(const AffineD& left, const AffineD& right) const {
        return {right.a * left.a, right.a * left.p + right.p};
    }
};

template <int BLOCK>
__global__ void iir_row_local_scan(
    const float* __restrict__ x,
    float* __restrict__ y,
    int width,
    int height,
    float b0,
    float b1,
    float a1,
    int border_mode,
    float border_const,
    float* __restrict__ block_a,
    float* __restrict__ block_p
) {
    int row = blockIdx.y;
    int block = blockIdx.x;
    int tid = threadIdx.x;
    int start = block * BLOCK;
    int idx = row * width + start + tid;
    if (row >= height || start >= width) return;

    bool active = (start + tid) < width;
    float xi = 0.0f;
    float xim1 = 0.0f;
    if (active) {
        xi = x[idx];
        if (start + tid > 0) {
            xim1 = x[idx - 1];
        } else {
            xim1 = border_sample_f(x + size_t(row) * size_t(width), width, -1, border_mode, border_const);
        }
    }

    Affine in;
    if (active) {
        float p = b0 * xi + b1 * xim1;
        in = {a1, p};
    } else {
        in = {1.0f, 0.0f};
    }

    using BlockScan = cub::BlockScan<Affine, BLOCK>;
    __shared__ typename BlockScan::TempStorage temp;
    Affine out;
    BlockScan(temp).InclusiveScan(in, out, AffineOp());
    if (active) y[idx] = out.p;

    int block_len = width - start;
    if (block_len > BLOCK) block_len = BLOCK;
    int last = block_len - 1;
    if (tid == last) {
        int bidx = row * gridDim.x + block;
        block_a[bidx] = out.a;
        block_p[bidx] = out.p;
    }
}

template <int BLOCK>
__global__ void iir_row_local_scan_d(
    const double* __restrict__ x,
    double* __restrict__ y,
    int width,
    int height,
    double b0,
    double b1,
    double a1,
    int border_mode,
    double border_const,
    double* __restrict__ block_a,
    double* __restrict__ block_p
) {
    int row = blockIdx.y;
    int block = blockIdx.x;
    int tid = threadIdx.x;
    int start = block * BLOCK;
    int idx = row * width + start + tid;
    if (row >= height || start >= width) return;

    bool active = (start + tid) < width;
    double xi = 0.0;
    double xim1 = 0.0;
    if (active) {
        xi = x[idx];
        if (start + tid > 0) {
            xim1 = x[idx - 1];
        } else {
            xim1 = border_sample_d(x + size_t(row) * size_t(width), width, -1, border_mode, border_const);
        }
    }

    AffineD in;
    if (active) {
        double p = b0 * xi + b1 * xim1;
        in = {a1, p};
    } else {
        in = {1.0, 0.0};
    }

    using BlockScan = cub::BlockScan<AffineD, BLOCK>;
    __shared__ typename BlockScan::TempStorage temp;
    AffineD out;
    BlockScan(temp).InclusiveScan(in, out, AffineOpD());
    if (active) y[idx] = out.p;

    int block_len = width - start;
    if (block_len > BLOCK) block_len = BLOCK;
    int last = block_len - 1;
    if (tid == last) {
        int bidx = row * gridDim.x + block;
        block_a[bidx] = out.a;
        block_p[bidx] = out.p;
    }
}

__global__ void iir_row_block_prefix_scan(
    const float* __restrict__ block_a,
    const float* __restrict__ block_p,
    float* __restrict__ block_in,
    int num_blocks,
    int height,
    const float* __restrict__ row_yprev
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    float y_in = row_yprev ? row_yprev[row] : 0.0f;
    int base = row * num_blocks;
    for (int b = 0; b < num_blocks; ++b) {
        block_in[base + b] = y_in;
        y_in = block_a[base + b] * y_in + block_p[base + b];
    }
}

__global__ void iir_row_block_prefix_scan_d(
    const double* __restrict__ block_a,
    const double* __restrict__ block_p,
    double* __restrict__ block_in,
    int num_blocks,
    int height,
    const double* __restrict__ row_yprev
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    double y_in = row_yprev ? row_yprev[row] : 0.0;
    int base = row * num_blocks;
    for (int b = 0; b < num_blocks; ++b) {
        block_in[base + b] = y_in;
        y_in = block_a[base + b] * y_in + block_p[base + b];
    }
}

template <int BLOCK>
__global__ void iir_row_fixup_scan(
    float* __restrict__ y,
    int width,
    int height,
    float a1,
    const float* __restrict__ block_in
) {
    int row = blockIdx.y;
    int block = blockIdx.x;
    int tid = threadIdx.x;
    int start = block * BLOCK;
    int idx = row * width + start + tid;
    if (row >= height || start >= width) return;
    if (start + tid < width) {
        float y_in = block_in[row * gridDim.x + block];
        float scale = powf(a1, float(tid + 1));
        y[idx] += y_in * scale;
    }
}

template <int BLOCK>
__global__ void iir_row_fixup_scan_d(
    double* __restrict__ y,
    int width,
    int height,
    double a1,
    const double* __restrict__ block_in
) {
    int row = blockIdx.y;
    int block = blockIdx.x;
    int tid = threadIdx.x;
    int start = block * BLOCK;
    int idx = row * width + start + tid;
    if (row >= height || start >= width) return;
    if (start + tid < width) {
        double y_in = block_in[row * gridDim.x + block];
        double scale = pow(a1, double(tid + 1));
        y[idx] += y_in * scale;
    }
}

__global__ void row_init_first_f(
    const float* __restrict__ x,
    int width,
    int height,
    float b0,
    float b1,
    float a1,
    int border_mode,
    float border_const,
    float* __restrict__ row_yprev
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const float* in = x + size_t(row) * size_t(width);
    float xm1 = border_sample_f(in, width, -1, border_mode, border_const);
    float xm2 = border_sample_f(in, width, -2, border_mode, border_const);
    float y_m1 = b0 * xm1 + b1 * xm2;
    row_yprev[row] = y_m1;
}

__global__ void row_init_biquad_f(
    const float* __restrict__ x,
    int width,
    int height,
    float b0, float b1, float b2,
    float a1, float a2,
    int border_mode,
    float border_const,
    float4* __restrict__ row_state
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const float* in = x + size_t(row) * size_t(width);
    float xm1 = border_sample_f(in, width, -1, border_mode, border_const);
    float xm2 = border_sample_f(in, width, -2, border_mode, border_const);
    float xm3 = border_sample_f(in, width, -3, border_mode, border_const);
    float xm4 = border_sample_f(in, width, -4, border_mode, border_const);
    float y2 = b0 * xm2 + b1 * xm3 + b2 * xm4;
    float y1 = b0 * xm1 + b1 * xm2 + b2 * xm3 + a1 * y2;
    row_state[row] = make_float4(xm1, xm2, y1, y2);
}

__global__ void row_init_first_d(
    const double* __restrict__ x,
    int width,
    int height,
    double b0,
    double b1,
    double a1,
    int border_mode,
    double border_const,
    double* __restrict__ row_yprev
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const double* in = x + size_t(row) * size_t(width);
    double xm1 = border_sample_d(in, width, -1, border_mode, border_const);
    double xm2 = border_sample_d(in, width, -2, border_mode, border_const);
    double y_m1 = b0 * xm1 + b1 * xm2;
    row_yprev[row] = y_m1;
}

__global__ void row_init_biquad_d(
    const double* __restrict__ x,
    int width,
    int height,
    double b0, double b1, double b2,
    double a1, double a2,
    int border_mode,
    double border_const,
    double4* __restrict__ row_state
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const double* in = x + size_t(row) * size_t(width);
    double xm1 = border_sample_d(in, width, -1, border_mode, border_const);
    double xm2 = border_sample_d(in, width, -2, border_mode, border_const);
    double xm3 = border_sample_d(in, width, -3, border_mode, border_const);
    double xm4 = border_sample_d(in, width, -4, border_mode, border_const);
    double y2 = b0 * xm2 + b1 * xm3 + b2 * xm4;
    double y1 = b0 * xm1 + b1 * xm2 + b2 * xm3 + a1 * y2;
    row_state[row] = make_double4(xm1, xm2, y1, y2);
}

__global__ void convert_f2d(const float* __restrict__ in, double* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (double)in[i];
}

__global__ void convert_d2f(const double* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (float)in[i];
}

struct Transform4 {
    float a00, a01, a02, a03;
    float a10, a11, a12, a13;
    float a20, a21, a22, a23;
    float a30, a31, a32, a33;
    float b0, b1, b2, b3;
};

struct Transform4D {
    double a00, a01, a02, a03;
    double a10, a11, a12, a13;
    double a20, a21, a22, a23;
    double a30, a31, a32, a33;
    double b0, b1, b2, b3;
};

struct Transform4OpD {
    __device__ __forceinline__ Transform4D operator()(const Transform4D& left, const Transform4D& right) const {
        Transform4D o;
        // Prefix scan composes left->right; return (right ∘ left).
        o.a00 = right.a00*left.a00 + right.a01*left.a10 + right.a02*left.a20 + right.a03*left.a30;
        o.a01 = right.a00*left.a01 + right.a01*left.a11 + right.a02*left.a21 + right.a03*left.a31;
        o.a02 = right.a00*left.a02 + right.a01*left.a12 + right.a02*left.a22 + right.a03*left.a32;
        o.a03 = right.a00*left.a03 + right.a01*left.a13 + right.a02*left.a23 + right.a03*left.a33;
        o.a10 = right.a10*left.a00 + right.a11*left.a10 + right.a12*left.a20 + right.a13*left.a30;
        o.a11 = right.a10*left.a01 + right.a11*left.a11 + right.a12*left.a21 + right.a13*left.a31;
        o.a12 = right.a10*left.a02 + right.a11*left.a12 + right.a12*left.a22 + right.a13*left.a32;
        o.a13 = right.a10*left.a03 + right.a11*left.a13 + right.a12*left.a23 + right.a13*left.a33;
        o.a20 = right.a20*left.a00 + right.a21*left.a10 + right.a22*left.a20 + right.a23*left.a30;
        o.a21 = right.a20*left.a01 + right.a21*left.a11 + right.a22*left.a21 + right.a23*left.a31;
        o.a22 = right.a20*left.a02 + right.a21*left.a12 + right.a22*left.a22 + right.a23*left.a32;
        o.a23 = right.a20*left.a03 + right.a21*left.a13 + right.a22*left.a23 + right.a23*left.a33;
        o.a30 = right.a30*left.a00 + right.a31*left.a10 + right.a32*left.a20 + right.a33*left.a30;
        o.a31 = right.a30*left.a01 + right.a31*left.a11 + right.a32*left.a21 + right.a33*left.a31;
        o.a32 = right.a30*left.a02 + right.a31*left.a12 + right.a32*left.a22 + right.a33*left.a32;
        o.a33 = right.a30*left.a03 + right.a31*left.a13 + right.a32*left.a23 + right.a33*left.a33;
        o.b0 = right.a00*left.b0 + right.a01*left.b1 + right.a02*left.b2 + right.a03*left.b3 + right.b0;
        o.b1 = right.a10*left.b0 + right.a11*left.b1 + right.a12*left.b2 + right.a13*left.b3 + right.b1;
        o.b2 = right.a20*left.b0 + right.a21*left.b1 + right.a22*left.b2 + right.a23*left.b3 + right.b2;
        o.b3 = right.a30*left.b0 + right.a31*left.b1 + right.a32*left.b2 + right.a33*left.b3 + right.b3;
        return o;
    }
};

struct Transform4Op {
    __device__ __forceinline__ Transform4 operator()(const Transform4& left, const Transform4& right) const {
        Transform4 o;
        // Prefix scan composes left->right; return (right ∘ left).
        o.a00 = right.a00*left.a00 + right.a01*left.a10 + right.a02*left.a20 + right.a03*left.a30;
        o.a01 = right.a00*left.a01 + right.a01*left.a11 + right.a02*left.a21 + right.a03*left.a31;
        o.a02 = right.a00*left.a02 + right.a01*left.a12 + right.a02*left.a22 + right.a03*left.a32;
        o.a03 = right.a00*left.a03 + right.a01*left.a13 + right.a02*left.a23 + right.a03*left.a33;
        o.a10 = right.a10*left.a00 + right.a11*left.a10 + right.a12*left.a20 + right.a13*left.a30;
        o.a11 = right.a10*left.a01 + right.a11*left.a11 + right.a12*left.a21 + right.a13*left.a31;
        o.a12 = right.a10*left.a02 + right.a11*left.a12 + right.a12*left.a22 + right.a13*left.a32;
        o.a13 = right.a10*left.a03 + right.a11*left.a13 + right.a12*left.a23 + right.a13*left.a33;
        o.a20 = right.a20*left.a00 + right.a21*left.a10 + right.a22*left.a20 + right.a23*left.a30;
        o.a21 = right.a20*left.a01 + right.a21*left.a11 + right.a22*left.a21 + right.a23*left.a31;
        o.a22 = right.a20*left.a02 + right.a21*left.a12 + right.a22*left.a22 + right.a23*left.a32;
        o.a23 = right.a20*left.a03 + right.a21*left.a13 + right.a22*left.a23 + right.a23*left.a33;
        o.a30 = right.a30*left.a00 + right.a31*left.a10 + right.a32*left.a20 + right.a33*left.a30;
        o.a31 = right.a30*left.a01 + right.a31*left.a11 + right.a32*left.a21 + right.a33*left.a31;
        o.a32 = right.a30*left.a02 + right.a31*left.a12 + right.a32*left.a22 + right.a33*left.a32;
        o.a33 = right.a30*left.a03 + right.a31*left.a13 + right.a32*left.a23 + right.a33*left.a33;
        o.b0 = right.a00*left.b0 + right.a01*left.b1 + right.a02*left.b2 + right.a03*left.b3 + right.b0;
        o.b1 = right.a10*left.b0 + right.a11*left.b1 + right.a12*left.b2 + right.a13*left.b3 + right.b1;
        o.b2 = right.a20*left.b0 + right.a21*left.b1 + right.a22*left.b2 + right.a23*left.b3 + right.b2;
        o.b3 = right.a30*left.b0 + right.a31*left.b1 + right.a32*left.b2 + right.a33*left.b3 + right.b3;
        return o;
    }
};

template <int BLOCK>
__global__ void biquad_block_transform4(
    const float* __restrict__ x,
    int width,
    int height,
    float b0, float b1, float b2,
    float a1, float a2,
    Transform4* __restrict__ block_t
) {
    int row = blockIdx.y;
    int block = blockIdx.x;
    if (row >= height) return;
    int start = block * blockDim.x;
    if (start >= width) return;

    int tid = threadIdx.x;
    int idx = row * width + start + tid;
    bool active = (start + tid) < width;

    Transform4 in;
    if (active) {
        float xi = x[idx];
        in = {
            0.0f, 0.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 0.0f,
            b1,   b2,   a1,   a2,
            0.0f, 0.0f, 1.0f, 0.0f,
            1.0f * xi, 0.0f, b0 * xi, 0.0f
        };
    } else {
        in = {
            1.0f,0.0f,0.0f,0.0f,
            0.0f,1.0f,0.0f,0.0f,
            0.0f,0.0f,1.0f,0.0f,
            0.0f,0.0f,0.0f,1.0f,
            0.0f,0.0f,0.0f,0.0f
        };
    }

    using BlockScan = cub::BlockScan<Transform4, BLOCK>;
    __shared__ typename BlockScan::TempStorage temp;
    Transform4 out;
    BlockScan(temp).InclusiveScan(in, out, Transform4Op());

    int end = start + blockDim.x;
    if (end > width) end = width;
    int last = end - start - 1;
    if (active && tid == last) {
        int bidx = row * gridDim.x + block;
        block_t[bidx] = out;
    }
}

template <int BLOCK>
__global__ void biquad_block_transform4_d(
    const double* __restrict__ x,
    int width,
    int height,
    double b0, double b1, double b2,
    double a1, double a2,
    Transform4D* __restrict__ block_t
) {
    int row = blockIdx.y;
    int block = blockIdx.x;
    if (row >= height) return;
    int start = block * blockDim.x;
    if (start >= width) return;

    int tid = threadIdx.x;
    int idx = row * width + start + tid;
    bool active = (start + tid) < width;

    Transform4D in;
    if (active) {
        double xi = x[idx];
        in = {
            0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            b1,  b2,  a1,  a2,
            0.0, 0.0, 1.0, 0.0,
            1.0 * xi, 0.0, b0 * xi, 0.0
        };
    } else {
        in = {
            1.0,0.0,0.0,0.0,
            0.0,1.0,0.0,0.0,
            0.0,0.0,1.0,0.0,
            0.0,0.0,0.0,1.0,
            0.0,0.0,0.0,0.0
        };
    }

    using BlockScan = cub::BlockScan<Transform4D, BLOCK>;
    __shared__ typename BlockScan::TempStorage temp;
    Transform4D out;
    BlockScan(temp).InclusiveScan(in, out, Transform4OpD());

    int end = start + blockDim.x;
    if (end > width) end = width;
    int last = end - start - 1;
    if (active && tid == last) {
        int bidx = row * gridDim.x + block;
        block_t[bidx] = out;
    }
}

__global__ void biquad_block_prefix4(
    const Transform4* __restrict__ block_t,
    float4* __restrict__ block_in,
    int num_blocks,
    int height,
    const float4* __restrict__ row_state
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    float4 s = row_state ? row_state[row] : make_float4(0,0,0,0);
    int base = row * num_blocks;
    for (int b = 0; b < num_blocks; ++b) {
        block_in[base + b] = s;
        const Transform4& t = block_t[base + b];
        float ns0 = t.a00 * s.x + t.a01 * s.y + t.a02 * s.z + t.a03 * s.w + t.b0;
        float ns1 = t.a10 * s.x + t.a11 * s.y + t.a12 * s.z + t.a13 * s.w + t.b1;
        float ns2 = t.a20 * s.x + t.a21 * s.y + t.a22 * s.z + t.a23 * s.w + t.b2;
        float ns3 = t.a30 * s.x + t.a31 * s.y + t.a32 * s.z + t.a33 * s.w + t.b3;
        s = make_float4(ns0, ns1, ns2, ns3);
    }
}

__global__ void biquad_block_prefix4_d(
    const Transform4D* __restrict__ block_t,
    double4* __restrict__ block_in,
    int num_blocks,
    int height,
    const double4* __restrict__ row_state
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    double4 s = row_state ? row_state[row] : make_double4(0,0,0,0);
    int base = row * num_blocks;
    for (int b = 0; b < num_blocks; ++b) {
        block_in[base + b] = s;
        const Transform4D& t = block_t[base + b];
        double ns0 = t.a00 * s.x + t.a01 * s.y + t.a02 * s.z + t.a03 * s.w + t.b0;
        double ns1 = t.a10 * s.x + t.a11 * s.y + t.a12 * s.z + t.a13 * s.w + t.b1;
        double ns2 = t.a20 * s.x + t.a21 * s.y + t.a22 * s.z + t.a23 * s.w + t.b2;
        double ns3 = t.a30 * s.x + t.a31 * s.y + t.a32 * s.z + t.a33 * s.w + t.b3;
        s = make_double4(ns0, ns1, ns2, ns3);
    }
}

template <int BLOCK>
__global__ void biquad_block_fixup4_d(
    const double* __restrict__ x,
    double* __restrict__ y,
    int width,
    int height,
    double b0, double b1, double b2,
    double a1, double a2,
    const double4* __restrict__ block_in
) {
    int row = blockIdx.y;
    int block = blockIdx.x;
    if (row >= height) return;
    int start = block * blockDim.x;
    if (start >= width) return;

    int tid = threadIdx.x;
    int idx = row * width + start + tid;
    bool active = (start + tid) < width;

    Transform4D in;
    if (active) {
        double xi = x[idx];
        in = {
            0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            b1,  b2,  a1,  a2,
            0.0, 0.0, 1.0, 0.0,
            1.0 * xi, 0.0, b0 * xi, 0.0
        };
    } else {
        in = {
            1.0,0.0,0.0,0.0,
            0.0,1.0,0.0,0.0,
            0.0,0.0,1.0,0.0,
            0.0,0.0,0.0,1.0,
            0.0,0.0,0.0,0.0
        };
    }

    using BlockScan = cub::BlockScan<Transform4D, BLOCK>;
    __shared__ typename BlockScan::TempStorage temp;
    Transform4D out;
    BlockScan(temp).InclusiveScan(in, out, Transform4OpD());

    if (active) {
        double4 s = block_in[row * gridDim.x + block];
        double y2 = out.a20 * s.x + out.a21 * s.y + out.a22 * s.z + out.a23 * s.w + out.b2;
        y[idx] = y2;
    }
}

template <int BLOCK>
__global__ void biquad_block_fixup4(
    const float* __restrict__ x,
    float* __restrict__ y,
    int width,
    int height,
    float b0, float b1, float b2,
    float a1, float a2,
    const float4* __restrict__ block_in
) {
    int row = blockIdx.y;
    int block = blockIdx.x;
    if (row >= height) return;
    int start = block * blockDim.x;
    if (start >= width) return;

    int tid = threadIdx.x;
    int idx = row * width + start + tid;
    bool active = (start + tid) < width;

    Transform4 in;
    if (active) {
        float xi = x[idx];
        in = {
            0.0f, 0.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 0.0f,
            b1,   b2,   a1,   a2,
            0.0f, 0.0f, 1.0f, 0.0f,
            1.0f * xi, 0.0f, b0 * xi, 0.0f
        };
    } else {
        in = {
            1.0f,0.0f,0.0f,0.0f,
            0.0f,1.0f,0.0f,0.0f,
            0.0f,0.0f,1.0f,0.0f,
            0.0f,0.0f,0.0f,1.0f,
            0.0f,0.0f,0.0f,0.0f
        };
    }

    using BlockScan = cub::BlockScan<Transform4, BLOCK>;
    __shared__ typename BlockScan::TempStorage temp;
    Transform4 out;
    BlockScan(temp).InclusiveScan(in, out, Transform4Op());

    if (active) {
        float4 s = block_in[row * gridDim.x + block];
        float y0 = out.a00 * s.x + out.a01 * s.y + out.a02 * s.z + out.a03 * s.w + out.b0;
        float y1 = out.a10 * s.x + out.a11 * s.y + out.a12 * s.z + out.a13 * s.w + out.b1;
        float y2 = out.a20 * s.x + out.a21 * s.y + out.a22 * s.z + out.a23 * s.w + out.b2;
        y[idx] = y2;
        (void)y0; (void)y1;
    }
}

// ---- kernels (f32) ----
__global__ void iir_row_first_f(const float* x, float* y, int width, int height, float b0, float b1, float a1, int border_mode, float border_const) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const float* in = x + size_t(row) * size_t(width);
    float* out = y + size_t(row) * size_t(width);
    float xm1 = border_sample_f(in, width, -1, border_mode, border_const);
    float xm2 = border_sample_f(in, width, -2, border_mode, border_const);
    float yprev = b0 * xm1 + b1 * xm2;
    float xprev = xm1;
    for (int i = 0; i < width; ++i) {
        float xi = in[i];
        float yi = b0 * xi + b1 * xprev + a1 * yprev;
        out[i] = yi;
        xprev = xi;
        yprev = yi;
    }
}

__global__ void iir_row_biquad_f(const float* x, float* y, int width, int height, float b0, float b1, float b2, float a1, float a2, int border_mode, float border_const) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const float* in = x + size_t(row) * size_t(width);
    float* out = y + size_t(row) * size_t(width);
    float xm1 = border_sample_f(in, width, -1, border_mode, border_const);
    float xm2 = border_sample_f(in, width, -2, border_mode, border_const);
    float xm3 = border_sample_f(in, width, -3, border_mode, border_const);
    float xm4 = border_sample_f(in, width, -4, border_mode, border_const);
    float y2 = b0 * xm2 + b1 * xm3 + b2 * xm4;
    float y1 = b0 * xm1 + b1 * xm2 + b2 * xm3 + a1 * y2;
    float x1 = xm1, x2 = xm2;
    for (int i = 0; i < width; ++i) {
        float xi = in[i];
        float yi = b0 * xi + b1 * x1 + b2 * x2 + a1 * y1 + a2 * y2;
        out[i] = yi;
        x2 = x1; x1 = xi;
        y2 = y1; y1 = yi;
    }
}

__global__ void iir_row_fwd_bwd_f(const float* x, float* y, int width, int height, float b0, float b1, float a1, int border_mode, float border_const) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const float* in = x + size_t(row) * size_t(width);
    float* out = y + size_t(row) * size_t(width);
    float xm1 = border_sample_f(in, width, -1, border_mode, border_const);
    float xm2 = border_sample_f(in, width, -2, border_mode, border_const);
    float yprev = b0 * xm1 + b1 * xm2;
    float xprev = xm1;
    for (int i = 0; i < width; ++i) {
        float xi = in[i];
        float yi = b0 * xi + b1 * xprev + a1 * yprev;
        out[i] = yi;
        xprev = xi;
        yprev = yi;
    }
    float xp1 = border_sample_f(out, width, width, border_mode, border_const);
    float xp2 = border_sample_f(out, width, width + 1, border_mode, border_const);
    yprev = b0 * xp1 + b1 * xp2;
    xprev = xp1;
    for (int i = width - 1; i >= 0; --i) {
        float xi = out[i];
        float yi = b0 * xi + b1 * xprev + a1 * yprev;
        out[i] = yi;
        xprev = xi;
        yprev = yi;
    }
}

__global__ void iir_row_statespace_f(const float* x, float* y, int width, int height, float b0, float b1, float b2, float a1, float a2, int border_mode, float border_const) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const float* in = x + size_t(row) * size_t(width);
    float* out = y + size_t(row) * size_t(width);
    float xm2 = border_sample_f(in, width, -2, border_mode, border_const);
    float xm1 = border_sample_f(in, width, -1, border_mode, border_const);
    float z1 = 0.0f, z2 = 0.0f;
    for (int k = 0; k < 2; ++k) {
        float xi = (k == 0) ? xm2 : xm1;
        float yi = b0 * xi + z1;
        z1 = b1 * xi + z2 + a1 * yi;
        z2 = b2 * xi + a2 * yi;
    }
    for (int i = 0; i < width; ++i) {
        float xi = in[i];
        float yi = b0 * xi + z1;
        z1 = b1 * xi + z2 + a1 * yi;
        z2 = b2 * xi + a2 * yi;
        out[i] = yi;
    }
}

__global__ void deriche_row_forward_f(const float* x, float* yp, int width, int height, float a0, float a1, float b1, float b2, int border_mode, float border_const) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const float* in = x + size_t(row) * size_t(width);
    float* out = yp + size_t(row) * size_t(width);
    float xm1 = border_sample_f(in, width, -1, border_mode, border_const);
    float xm2 = border_sample_f(in, width, -2, border_mode, border_const);
    float xm3 = border_sample_f(in, width, -3, border_mode, border_const);
    float ym2 = a0 * xm2 + a1 * xm3;
    float ym1 = a0 * xm1 + a1 * xm2 + b1 * ym2;
    for (int i = 0; i < width; ++i) {
        float xi = in[i];
        float yi = a0 * xi + a1 * xm1 + b1 * ym1 + b2 * ym2;
        out[i] = yi;
        xm1 = xi; ym2 = ym1; ym1 = yi;
    }
}

__global__ void deriche_row_backward_f(const float* x, float* yn, int width, int height, float a2, float a3, float b1, float b2, int border_mode, float border_const) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const float* in = x + size_t(row) * size_t(width);
    float* out = yn + size_t(row) * size_t(width);
    float xp1 = border_sample_f(in, width, width, border_mode, border_const);
    float xp2 = border_sample_f(in, width, width + 1, border_mode, border_const);
    float xp3 = border_sample_f(in, width, width + 2, border_mode, border_const);
    float yn2 = a2 * xp2 + a3 * xp3;
    float yn1 = a2 * xp1 + a3 * xp2 + b1 * yn2;
    for (int i = width - 1; i >= 0; --i) {
        float xi = in[i];
        float yi = a2 * xp1 + a3 * xp2 + b1 * yn1 + b2 * yn2;
        out[i] = yi;
        xp2 = xp1; xp1 = xi; yn2 = yn1; yn1 = yi;
    }
}

__global__ void combine_add_f(const float* a, const float* b, float* out, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = scale * (a[i] + b[i]);
}

// ---- kernels (mixed: f32 IO, f64 internal) ----
__global__ void iir_row_first_mixed(const float* x, float* y, int width, int height, double b0, double b1, double a1, int border_mode, float border_const) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const float* in = x + size_t(row) * size_t(width);
    float* out = y + size_t(row) * size_t(width);
    double xm1 = (double)border_sample_f(in, width, -1, border_mode, border_const);
    double xm2 = (double)border_sample_f(in, width, -2, border_mode, border_const);
    double yprev = b0 * xm1 + b1 * xm2;
    double xprev = xm1;
    for (int i = 0; i < width; ++i) {
        double xi = in[i];
        double yi = b0 * xi + b1 * xprev + a1 * yprev;
        out[i] = (float)yi;
        xprev = xi;
        yprev = yi;
    }
}

__global__ void iir_row_biquad_mixed(const float* x, float* y, int width, int height, double b0, double b1, double b2, double a1, double a2, int border_mode, float border_const) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const float* in = x + size_t(row) * size_t(width);
    float* out = y + size_t(row) * size_t(width);
    double xm1 = (double)border_sample_f(in, width, -1, border_mode, border_const);
    double xm2 = (double)border_sample_f(in, width, -2, border_mode, border_const);
    double xm3 = (double)border_sample_f(in, width, -3, border_mode, border_const);
    double xm4 = (double)border_sample_f(in, width, -4, border_mode, border_const);
    double y2 = b0 * xm2 + b1 * xm3 + b2 * xm4;
    double y1 = b0 * xm1 + b1 * xm2 + b2 * xm3 + a1 * y2;
    double x1 = xm1, x2 = xm2;
    for (int i = 0; i < width; ++i) {
        double xi = in[i];
        double yi = b0 * xi + b1 * x1 + b2 * x2 + a1 * y1 + a2 * y2;
        out[i] = (float)yi;
        x2 = x1; x1 = xi;
        y2 = y1; y1 = yi;
    }
}

__global__ void iir_row_fwd_bwd_mixed(const float* x, float* y, int width, int height, double b0, double b1, double a1, int border_mode, float border_const) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const float* in = x + size_t(row) * size_t(width);
    float* out = y + size_t(row) * size_t(width);
    double xm1 = (double)border_sample_f(in, width, -1, border_mode, border_const);
    double xm2 = (double)border_sample_f(in, width, -2, border_mode, border_const);
    double yprev = b0 * xm1 + b1 * xm2;
    double xprev = xm1;
    for (int i = 0; i < width; ++i) {
        double xi = in[i];
        double yi = b0 * xi + b1 * xprev + a1 * yprev;
        out[i] = (float)yi;
        xprev = xi;
        yprev = yi;
    }
    double xp1 = (double)border_sample_f(out, width, width, border_mode, border_const);
    double xp2 = (double)border_sample_f(out, width, width + 1, border_mode, border_const);
    yprev = b0 * xp1 + b1 * xp2;
    xprev = xp1;
    for (int i = width - 1; i >= 0; --i) {
        double xi = out[i];
        double yi = b0 * xi + b1 * xprev + a1 * yprev;
        out[i] = (float)yi;
        xprev = xi;
        yprev = yi;
    }
}

__global__ void iir_row_statespace_mixed(const float* x, float* y, int width, int height, double b0, double b1, double b2, double a1, double a2, int border_mode, float border_const) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const float* in = x + size_t(row) * size_t(width);
    float* out = y + size_t(row) * size_t(width);
    double xm2 = (double)border_sample_f(in, width, -2, border_mode, border_const);
    double xm1 = (double)border_sample_f(in, width, -1, border_mode, border_const);
    double z1 = 0.0, z2 = 0.0;
    for (int k = 0; k < 2; ++k) {
        double xi = (k == 0) ? xm2 : xm1;
        double yi = b0 * xi + z1;
        z1 = b1 * xi + z2 + a1 * yi;
        z2 = b2 * xi + a2 * yi;
    }
    for (int i = 0; i < width; ++i) {
        double xi = in[i];
        double yi = b0 * xi + z1;
        z1 = b1 * xi + z2 + a1 * yi;
        z2 = b2 * xi + a2 * yi;
        out[i] = (float)yi;
    }
}

__global__ void deriche_row_forward_mixed(const float* x, float* yp, int width, int height, double a0, double a1, double b1, double b2, int border_mode, float border_const) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const float* in = x + size_t(row) * size_t(width);
    float* out = yp + size_t(row) * size_t(width);
    double xm1 = (double)border_sample_f(in, width, -1, border_mode, border_const);
    double xm2 = (double)border_sample_f(in, width, -2, border_mode, border_const);
    double xm3 = (double)border_sample_f(in, width, -3, border_mode, border_const);
    double ym2 = a0 * xm2 + a1 * xm3;
    double ym1 = a0 * xm1 + a1 * xm2 + b1 * ym2;
    for (int i = 0; i < width; ++i) {
        double xi = in[i];
        double yi = a0 * xi + a1 * xm1 + b1 * ym1 + b2 * ym2;
        out[i] = (float)yi;
        xm1 = xi; ym2 = ym1; ym1 = yi;
    }
}

__global__ void deriche_row_backward_mixed(const float* x, float* yn, int width, int height, double a2, double a3, double b1, double b2, int border_mode, float border_const) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const float* in = x + size_t(row) * size_t(width);
    float* out = yn + size_t(row) * size_t(width);
    double xp1 = (double)border_sample_f(in, width, width, border_mode, border_const);
    double xp2 = (double)border_sample_f(in, width, width + 1, border_mode, border_const);
    double xp3 = (double)border_sample_f(in, width, width + 2, border_mode, border_const);
    double yn2 = a2 * xp2 + a3 * xp3;
    double yn1 = a2 * xp1 + a3 * xp2 + b1 * yn2;
    for (int i = width - 1; i >= 0; --i) {
        double xi = in[i];
        double yi = a2 * xp1 + a3 * xp2 + b1 * yn1 + b2 * yn2;
        out[i] = (float)yi;
        xp2 = xp1; xp1 = xi; yn2 = yn1; yn1 = yi;
    }
}

// ---- kernels (f64) ----
__global__ void iir_row_first_d(const double* x, double* y, int width, int height, double b0, double b1, double a1, int border_mode, double border_const) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const double* in = x + size_t(row) * size_t(width);
    double* out = y + size_t(row) * size_t(width);
    double xm1 = border_sample_d(in, width, -1, border_mode, border_const);
    double xm2 = border_sample_d(in, width, -2, border_mode, border_const);
    double yprev = b0 * xm1 + b1 * xm2;
    double xprev = xm1;
    for (int i = 0; i < width; ++i) {
        double xi = in[i];
        double yi = b0 * xi + b1 * xprev + a1 * yprev;
        out[i] = yi;
        xprev = xi;
        yprev = yi;
    }
}

__global__ void iir_row_biquad_d(const double* x, double* y, int width, int height, double b0, double b1, double b2, double a1, double a2, int border_mode, double border_const) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const double* in = x + size_t(row) * size_t(width);
    double* out = y + size_t(row) * size_t(width);
    double xm1 = border_sample_d(in, width, -1, border_mode, border_const);
    double xm2 = border_sample_d(in, width, -2, border_mode, border_const);
    double xm3 = border_sample_d(in, width, -3, border_mode, border_const);
    double xm4 = border_sample_d(in, width, -4, border_mode, border_const);
    double y2 = b0 * xm2 + b1 * xm3 + b2 * xm4;
    double y1 = b0 * xm1 + b1 * xm2 + b2 * xm3 + a1 * y2;
    double x1 = xm1, x2 = xm2;
    for (int i = 0; i < width; ++i) {
        double xi = in[i];
        double yi = b0 * xi + b1 * x1 + b2 * x2 + a1 * y1 + a2 * y2;
        out[i] = yi;
        x2 = x1; x1 = xi;
        y2 = y1; y1 = yi;
    }
}

__global__ void iir_row_fwd_bwd_d(const double* x, double* y, int width, int height, double b0, double b1, double a1, int border_mode, double border_const) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const double* in = x + size_t(row) * size_t(width);
    double* out = y + size_t(row) * size_t(width);
    double xm1 = border_sample_d(in, width, -1, border_mode, border_const);
    double xm2 = border_sample_d(in, width, -2, border_mode, border_const);
    double yprev = b0 * xm1 + b1 * xm2;
    double xprev = xm1;
    for (int i = 0; i < width; ++i) {
        double xi = in[i];
        double yi = b0 * xi + b1 * xprev + a1 * yprev;
        out[i] = yi;
        xprev = xi;
        yprev = yi;
    }
    double xp1 = border_sample_d(out, width, width, border_mode, border_const);
    double xp2 = border_sample_d(out, width, width + 1, border_mode, border_const);
    yprev = b0 * xp1 + b1 * xp2;
    xprev = xp1;
    for (int i = width - 1; i >= 0; --i) {
        double xi = out[i];
        double yi = b0 * xi + b1 * xprev + a1 * yprev;
        out[i] = yi;
        xprev = xi;
        yprev = yi;
    }
}

__global__ void iir_row_statespace_d(const double* x, double* y, int width, int height, double b0, double b1, double b2, double a1, double a2, int border_mode, double border_const) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const double* in = x + size_t(row) * size_t(width);
    double* out = y + size_t(row) * size_t(width);
    double xm2 = border_sample_d(in, width, -2, border_mode, border_const);
    double xm1 = border_sample_d(in, width, -1, border_mode, border_const);
    double z1 = 0.0, z2 = 0.0;
    for (int k = 0; k < 2; ++k) {
        double xi = (k == 0) ? xm2 : xm1;
        double yi = b0 * xi + z1;
        z1 = b1 * xi + z2 + a1 * yi;
        z2 = b2 * xi + a2 * yi;
    }
    for (int i = 0; i < width; ++i) {
        double xi = in[i];
        double yi = b0 * xi + z1;
        z1 = b1 * xi + z2 + a1 * yi;
        z2 = b2 * xi + a2 * yi;
        out[i] = yi;
    }
}

__global__ void deriche_row_forward_d(const double* x, double* yp, int width, int height, double a0, double a1, double b1, double b2, int border_mode, double border_const) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const double* in = x + size_t(row) * size_t(width);
    double* out = yp + size_t(row) * size_t(width);
    double xm1 = border_sample_d(in, width, -1, border_mode, border_const);
    double xm2 = border_sample_d(in, width, -2, border_mode, border_const);
    double xm3 = border_sample_d(in, width, -3, border_mode, border_const);
    double ym2 = a0 * xm2 + a1 * xm3;
    double ym1 = a0 * xm1 + a1 * xm2 + b1 * ym2;
    for (int i = 0; i < width; ++i) {
        double xi = in[i];
        double yi = a0 * xi + a1 * xm1 + b1 * ym1 + b2 * ym2;
        out[i] = yi;
        xm1 = xi; ym2 = ym1; ym1 = yi;
    }
}

__global__ void deriche_row_backward_d(const double* x, double* yn, int width, int height, double a2, double a3, double b1, double b2, int border_mode, double border_const) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const double* in = x + size_t(row) * size_t(width);
    double* out = yn + size_t(row) * size_t(width);
    double xp1 = border_sample_d(in, width, width, border_mode, border_const);
    double xp2 = border_sample_d(in, width, width + 1, border_mode, border_const);
    double xp3 = border_sample_d(in, width, width + 2, border_mode, border_const);
    double yn2 = a2 * xp2 + a3 * xp3;
    double yn1 = a2 * xp1 + a3 * xp2 + b1 * yn2;
    for (int i = width - 1; i >= 0; --i) {
        double xi = in[i];
        double yi = a2 * xp1 + a3 * xp2 + b1 * yn1 + b2 * yn2;
        out[i] = yi;
        xp2 = xp1; xp1 = xi; yn2 = yn1; yn1 = yi;
    }
}

__global__ void combine_add_d(const double* a, const double* b, double* out, int n, double scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = scale * (a[i] + b[i]);
}

// ---- transpose ----
__global__ void transpose32_f(const float* in, float* out, int width, int height) {
    __shared__ float tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (x < width && y < height) tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    __syncthreads();
    int ox = blockIdx.y * 32 + threadIdx.x;
    int oy = blockIdx.x * 32 + threadIdx.y;
    if (ox < height && oy < width) out[oy * height + ox] = tile[threadIdx.x][threadIdx.y];
}

__global__ void transpose32_d(const double* in, double* out, int width, int height) {
    __shared__ double tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (x < width && y < height) tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    __syncthreads();
    int ox = blockIdx.y * 32 + threadIdx.x;
    int oy = blockIdx.x * 32 + threadIdx.y;
    if (ox < height && oy < width) out[oy * height + ox] = tile[threadIdx.x][threadIdx.y];
}

static inline int iir_rows_scan_first(
    const float* d_x,
    float* d_y,
    int width,
    int height,
    float b0, float b1, float a1,
    float* d_block_a,
    float* d_block_p,
    float* d_block_in,
    float* d_row_yprev,
    int border_mode,
    float border_const,
    cudaStream_t stream
) {
    constexpr int BLOCK = 256;
    dim3 block(BLOCK);
    dim3 grid((width + BLOCK - 1) / BLOCK, height);
    int threads = 128;
    int blocks = (height + threads - 1) / threads;
    row_init_first_f<<<blocks, threads, 0, stream>>>(d_x, width, height, b0, b1, a1, border_mode, border_const, d_row_yprev);
    CUDA_CHECK_RET(cudaGetLastError());
    iir_row_local_scan<BLOCK><<<grid, block, 0, stream>>>(d_x, d_y, width, height, b0, b1, a1, border_mode, border_const, d_block_a, d_block_p);
    CUDA_CHECK_RET(cudaGetLastError());
    int num_blocks = grid.x;
    iir_row_block_prefix_scan<<<blocks, threads, 0, stream>>>(d_block_a, d_block_p, d_block_in, num_blocks, height, d_row_yprev);
    CUDA_CHECK_RET(cudaGetLastError());
    iir_row_fixup_scan<BLOCK><<<grid, block, 0, stream>>>(d_y, width, height, a1, d_block_in);
    CUDA_CHECK_RET(cudaGetLastError());
    return IIR2D_STATUS_OK;
}

static inline int iir_rows_biquad_scan(
    const float* d_x,
    float* d_y,
    int width,
    int height,
    float b0, float b1, float b2,
    float a1, float a2,
    Transform4* d_block_t,
    float4* d_block_in,
    float4* d_row_state,
    int border_mode,
    float border_const,
    cudaStream_t stream
) {
    constexpr int BLOCK = 256;
    dim3 block(BLOCK);
    dim3 grid((width + BLOCK - 1) / BLOCK, height);
    int threads = 128;
    int blocks = (height + threads - 1) / threads;
    row_init_biquad_f<<<blocks, threads, 0, stream>>>(d_x, width, height, b0, b1, b2, a1, a2, border_mode, border_const, d_row_state);
    CUDA_CHECK_RET(cudaGetLastError());
    biquad_block_transform4<BLOCK><<<grid, block, 0, stream>>>(d_x, width, height, b0, b1, b2, a1, a2, d_block_t);
    CUDA_CHECK_RET(cudaGetLastError());
    int num_blocks = grid.x;
    biquad_block_prefix4<<<blocks, threads, 0, stream>>>(d_block_t, d_block_in, num_blocks, height, d_row_state);
    CUDA_CHECK_RET(cudaGetLastError());
    biquad_block_fixup4<BLOCK><<<grid, block, 0, stream>>>(d_x, d_y, width, height, b0, b1, b2, a1, a2, d_block_in);
    CUDA_CHECK_RET(cudaGetLastError());
    return IIR2D_STATUS_OK;
}

static inline int iir_rows_scan_first_d(
    const double* d_x,
    double* d_y,
    int width,
    int height,
    double b0, double b1, double a1,
    double* d_block_a,
    double* d_block_p,
    double* d_block_in,
    double* d_row_yprev,
    int border_mode,
    double border_const,
    cudaStream_t stream
) {
    constexpr int BLOCK = 256;
    dim3 block(BLOCK);
    dim3 grid((width + BLOCK - 1) / BLOCK, height);
    int threads = 128;
    int blocks = (height + threads - 1) / threads;
    // init yprev with border
    row_init_first_d<<<blocks, threads, 0, stream>>>(d_x, width, height, b0, b1, a1, border_mode, border_const, d_row_yprev);
    CUDA_CHECK_RET(cudaGetLastError());
    iir_row_local_scan_d<BLOCK><<<grid, block, 0, stream>>>(d_x, d_y, width, height, b0, b1, a1, border_mode, border_const, d_block_a, d_block_p);
    CUDA_CHECK_RET(cudaGetLastError());
    int num_blocks = grid.x;
    iir_row_block_prefix_scan_d<<<blocks, threads, 0, stream>>>(d_block_a, d_block_p, d_block_in, num_blocks, height, d_row_yprev);
    CUDA_CHECK_RET(cudaGetLastError());
    iir_row_fixup_scan_d<BLOCK><<<grid, block, 0, stream>>>(d_y, width, height, a1, d_block_in);
    CUDA_CHECK_RET(cudaGetLastError());
    return IIR2D_STATUS_OK;
}

static inline int iir_rows_biquad_scan_d(
    const double* d_x,
    double* d_y,
    int width,
    int height,
    double b0, double b1, double b2,
    double a1, double a2,
    Transform4D* d_block_t,
    double4* d_block_in,
    double4* d_row_state,
    int border_mode,
    double border_const,
    cudaStream_t stream
) {
    constexpr int BLOCK = 256;
    dim3 block(BLOCK);
    dim3 grid((width + BLOCK - 1) / BLOCK, height);
    int threads = 128;
    int blocks = (height + threads - 1) / threads;
    // init row state
    row_init_biquad_d<<<blocks, threads, 0, stream>>>(d_x, width, height, b0, b1, b2, a1, a2, border_mode, border_const, d_row_state);
    CUDA_CHECK_RET(cudaGetLastError());
    biquad_block_transform4_d<BLOCK><<<grid, block, 0, stream>>>(d_x, width, height, b0, b1, b2, a1, a2, d_block_t);
    CUDA_CHECK_RET(cudaGetLastError());
    int num_blocks = grid.x;
    biquad_block_prefix4_d<<<blocks, threads, 0, stream>>>(d_block_t, d_block_in, num_blocks, height, d_row_state);
    CUDA_CHECK_RET(cudaGetLastError());
    biquad_block_fixup4_d<BLOCK><<<grid, block, 0, stream>>>(d_x, d_y, width, height, b0, b1, b2, a1, a2, d_block_in);
    CUDA_CHECK_RET(cudaGetLastError());
    return 0;
}

// ---- helpers ----
static inline int launch_rows_f(const float* in, float* out, int w, int h, int filter_id, int border_mode, float border_const, cudaStream_t stream, float* tmp1, float* tmp2) {
    // coefficients (image-friendly defaults)
    float b0 = 0.7f, b1 = 0.2f, a1 = 0.9f;
    float b2 = 0.0f, a2 = 0.0f;
    float fb_b0 = 0.4f, fb_b1 = 0.0f, fb_a1 = 0.6f;
    float ss_b0 = 0.2f, ss_b1 = 0.2f, ss_b2 = 0.2f, ss_a1 = 0.3f, ss_a2 = -0.1f;
    float der_sigma = 2.0f;
    // Deriche coeffs
    float alpha = 1.695f / der_sigma;
    float ema = expf(-alpha);
    float ema2 = expf(-2.0f * alpha);
    float k = (1.0f - ema) * (1.0f - ema) / (1.0f + 2.0f * alpha * ema - ema2);
    float der_a0 = k;
    float der_a1 = k * (alpha - 1.0f) * ema;
    float der_a2 = k * (alpha + 1.0f) * ema;
    float der_a3 = -k * ema2;
    float der_b1 = 2.0f * ema;
    float der_b2 = -ema2;
    float der_c1 = 1.0f;

    if (filter_id == 7) {
        float alpha2 = 0.85f;
        b0 = 1.0f - alpha2;
        b1 = 0.0f;
        a1 = alpha2;
    } else if (filter_id == 1) {
        float alpha2 = 0.85f;
        b0 = 1.0f - alpha2;
        b1 = 0.0f;
        a1 = alpha2;
    }

    int threads = 128;
    int blocks = (h + threads - 1) / threads;
    if (filter_id == 1 || filter_id == 7) {
        iir_row_first_f<<<blocks, threads, 0, stream>>>(in, out, w, h, b0, b1, a1, border_mode, border_const);
    } else if (filter_id == 2) {
        float cas_a = 0.75f, cas_b = 0.25f;
        float* tmp = tmp1;
        iir_row_first_f<<<blocks, threads, 0, stream>>>(in, tmp, w, h, cas_b, 0.0f, cas_a, border_mode, border_const);
        iir_row_first_f<<<blocks, threads, 0, stream>>>(tmp, out, w, h, cas_b, 0.0f, cas_a, border_mode, border_const);
    } else if (filter_id == 3) {
        iir_row_biquad_f<<<blocks, threads, 0, stream>>>(in, out, w, h, 0.2f, 0.2f, 0.2f, 0.3f, -0.1f, border_mode, border_const);
    } else if (filter_id == 4) {
        // SOS as cascade of two biquads
        float* tmp = tmp1;
        iir_row_biquad_f<<<blocks, threads, 0, stream>>>(in, tmp, w, h, 0.2f, 0.2f, 0.2f, 0.3f, -0.1f, border_mode, border_const);
        iir_row_biquad_f<<<blocks, threads, 0, stream>>>(tmp, out, w, h, 0.3f, 0.1f, 0.1f, 0.2f, -0.05f, border_mode, border_const);
    } else if (filter_id == 5) {
        iir_row_fwd_bwd_f<<<blocks, threads, 0, stream>>>(in, out, w, h, fb_b0, fb_b1, fb_a1, border_mode, border_const);
    } else if (filter_id == 6) {
        float* t1 = tmp1;
        float* t2 = tmp2;
        deriche_row_forward_f<<<blocks, threads, 0, stream>>>(in, t1, w, h, der_a0, der_a1, der_b1, der_b2, border_mode, border_const);
        deriche_row_backward_f<<<blocks, threads, 0, stream>>>(in, t2, w, h, der_a2, der_a3, der_b1, der_b2, border_mode, border_const);
        int n = w * h;
        int tpb = 256;
        int b2 = (n + tpb - 1) / tpb;
        combine_add_f<<<b2, tpb, 0, stream>>>(t1, t2, out, n, der_c1);
    } else if (filter_id == 8) {
        iir_row_statespace_f<<<blocks, threads, 0, stream>>>(in, out, w, h, ss_b0, ss_b1, ss_b2, ss_a1, ss_a2, border_mode, border_const);
    }
    CUDA_CHECK_RET(cudaGetLastError());
    return 0;
}

static inline int launch_rows_mixed(const float* in, float* out, int w, int h, int filter_id, int border_mode, float border_const, cudaStream_t stream, float* tmp1, float* tmp2) {
    double b0 = 0.7, b1 = 0.2, a1 = 0.9;
    double fb_b0 = 0.4, fb_b1 = 0.0, fb_a1 = 0.6;
    double ss_b0 = 0.2, ss_b1 = 0.2, ss_b2 = 0.2, ss_a1 = 0.3, ss_a2 = -0.1;
    double der_sigma = 2.0;
    double alpha = 1.695 / der_sigma;
    double ema = exp(-alpha);
    double ema2 = exp(-2.0 * alpha);
    double k = (1.0 - ema) * (1.0 - ema) / (1.0 + 2.0 * alpha * ema - ema2);
    double der_a0 = k;
    double der_a1 = k * (alpha - 1.0) * ema;
    double der_a2 = k * (alpha + 1.0) * ema;
    double der_a3 = -k * ema2;
    double der_b1 = 2.0 * ema;
    double der_b2 = -ema2;
    double der_c1 = 1.0;

    if (filter_id == 7 || filter_id == 1) {
        double alpha2 = 0.85;
        b0 = 1.0 - alpha2;
        b1 = 0.0;
        a1 = alpha2;
    }

    int threads = 128;
    int blocks = (h + threads - 1) / threads;
    if (filter_id == 1 || filter_id == 7) {
        iir_row_first_mixed<<<blocks, threads, 0, stream>>>(in, out, w, h, b0, b1, a1, border_mode, border_const);
    } else if (filter_id == 2) {
        double cas_a = 0.75, cas_b = 0.25;
        float* tmp = tmp1;
        iir_row_first_mixed<<<blocks, threads, 0, stream>>>(in, tmp, w, h, cas_b, 0.0, cas_a, border_mode, border_const);
        iir_row_first_mixed<<<blocks, threads, 0, stream>>>(tmp, out, w, h, cas_b, 0.0, cas_a, border_mode, border_const);
    } else if (filter_id == 3) {
        iir_row_biquad_mixed<<<blocks, threads, 0, stream>>>(in, out, w, h, 0.2, 0.2, 0.2, 0.3, -0.1, border_mode, border_const);
    } else if (filter_id == 4) {
        float* tmp = tmp1;
        iir_row_biquad_mixed<<<blocks, threads, 0, stream>>>(in, tmp, w, h, 0.2, 0.2, 0.2, 0.3, -0.1, border_mode, border_const);
        iir_row_biquad_mixed<<<blocks, threads, 0, stream>>>(tmp, out, w, h, 0.3, 0.1, 0.1, 0.2, -0.05, border_mode, border_const);
    } else if (filter_id == 5) {
        iir_row_fwd_bwd_mixed<<<blocks, threads, 0, stream>>>(in, out, w, h, fb_b0, fb_b1, fb_a1, border_mode, border_const);
    } else if (filter_id == 6) {
        float* t1 = tmp1;
        float* t2 = tmp2;
        deriche_row_forward_mixed<<<blocks, threads, 0, stream>>>(in, t1, w, h, der_a0, der_a1, der_b1, der_b2, border_mode, border_const);
        deriche_row_backward_mixed<<<blocks, threads, 0, stream>>>(in, t2, w, h, der_a2, der_a3, der_b1, der_b2, border_mode, border_const);
        int n = w * h;
        int tpb = 256;
        int b2 = (n + tpb - 1) / tpb;
        combine_add_f<<<b2, tpb, 0, stream>>>(t1, t2, out, n, (float)der_c1);
    } else if (filter_id == 8) {
        iir_row_statespace_mixed<<<blocks, threads, 0, stream>>>(in, out, w, h, ss_b0, ss_b1, ss_b2, ss_a1, ss_a2, border_mode, border_const);
    }
    CUDA_CHECK_RET(cudaGetLastError());
    return 0;
}

static inline int launch_rows_d(const double* in, double* out, int w, int h, int filter_id, int border_mode, double border_const, cudaStream_t stream, double* tmp1, double* tmp2) {
    double b0 = 0.7, b1 = 0.2, a1 = 0.9;
    double fb_b0 = 0.4, fb_b1 = 0.0, fb_a1 = 0.6;
    double ss_b0 = 0.2, ss_b1 = 0.2, ss_b2 = 0.2, ss_a1 = 0.3, ss_a2 = -0.1;
    double der_sigma = 2.0;
    double alpha = 1.695 / der_sigma;
    double ema = exp(-alpha);
    double ema2 = exp(-2.0 * alpha);
    double k = (1.0 - ema) * (1.0 - ema) / (1.0 + 2.0 * alpha * ema - ema2);
    double der_a0 = k;
    double der_a1 = k * (alpha - 1.0) * ema;
    double der_a2 = k * (alpha + 1.0) * ema;
    double der_a3 = -k * ema2;
    double der_b1 = 2.0 * ema;
    double der_b2 = -ema2;
    double der_c1 = 1.0;

    if (filter_id == 7 || filter_id == 1) {
        double alpha2 = 0.85;
        b0 = 1.0 - alpha2;
        b1 = 0.0;
        a1 = alpha2;
    }

    int threads = 128;
    int blocks = (h + threads - 1) / threads;
    if (filter_id == 1 || filter_id == 7) {
        iir_row_first_d<<<blocks, threads, 0, stream>>>(in, out, w, h, b0, b1, a1, border_mode, border_const);
    } else if (filter_id == 2) {
        double cas_a = 0.75, cas_b = 0.25;
        double* tmp = tmp1;
        iir_row_first_d<<<blocks, threads, 0, stream>>>(in, tmp, w, h, cas_b, 0.0, cas_a, border_mode, border_const);
        iir_row_first_d<<<blocks, threads, 0, stream>>>(tmp, out, w, h, cas_b, 0.0, cas_a, border_mode, border_const);
    } else if (filter_id == 3) {
        iir_row_biquad_d<<<blocks, threads, 0, stream>>>(in, out, w, h, 0.2, 0.2, 0.2, 0.3, -0.1, border_mode, border_const);
    } else if (filter_id == 4) {
        double* tmp = tmp1;
        iir_row_biquad_d<<<blocks, threads, 0, stream>>>(in, tmp, w, h, 0.2, 0.2, 0.2, 0.3, -0.1, border_mode, border_const);
        iir_row_biquad_d<<<blocks, threads, 0, stream>>>(tmp, out, w, h, 0.3, 0.1, 0.1, 0.2, -0.05, border_mode, border_const);
    } else if (filter_id == 5) {
        iir_row_fwd_bwd_d<<<blocks, threads, 0, stream>>>(in, out, w, h, fb_b0, fb_b1, fb_a1, border_mode, border_const);
    } else if (filter_id == 6) {
        double* t1 = tmp1;
        double* t2 = tmp2;
        deriche_row_forward_d<<<blocks, threads, 0, stream>>>(in, t1, w, h, der_a0, der_a1, der_b1, der_b2, border_mode, border_const);
        deriche_row_backward_d<<<blocks, threads, 0, stream>>>(in, t2, w, h, der_a2, der_a3, der_b1, der_b2, border_mode, border_const);
        int n = w * h;
        int tpb = 256;
        int b2 = (n + tpb - 1) / tpb;
        combine_add_d<<<b2, tpb, 0, stream>>>(t1, t2, out, n, der_c1);
    } else if (filter_id == 8) {
        iir_row_statespace_d<<<blocks, threads, 0, stream>>>(in, out, w, h, ss_b0, ss_b1, ss_b2, ss_a1, ss_a2, border_mode, border_const);
    }
    CUDA_CHECK_RET(cudaGetLastError());
    return 0;
}

static inline bool is_valid_filter_id(int filter_id) {
    return filter_id >= 1 && filter_id <= 8;
}

static inline bool is_valid_border_mode(int border_mode) {
    return border_mode == IIR2D_BORDER_CLAMP ||
           border_mode == IIR2D_BORDER_MIRROR ||
           border_mode == IIR2D_BORDER_WRAP ||
           border_mode == IIR2D_BORDER_CONSTANT;
}

static inline bool is_valid_precision(int precision) {
    return precision == IIR2D_PREC_F32 ||
           precision == IIR2D_PREC_MIXED ||
           precision == IIR2D_PREC_F64;
}

static inline int validate_params(const void* in, void* out, const IIR2D_Params* params) {
    if (!params) return IIR2D_STATUS_INVALID_ARGUMENT;
    if (!in || !out) return IIR2D_STATUS_NULL_POINTER;
    if (params->width <= 0 || params->height <= 0) return IIR2D_STATUS_INVALID_DIMENSION;
    if (!is_valid_filter_id(params->filter_id)) return IIR2D_STATUS_INVALID_FILTER_ID;
    if (!is_valid_border_mode(params->border_mode)) return IIR2D_STATUS_INVALID_BORDER_MODE;
    if (!is_valid_precision(params->precision)) return IIR2D_STATUS_INVALID_PRECISION;
    return IIR2D_STATUS_OK;
}

// ---- public API ----
int iir2d_forward_cuda_stream(const void* in, void* out, const IIR2D_Params* params, void* stream_ptr) {
    const int status = validate_params(in, out, params);
    if (status != IIR2D_STATUS_OK) return status;
    int w = params->width;
    int h = params->height;
    int filter_id = params->filter_id;
    int border_mode = params->border_mode;
    float border_const = params->border_const;
    int precision = params->precision;

    dim3 tblock(32, 32);
    dim3 tgrid((w + 31) / 32, (h + 31) / 32);
    dim3 tgrid2((h + 31) / 32, (w + 31) / 32);

    cudaStream_t stream = stream_ptr ? (cudaStream_t)stream_ptr : 0;

    if (precision == IIR2D_PREC_F64) {
        const double* din = (const double*)in;
        double* dout = (double*)out;
        const bool use_scan = (filter_id == 1 || filter_id == 2 || filter_id == 3 || filter_id == 4 || filter_id == 8);
        if (use_scan) {
            const size_t n = size_t(w) * size_t(h);
            const int BLOCK = 256;
            int num_blocks_row = (w + BLOCK - 1) / BLOCK;
            int num_blocks_col = (h + BLOCK - 1) / BLOCK;
            size_t max_blocks = (size_t)h * (size_t)num_blocks_row;
            size_t alt_blocks = (size_t)w * (size_t)num_blocks_col;
            if (alt_blocks > max_blocks) max_blocks = alt_blocks;

            size_t max_wh = (size_t)(w > h ? w : h);
            size_t bytes_tmp = n * sizeof(double);
            size_t bytes_block_d = max_blocks * sizeof(double);
            size_t bytes_block_t = max_blocks * sizeof(Transform4D);
            size_t bytes_block_in4 = max_blocks * sizeof(double4);
            size_t bytes_row = max_wh * sizeof(double);
            size_t bytes_row4 = max_wh * sizeof(double4);
            size_t total = 0;
            total += align_up(bytes_tmp, 256);
            total += align_up(bytes_tmp, 256);
            total += align_up(bytes_block_d, 256) * 3;
            total += align_up(bytes_block_t, 256);
            total += align_up(bytes_block_in4, 256);
            total += align_up(bytes_row, 256);
            total += align_up(bytes_row4, 256);

            void* base = nullptr;
            CUDA_CHECK_RET(get_workspace_bytes(total, &base));
            char* ptr = (char*)base;

            double* tmp = (double*)ptr; ptr += align_up(bytes_tmp, 256);
            double* tmp2 = (double*)ptr; ptr += align_up(bytes_tmp, 256);
            double* block_a = (double*)ptr; ptr += align_up(bytes_block_d, 256);
            double* block_p = (double*)ptr; ptr += align_up(bytes_block_d, 256);
            double* block_in = (double*)ptr; ptr += align_up(bytes_block_d, 256);
            Transform4D* block_t = (Transform4D*)ptr; ptr += align_up(bytes_block_t, 256);
            double4* block_in4 = (double4*)ptr; ptr += align_up(bytes_block_in4, 256);
            double* row_yprev = (double*)ptr; ptr += align_up(bytes_row, 256);
            double4* row_state = (double4*)ptr; ptr += align_up(bytes_row4, 256);

            if (filter_id == 1 || filter_id == 7) {
                double b0 = 1.0 - 0.85;
                double a1 = 0.85;
                CUDA_CHECK_RET(iir_rows_scan_first_d(din, dout, w, h, b0, 0.0, a1, block_a, block_p, block_in, row_yprev, border_mode, (double)border_const, stream));
            } else if (filter_id == 2) {
                double a = 0.75, b = 0.25;
                CUDA_CHECK_RET(iir_rows_scan_first_d(din, dout, w, h, b, 0.0, a, block_a, block_p, block_in, row_yprev, border_mode, (double)border_const, stream));
                CUDA_CHECK_RET(iir_rows_scan_first_d(dout, tmp, w, h, b, 0.0, a, block_a, block_p, block_in, row_yprev, border_mode, (double)border_const, stream));
                CUDA_CHECK_RET(cudaMemcpyAsync(dout, tmp, bytes_tmp, cudaMemcpyDeviceToDevice, stream));
            } else if (filter_id == 3) {
                CUDA_CHECK_RET(iir_rows_biquad_scan_d(din, dout, w, h, 0.2, 0.2, 0.2, 0.3, -0.1, block_t, block_in4, row_state, border_mode, (double)border_const, stream));
            } else if (filter_id == 4) {
                CUDA_CHECK_RET(iir_rows_biquad_scan_d(din, tmp, w, h, 0.2, 0.2, 0.2, 0.3, -0.1, block_t, block_in4, row_state, border_mode, (double)border_const, stream));
                CUDA_CHECK_RET(iir_rows_biquad_scan_d(tmp, dout, w, h, 0.3, 0.1, 0.1, 0.2, -0.05, block_t, block_in4, row_state, border_mode, (double)border_const, stream));
            } else if (filter_id == 8) {
                CUDA_CHECK_RET(iir_rows_biquad_scan_d(din, dout, w, h, 0.2, 0.2, 0.2, 0.3, -0.1, block_t, block_in4, row_state, border_mode, (double)border_const, stream));
            } else {
                CUDA_CHECK_RET(launch_rows_d(din, dout, w, h, filter_id, border_mode, (double)border_const, stream, tmp, tmp2));
            }

            transpose32_d<<<tgrid, tblock, 0, stream>>>(dout, tmp, w, h);
            CUDA_CHECK_RET(cudaGetLastError());

            if (filter_id == 1 || filter_id == 7) {
                double b0 = 1.0 - 0.85;
                double a1 = 0.85;
                CUDA_CHECK_RET(iir_rows_scan_first_d(tmp, tmp2, h, w, b0, 0.0, a1, block_a, block_p, block_in, row_yprev, border_mode, (double)border_const, stream));
            } else if (filter_id == 2) {
                double a = 0.75, b = 0.25;
                CUDA_CHECK_RET(iir_rows_scan_first_d(tmp, tmp2, h, w, b, 0.0, a, block_a, block_p, block_in, row_yprev, border_mode, (double)border_const, stream));
                CUDA_CHECK_RET(iir_rows_scan_first_d(tmp2, tmp, h, w, b, 0.0, a, block_a, block_p, block_in, row_yprev, border_mode, (double)border_const, stream));
                CUDA_CHECK_RET(cudaMemcpyAsync(tmp2, tmp, bytes_tmp, cudaMemcpyDeviceToDevice, stream));
            } else if (filter_id == 3) {
                CUDA_CHECK_RET(iir_rows_biquad_scan_d(tmp, tmp2, h, w, 0.2, 0.2, 0.2, 0.3, -0.1, block_t, block_in4, row_state, border_mode, (double)border_const, stream));
            } else if (filter_id == 4) {
                CUDA_CHECK_RET(iir_rows_biquad_scan_d(tmp, tmp2, h, w, 0.2, 0.2, 0.2, 0.3, -0.1, block_t, block_in4, row_state, border_mode, (double)border_const, stream));
                CUDA_CHECK_RET(iir_rows_biquad_scan_d(tmp2, tmp, h, w, 0.3, 0.1, 0.1, 0.2, -0.05, block_t, block_in4, row_state, border_mode, (double)border_const, stream));
                CUDA_CHECK_RET(cudaMemcpyAsync(tmp2, tmp, bytes_tmp, cudaMemcpyDeviceToDevice, stream));
            } else if (filter_id == 8) {
                CUDA_CHECK_RET(iir_rows_biquad_scan_d(tmp, tmp2, h, w, 0.2, 0.2, 0.2, 0.3, -0.1, block_t, block_in4, row_state, border_mode, (double)border_const, stream));
            } else {
                CUDA_CHECK_RET(launch_rows_d(tmp, tmp2, h, w, filter_id, border_mode, (double)border_const, stream, tmp, tmp2));
            }

            transpose32_d<<<tgrid2, tblock, 0, stream>>>(tmp2, dout, h, w);
            CUDA_CHECK_RET(cudaGetLastError());
        } else {
            double* tmp = nullptr;
            double* tmp2 = nullptr;
            void* base = nullptr;
            size_t bytes = size_t(w) * size_t(h) * sizeof(double) * 2;
            CUDA_CHECK_RET(get_workspace_bytes(bytes, &base));
            tmp = (double*)base;
            tmp2 = (double*)((char*)base + size_t(w) * size_t(h) * sizeof(double));

            CUDA_CHECK_RET(launch_rows_d(din, dout, w, h, filter_id, border_mode, (double)border_const, stream, tmp, tmp2));
            transpose32_d<<<tgrid, tblock, 0, stream>>>(dout, tmp, w, h);
            CUDA_CHECK_RET(cudaGetLastError());
            CUDA_CHECK_RET(launch_rows_d(tmp, tmp2, h, w, filter_id, border_mode, (double)border_const, stream, tmp, tmp2));
            transpose32_d<<<tgrid2, tblock, 0, stream>>>(tmp2, dout, h, w);
            CUDA_CHECK_RET(cudaGetLastError());
        }
    } else if (precision == IIR2D_PREC_MIXED) {
        const float* fin = (const float*)in;
        float* fout = (float*)out;
        const bool use_scan = (filter_id == 1 || filter_id == 2 || filter_id == 3 || filter_id == 4 || filter_id == 8);
        if (use_scan) {
            const size_t n = size_t(w) * size_t(h);
            const int BLOCK = 256;
            int num_blocks_row = (w + BLOCK - 1) / BLOCK;
            int num_blocks_col = (h + BLOCK - 1) / BLOCK;
            size_t max_blocks = (size_t)h * (size_t)num_blocks_row;
            size_t alt_blocks = (size_t)w * (size_t)num_blocks_col;
            if (alt_blocks > max_blocks) max_blocks = alt_blocks;

            size_t max_wh = (size_t)(w > h ? w : h);
            size_t bytes_tmp = n * sizeof(double);
            size_t bytes_block_d = max_blocks * sizeof(double);
            size_t bytes_block_t = max_blocks * sizeof(Transform4D);
            size_t bytes_block_in4 = max_blocks * sizeof(double4);
            size_t bytes_row = max_wh * sizeof(double);
            size_t bytes_row4 = max_wh * sizeof(double4);
            size_t total = 0;
            total += align_up(bytes_tmp, 256);
            total += align_up(bytes_tmp, 256);
            total += align_up(bytes_block_d, 256) * 3;
            total += align_up(bytes_block_t, 256);
            total += align_up(bytes_block_in4, 256);
            total += align_up(bytes_row, 256);
            total += align_up(bytes_row4, 256);
            total += align_up(n * sizeof(float), 256); // scratch float buffer for output

            void* base = nullptr;
            CUDA_CHECK_RET(get_workspace_bytes(total, &base));
            char* ptr = (char*)base;

            double* tmp = (double*)ptr; ptr += align_up(bytes_tmp, 256);
            double* tmp2 = (double*)ptr; ptr += align_up(bytes_tmp, 256);
            double* block_a = (double*)ptr; ptr += align_up(bytes_block_d, 256);
            double* block_p = (double*)ptr; ptr += align_up(bytes_block_d, 256);
            double* block_in = (double*)ptr; ptr += align_up(bytes_block_d, 256);
            Transform4D* block_t = (Transform4D*)ptr; ptr += align_up(bytes_block_t, 256);
            double4* block_in4 = (double4*)ptr; ptr += align_up(bytes_block_in4, 256);
            double* row_yprev = (double*)ptr; ptr += align_up(bytes_row, 256);
            double4* row_state = (double4*)ptr; ptr += align_up(bytes_row4, 256);
            float* scratch_f = (float*)ptr; ptr += align_up(n * sizeof(float), 256);

            int threads = 256;
            int blocks = (int)((n + threads - 1) / threads);
            convert_f2d<<<blocks, threads, 0, stream>>>(fin, tmp, (int)n);
            CUDA_CHECK_RET(cudaGetLastError());

            if (filter_id == 1 || filter_id == 7) {
                double b0 = 1.0 - 0.85;
                double a1 = 0.85;
                CUDA_CHECK_RET(iir_rows_scan_first_d(tmp, tmp2, w, h, b0, 0.0, a1, block_a, block_p, block_in, row_yprev, border_mode, (double)border_const, stream));
            } else if (filter_id == 2) {
                double a = 0.75, b = 0.25;
                CUDA_CHECK_RET(iir_rows_scan_first_d(tmp, tmp2, w, h, b, 0.0, a, block_a, block_p, block_in, row_yprev, border_mode, (double)border_const, stream));
                CUDA_CHECK_RET(iir_rows_scan_first_d(tmp2, tmp, w, h, b, 0.0, a, block_a, block_p, block_in, row_yprev, border_mode, (double)border_const, stream));
                CUDA_CHECK_RET(cudaMemcpyAsync(tmp2, tmp, bytes_tmp, cudaMemcpyDeviceToDevice, stream));
            } else if (filter_id == 3) {
                CUDA_CHECK_RET(iir_rows_biquad_scan_d(tmp, tmp2, w, h, 0.2, 0.2, 0.2, 0.3, -0.1, block_t, block_in4, row_state, border_mode, (double)border_const, stream));
            } else if (filter_id == 4) {
                CUDA_CHECK_RET(iir_rows_biquad_scan_d(tmp, tmp2, w, h, 0.2, 0.2, 0.2, 0.3, -0.1, block_t, block_in4, row_state, border_mode, (double)border_const, stream));
                CUDA_CHECK_RET(iir_rows_biquad_scan_d(tmp2, tmp, w, h, 0.3, 0.1, 0.1, 0.2, -0.05, block_t, block_in4, row_state, border_mode, (double)border_const, stream));
                CUDA_CHECK_RET(cudaMemcpyAsync(tmp2, tmp, bytes_tmp, cudaMemcpyDeviceToDevice, stream));
            } else if (filter_id == 8) {
                CUDA_CHECK_RET(iir_rows_biquad_scan_d(tmp, tmp2, w, h, 0.2, 0.2, 0.2, 0.3, -0.1, block_t, block_in4, row_state, border_mode, (double)border_const, stream));
            }

            transpose32_d<<<tgrid, tblock, 0, stream>>>(tmp2, tmp, w, h);
            CUDA_CHECK_RET(cudaGetLastError());

            if (filter_id == 1 || filter_id == 7) {
                double b0 = 1.0 - 0.85;
                double a1 = 0.85;
                CUDA_CHECK_RET(iir_rows_scan_first_d(tmp, tmp2, h, w, b0, 0.0, a1, block_a, block_p, block_in, row_yprev, border_mode, (double)border_const, stream));
            } else if (filter_id == 2) {
                double a = 0.75, b = 0.25;
                CUDA_CHECK_RET(iir_rows_scan_first_d(tmp, tmp2, h, w, b, 0.0, a, block_a, block_p, block_in, row_yprev, border_mode, (double)border_const, stream));
                CUDA_CHECK_RET(iir_rows_scan_first_d(tmp2, tmp, h, w, b, 0.0, a, block_a, block_p, block_in, row_yprev, border_mode, (double)border_const, stream));
                CUDA_CHECK_RET(cudaMemcpyAsync(tmp2, tmp, bytes_tmp, cudaMemcpyDeviceToDevice, stream));
            } else if (filter_id == 3) {
                CUDA_CHECK_RET(iir_rows_biquad_scan_d(tmp, tmp2, h, w, 0.2, 0.2, 0.2, 0.3, -0.1, block_t, block_in4, row_state, border_mode, (double)border_const, stream));
            } else if (filter_id == 4) {
                CUDA_CHECK_RET(iir_rows_biquad_scan_d(tmp, tmp2, h, w, 0.2, 0.2, 0.2, 0.3, -0.1, block_t, block_in4, row_state, border_mode, (double)border_const, stream));
                CUDA_CHECK_RET(iir_rows_biquad_scan_d(tmp2, tmp, h, w, 0.3, 0.1, 0.1, 0.2, -0.05, block_t, block_in4, row_state, border_mode, (double)border_const, stream));
                CUDA_CHECK_RET(cudaMemcpyAsync(tmp2, tmp, bytes_tmp, cudaMemcpyDeviceToDevice, stream));
            } else if (filter_id == 8) {
                CUDA_CHECK_RET(iir_rows_biquad_scan_d(tmp, tmp2, h, w, 0.2, 0.2, 0.2, 0.3, -0.1, block_t, block_in4, row_state, border_mode, (double)border_const, stream));
            }

            transpose32_d<<<tgrid2, tblock, 0, stream>>>(tmp2, tmp, h, w);
            CUDA_CHECK_RET(cudaGetLastError());
            convert_d2f<<<blocks, threads, 0, stream>>>(tmp, fout, (int)n);
            CUDA_CHECK_RET(cudaGetLastError());
        } else {
            float* tmp = nullptr;
            float* tmp2 = nullptr;
            void* base = nullptr;
            size_t bytes = size_t(w) * size_t(h) * sizeof(float) * 2;
            CUDA_CHECK_RET(get_workspace_bytes(bytes, &base));
            tmp = (float*)base;
            tmp2 = (float*)((char*)base + size_t(w) * size_t(h) * sizeof(float));

            CUDA_CHECK_RET(launch_rows_mixed(fin, fout, w, h, filter_id, border_mode, border_const, stream, tmp, tmp2));
            transpose32_f<<<tgrid, tblock, 0, stream>>>(fout, tmp, w, h);
            CUDA_CHECK_RET(cudaGetLastError());
            CUDA_CHECK_RET(launch_rows_mixed(tmp, tmp2, h, w, filter_id, border_mode, border_const, stream, tmp, tmp2));
            transpose32_f<<<tgrid2, tblock, 0, stream>>>(tmp2, fout, h, w);
            CUDA_CHECK_RET(cudaGetLastError());
        }
    } else {
        const float* fin = (const float*)in;
        float* fout = (float*)out;
        // workspace layout
        const size_t n = size_t(w) * size_t(h);
        const int BLOCK = 256;
        int num_blocks_row = (w + BLOCK - 1) / BLOCK;
        int num_blocks_col = (h + BLOCK - 1) / BLOCK;
        size_t max_blocks = (size_t)h * (size_t)num_blocks_row;
        size_t alt_blocks = (size_t)w * (size_t)num_blocks_col;
        if (alt_blocks > max_blocks) max_blocks = alt_blocks;

        size_t off = 0;
        size_t bytes_tmp = n * sizeof(float);
        size_t bytes_block_f = max_blocks * sizeof(float);
        size_t bytes_block_t = max_blocks * sizeof(Transform4);
        size_t bytes_block_in4 = max_blocks * sizeof(float4);
        size_t max_wh = (size_t)(w > h ? w : h);
        size_t bytes_row = max_wh * sizeof(float);
        size_t bytes_row4 = max_wh * sizeof(float4);
        size_t total = 0;
        total += align_up(bytes_tmp, 256);
        total += align_up(bytes_tmp, 256);
        total += align_up(bytes_block_f, 256) * 3; // block_a, block_p, block_in
        total += align_up(bytes_block_t, 256);
        total += align_up(bytes_block_in4, 256);
        total += align_up(bytes_row, 256);
        total += align_up(bytes_row4, 256);

        void* base = nullptr;
        CUDA_CHECK_RET(get_workspace_bytes(total, &base));
        char* ptr = (char*)base;

        float* tmp = (float*)ptr; ptr += align_up(bytes_tmp, 256);
        float* tmp2 = (float*)ptr; ptr += align_up(bytes_tmp, 256);
        float* block_a = (float*)ptr; ptr += align_up(bytes_block_f, 256);
        float* block_p = (float*)ptr; ptr += align_up(bytes_block_f, 256);
        float* block_in = (float*)ptr; ptr += align_up(bytes_block_f, 256);
        Transform4* block_t = (Transform4*)ptr; ptr += align_up(bytes_block_t, 256);
        float4* block_in4 = (float4*)ptr; ptr += align_up(bytes_block_in4, 256);
        float* row_yprev = (float*)ptr; ptr += align_up(bytes_row, 256);
        float4* row_state = (float4*)ptr; ptr += align_up(bytes_row4, 256);

        // row pass
        if (filter_id == 1 || filter_id == 7) {
            float b0 = (filter_id == 7) ? (1.0f - 0.85f) : (1.0f - 0.85f);
            float a1 = 0.85f;
            CUDA_CHECK_RET(iir_rows_scan_first(fin, fout, w, h, b0, 0.0f, a1, block_a, block_p, block_in, row_yprev, border_mode, border_const, stream));
        } else if (filter_id == 2) {
            float a = 0.75f, b = 0.25f;
            CUDA_CHECK_RET(iir_rows_scan_first(fin, fout, w, h, b, 0.0f, a, block_a, block_p, block_in, row_yprev, border_mode, border_const, stream));
            CUDA_CHECK_RET(iir_rows_scan_first(fout, tmp, w, h, b, 0.0f, a, block_a, block_p, block_in, row_yprev, border_mode, border_const, stream));
            CUDA_CHECK_RET(cudaMemcpyAsync(fout, tmp, bytes_tmp, cudaMemcpyDeviceToDevice, stream));
        } else if (filter_id == 3) {
            CUDA_CHECK_RET(iir_rows_biquad_scan(fin, fout, w, h, 0.2f, 0.2f, 0.2f, 0.3f, -0.1f, block_t, block_in4, row_state, border_mode, border_const, stream));
        } else if (filter_id == 4) {
            CUDA_CHECK_RET(iir_rows_biquad_scan(fin, tmp, w, h, 0.2f, 0.2f, 0.2f, 0.3f, -0.1f, block_t, block_in4, row_state, border_mode, border_const, stream));
            CUDA_CHECK_RET(iir_rows_biquad_scan(tmp, fout, w, h, 0.3f, 0.1f, 0.1f, 0.2f, -0.05f, block_t, block_in4, row_state, border_mode, border_const, stream));
        } else if (filter_id == 8) {
            CUDA_CHECK_RET(iir_rows_biquad_scan(fin, fout, w, h, 0.2f, 0.2f, 0.2f, 0.3f, -0.1f, block_t, block_in4, row_state, border_mode, border_const, stream));
        } else {
            CUDA_CHECK_RET(launch_rows_f(fin, fout, w, h, filter_id, border_mode, border_const, stream, tmp, tmp2));
        }

        transpose32_f<<<tgrid, tblock, 0, stream>>>(fout, tmp, w, h);
        CUDA_CHECK_RET(cudaGetLastError());

        // col pass (on transposed)
        if (filter_id == 1 || filter_id == 7) {
            float b0 = (filter_id == 7) ? (1.0f - 0.85f) : (1.0f - 0.85f);
            float a1 = 0.85f;
            CUDA_CHECK_RET(iir_rows_scan_first(tmp, tmp2, h, w, b0, 0.0f, a1, block_a, block_p, block_in, row_yprev, border_mode, border_const, stream));
        } else if (filter_id == 2) {
            float a = 0.75f, b = 0.25f;
            CUDA_CHECK_RET(iir_rows_scan_first(tmp, tmp2, h, w, b, 0.0f, a, block_a, block_p, block_in, row_yprev, border_mode, border_const, stream));
            CUDA_CHECK_RET(iir_rows_scan_first(tmp2, tmp, h, w, b, 0.0f, a, block_a, block_p, block_in, row_yprev, border_mode, border_const, stream));
            CUDA_CHECK_RET(cudaMemcpyAsync(tmp2, tmp, bytes_tmp, cudaMemcpyDeviceToDevice, stream));
        } else if (filter_id == 3) {
            CUDA_CHECK_RET(iir_rows_biquad_scan(tmp, tmp2, h, w, 0.2f, 0.2f, 0.2f, 0.3f, -0.1f, block_t, block_in4, row_state, border_mode, border_const, stream));
        } else if (filter_id == 4) {
            CUDA_CHECK_RET(iir_rows_biquad_scan(tmp, tmp2, h, w, 0.2f, 0.2f, 0.2f, 0.3f, -0.1f, block_t, block_in4, row_state, border_mode, border_const, stream));
            CUDA_CHECK_RET(iir_rows_biquad_scan(tmp2, tmp, h, w, 0.3f, 0.1f, 0.1f, 0.2f, -0.05f, block_t, block_in4, row_state, border_mode, border_const, stream));
            CUDA_CHECK_RET(cudaMemcpyAsync(tmp2, tmp, bytes_tmp, cudaMemcpyDeviceToDevice, stream));
        } else if (filter_id == 8) {
            CUDA_CHECK_RET(iir_rows_biquad_scan(tmp, tmp2, h, w, 0.2f, 0.2f, 0.2f, 0.3f, -0.1f, block_t, block_in4, row_state, border_mode, border_const, stream));
        } else {
            CUDA_CHECK_RET(launch_rows_f(tmp, tmp2, h, w, filter_id, border_mode, border_const, stream, tmp, tmp2));
        }

        transpose32_f<<<tgrid2, tblock, 0, stream>>>(tmp2, fout, h, w);
        CUDA_CHECK_RET(cudaGetLastError());
    }

    return IIR2D_STATUS_OK;
}

int iir2d_forward_cuda(const void* in, void* out, const IIR2D_Params* params) {
    return iir2d_forward_cuda_stream(in, out, params, nullptr);
}

void iir2d_custom_call(void* stream, void** buffers, const char* opaque, std::size_t opaque_len) {
    if (!buffers || !opaque || opaque_len < sizeof(IIR2D_Params)) return;
    IIR2D_Params params;
    std::memcpy(&params, opaque, sizeof(IIR2D_Params));
    const void* in = buffers[0];
    void* out = buffers[1];
    iir2d_forward_cuda_stream(in, out, &params, stream);
}

const char* iir2d_status_string(int status_code) {
    switch (status_code) {
        case IIR2D_STATUS_OK:
            return "ok";
        case IIR2D_STATUS_INVALID_ARGUMENT:
            return "invalid_argument";
        case IIR2D_STATUS_INVALID_DIMENSION:
            return "invalid_dimension";
        case IIR2D_STATUS_INVALID_FILTER_ID:
            return "invalid_filter_id";
        case IIR2D_STATUS_INVALID_BORDER_MODE:
            return "invalid_border_mode";
        case IIR2D_STATUS_INVALID_PRECISION:
            return "invalid_precision";
        case IIR2D_STATUS_NULL_POINTER:
            return "null_pointer";
        case IIR2D_STATUS_WORKSPACE_ERROR:
            return "workspace_error";
        case IIR2D_STATUS_CUDA_ERROR:
            return "cuda_error";
        default:
            return "unknown_status";
    }
}

int iir2d_api_version_major(void) {
    return IIR2D_API_VERSION_MAJOR;
}

int iir2d_api_version_minor(void) {
    return IIR2D_API_VERSION_MINOR;
}

int iir2d_api_version_patch(void) {
    return IIR2D_API_VERSION_PATCH;
}

int iir2d_api_version_packed(void) {
    return IIR2D_API_VERSION_MAJOR * 10000 + IIR2D_API_VERSION_MINOR * 100 + IIR2D_API_VERSION_PATCH;
}

const char* iir2d_build_fingerprint(void) {
    static char fp[96];
    static bool initialized = false;
    if (!initialized) {
        std::snprintf(
            fp,
            sizeof(fp),
            "iir2d/%d.%d.%d %s %s",
            IIR2D_API_VERSION_MAJOR,
            IIR2D_API_VERSION_MINOR,
            IIR2D_API_VERSION_PATCH,
            __DATE__,
            __TIME__);
        initialized = true;
    }
    return fp;
}
