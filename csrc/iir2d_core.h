#pragma once

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define IIR2D_EXPORT __declspec(dllexport)
#else
#define IIR2D_EXPORT __attribute__((visibility("default")))
#endif

// Public C API version (semantic versioning).
// ABI compatibility is guaranteed for patch/minor updates within the same major.
#define IIR2D_API_VERSION_MAJOR 1
#define IIR2D_API_VERSION_MINOR 0
#define IIR2D_API_VERSION_PATCH 0

typedef enum IIR2D_BorderMode {
    IIR2D_BORDER_CLAMP = 0,
    IIR2D_BORDER_MIRROR = 1,
    IIR2D_BORDER_WRAP = 2,
    IIR2D_BORDER_CONSTANT = 3
} IIR2D_BorderMode;

typedef enum IIR2D_Precision {
    IIR2D_PREC_F32 = 0,
    IIR2D_PREC_MIXED = 1,
    IIR2D_PREC_F64 = 2
} IIR2D_Precision;

typedef enum IIR2D_Status {
    IIR2D_STATUS_OK = 0,
    IIR2D_STATUS_INVALID_ARGUMENT = -1,
    IIR2D_STATUS_INVALID_DIMENSION = -2,
    IIR2D_STATUS_INVALID_FILTER_ID = -3,
    IIR2D_STATUS_INVALID_BORDER_MODE = -4,
    IIR2D_STATUS_INVALID_PRECISION = -5,
    IIR2D_STATUS_NULL_POINTER = -6,
    IIR2D_STATUS_WORKSPACE_ERROR = -7,
    IIR2D_STATUS_CUDA_ERROR = -8
} IIR2D_Status;

typedef struct IIR2D_Params {
    int width;
    int height;
    int filter_id;   // 1..8
    int border_mode; // IIR2D_BorderMode
    float border_const;
    int precision;   // IIR2D_Precision
} IIR2D_Params;

// Launch on current CUDA stream.
// in/out are device pointers to contiguous HxW (row-major).
// dtype is float for PREC_F32/PREC_MIXED, double for PREC_F64.
// Returns IIR2D_Status.
IIR2D_EXPORT int iir2d_forward_cuda(const void* in, void* out, const IIR2D_Params* params);

// Launch on a specific CUDA stream (opaque pointer).
// Returns IIR2D_Status.
IIR2D_EXPORT int iir2d_forward_cuda_stream(const void* in, void* out, const IIR2D_Params* params, void* stream);

// Returns a static string for an IIR2D_Status code.
IIR2D_EXPORT const char* iir2d_status_string(int status_code);

// JAX custom call entrypoint (CUDA backend)
IIR2D_EXPORT void iir2d_custom_call(void* stream, void** buffers, const char* opaque, std::size_t opaque_len);

#ifdef __cplusplus
}
#endif
