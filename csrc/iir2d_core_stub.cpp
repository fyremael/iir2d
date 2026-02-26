#include "iir2d_core.h"

#include <cstddef>

namespace {

int validate_params(const void* in, void* out, const IIR2D_Params* params) {
    if (!params) {
        return IIR2D_STATUS_NULL_POINTER;
    }
    if (!in || !out) {
        return IIR2D_STATUS_NULL_POINTER;
    }
    if (params->width <= 0 || params->height <= 0) {
        return IIR2D_STATUS_INVALID_DIMENSION;
    }
    if (params->filter_id < 1 || params->filter_id > 8) {
        return IIR2D_STATUS_INVALID_FILTER_ID;
    }
    if (params->border_mode < IIR2D_BORDER_CLAMP || params->border_mode > IIR2D_BORDER_CONSTANT) {
        return IIR2D_STATUS_INVALID_BORDER_MODE;
    }
    if (params->precision < IIR2D_PREC_F32 || params->precision > IIR2D_PREC_F64) {
        return IIR2D_STATUS_INVALID_PRECISION;
    }
    return IIR2D_STATUS_OK;
}

}  // namespace

extern "C" IIR2D_EXPORT int iir2d_forward_cuda(const void* in, void* out, const IIR2D_Params* params) {
    int rc = validate_params(in, out, params);
    if (rc != IIR2D_STATUS_OK) {
        return rc;
    }
    return IIR2D_STATUS_CUDA_ERROR;
}

extern "C" IIR2D_EXPORT int iir2d_forward_cuda_stream(
    const void* in,
    void* out,
    const IIR2D_Params* params,
    void* stream
) {
    (void)stream;
    int rc = validate_params(in, out, params);
    if (rc != IIR2D_STATUS_OK) {
        return rc;
    }
    return IIR2D_STATUS_CUDA_ERROR;
}

extern "C" IIR2D_EXPORT const char* iir2d_status_string(int status_code) {
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

extern "C" IIR2D_EXPORT int iir2d_api_version_major(void) {
    return IIR2D_API_VERSION_MAJOR;
}

extern "C" IIR2D_EXPORT int iir2d_api_version_minor(void) {
    return IIR2D_API_VERSION_MINOR;
}

extern "C" IIR2D_EXPORT int iir2d_api_version_patch(void) {
    return IIR2D_API_VERSION_PATCH;
}

extern "C" IIR2D_EXPORT int iir2d_api_version_packed(void) {
    return (IIR2D_API_VERSION_MAJOR * 10000) + (IIR2D_API_VERSION_MINOR * 100) + IIR2D_API_VERSION_PATCH;
}

extern "C" IIR2D_EXPORT const char* iir2d_build_fingerprint(void) {
    return "iir2d_cpu_stub:api=1.0.0";
}

extern "C" IIR2D_EXPORT void iir2d_custom_call(void* stream, void** buffers, const char* opaque, std::size_t opaque_len) {
    (void)stream;
    (void)buffers;
    (void)opaque;
    (void)opaque_len;
}
