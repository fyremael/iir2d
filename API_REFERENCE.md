# API Reference (C Core)

Header: `csrc/iir2d_core.h`

## Version Macros
1. `IIR2D_API_VERSION_MAJOR`
2. `IIR2D_API_VERSION_MINOR`
3. `IIR2D_API_VERSION_PATCH`

## Enums
1. `IIR2D_BorderMode`
   1. `IIR2D_BORDER_CLAMP`
   2. `IIR2D_BORDER_MIRROR`
   3. `IIR2D_BORDER_WRAP`
   4. `IIR2D_BORDER_CONSTANT`
2. `IIR2D_Precision`
   1. `IIR2D_PREC_F32`
   2. `IIR2D_PREC_MIXED`
   3. `IIR2D_PREC_F64`
3. `IIR2D_Status`
   1. `IIR2D_STATUS_OK`
   2. Negative error codes through `IIR2D_STATUS_CUDA_ERROR`

## Structs
1. `IIR2D_Params`
   1. `width`, `height`
   2. `filter_id` (`1..8`)
   3. `border_mode`
   4. `border_const`
   5. `precision`

## Functions
1. `int iir2d_forward_cuda(const void* in, void* out, const IIR2D_Params* params)`
   1. Launches on current CUDA stream.
2. `int iir2d_forward_cuda_stream(const void* in, void* out, const IIR2D_Params* params, void* stream)`
   1. Launches on provided CUDA stream.
3. `const char* iir2d_status_string(int status_code)`
4. `int iir2d_api_version_major(void)`
5. `int iir2d_api_version_minor(void)`
6. `int iir2d_api_version_patch(void)`
7. `int iir2d_api_version_packed(void)`
8. `const char* iir2d_build_fingerprint(void)`
9. `void iir2d_custom_call(void* stream, void** buffers, const char* opaque, std::size_t opaque_len)`

## Integration Notes
1. `in/out` must be contiguous CUDA device pointers.
2. Validate every non-zero return via `iir2d_status_string`.
3. Use runtime version/fingerprint calls for telemetry and support tickets.
