#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "iir2d_core.h"

static torch::Tensor iir2d_forward(torch::Tensor input,
                                   int filter_id,
                                   int border_mode,
                                   double border_const,
                                   int precision) {
    TORCH_CHECK(input.is_cuda(), "iir2d_forward: input must be CUDA");
    TORCH_CHECK(input.dim() == 2, "iir2d_forward: input must be [H, W]");
    TORCH_CHECK(input.is_contiguous(), "iir2d_forward: input must be contiguous");

    const int height = (int)input.size(0);
    const int width = (int)input.size(1);

    if (precision == IIR2D_PREC_F64) {
        TORCH_CHECK(input.scalar_type() == at::kDouble, "precision f64 requires float64 input");
    } else {
        TORCH_CHECK(input.scalar_type() == at::kFloat, "precision f32/mixed requires float32 input");
    }

    c10::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

    torch::Tensor output = torch::zeros_like(input);

    IIR2D_Params params;
    params.width = width;
    params.height = height;
    params.filter_id = filter_id;
    params.border_mode = border_mode;
    params.border_const = (float)border_const;
    params.precision = precision;

    int rc = iir2d_forward_cuda_stream(input.data_ptr(), output.data_ptr(), &params, stream);
    TORCH_CHECK(rc == IIR2D_STATUS_OK, "iir2d_forward_cuda failed (", rc, "): ", iir2d_status_string(rc));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &iir2d_forward, "IIR2D forward (CUDA)");
}
