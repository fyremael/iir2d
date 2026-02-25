cmake_minimum_required(VERSION 3.18)
project(iir2d_op LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(iir2d_jax SHARED
    csrc/iir2d_core.cu
)
target_include_directories(iir2d_jax PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
set_target_properties(iir2d_jax PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if (WIN32)
    set_target_properties(iir2d_jax PROPERTIES OUTPUT_NAME "iir2d_jax")
else()
    set_target_properties(iir2d_jax PROPERTIES OUTPUT_NAME "iir2d_jax")
endif()
