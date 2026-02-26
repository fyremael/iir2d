cmake_minimum_required(VERSION 3.18)
project(iir2d_op LANGUAGES C CXX)

option(IIR2D_BUILD_CPU_STUB "Build CPU stub library without CUDA dependency" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (IIR2D_BUILD_CPU_STUB)
    add_library(iir2d_jax SHARED
        csrc/iir2d_core_stub.cpp
    )
else()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_library(iir2d_jax SHARED
        csrc/iir2d_core.cu
    )
    set_target_properties(iir2d_jax PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

target_include_directories(iir2d_jax PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/csrc)

if (WIN32)
    set_target_properties(iir2d_jax PROPERTIES OUTPUT_NAME "iir2d_jax")
else()
    set_target_properties(iir2d_jax PROPERTIES OUTPUT_NAME "iir2d_jax")
endif()
