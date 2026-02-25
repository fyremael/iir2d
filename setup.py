from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="iir2d_torch",
    package_dir={"": "python"},
    packages=find_packages("python"),
    ext_modules=[
        CUDAExtension(
            name="iir2d_torch_ext",
            sources=[
                "csrc/torch_binding.cpp",
                "csrc/iir2d_core.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
