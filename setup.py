import os

from setuptools import find_packages, setup

SKIP_EXT_VALUES = {"1", "true", "yes", "on"}


def _should_skip_ext() -> bool:
    return os.environ.get("IIR2D_SKIP_EXT", "").strip().lower() in SKIP_EXT_VALUES


def _discover_packages() -> list[str]:
    python_packages = find_packages(where="python")
    root_packages = find_packages(where=".", include=["iir2d_video", "iir2d_video.*", "scripts", "scripts.*"])
    return sorted(set(python_packages + root_packages))


def _build_ext_config():
    if _should_skip_ext():
        return [], {}

    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except Exception as exc:  # pragma: no cover - exercised in wheel builds
        raise RuntimeError(
            "Building CUDA extension requires torch build tooling. "
            "Set IIR2D_SKIP_EXT=1 to build a pure-Python wheel (includes iir2d_video + scripts)."
        ) from exc

    ext_modules = [
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
    ]
    return ext_modules, {"build_ext": BuildExtension}


ext_modules, cmdclass = _build_ext_config()

setup(
    name="iir2d_torch",
    package_dir={
        "": "python",
        "iir2d_video": "iir2d_video",
        "scripts": "scripts",
    },
    packages=_discover_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
