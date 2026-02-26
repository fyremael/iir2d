import ctypes
import os
import struct
from functools import partial

import jax
import jax.numpy as jnp
from jax import core, lax
from jax.interpreters import batching, mlir, xla
from jax.interpreters.mlir import ir
from jaxlib import xla_client
from jaxlib.hlo_helpers import custom_call

_lib = None
_registered = False

def _make_custom_call_capsule(func):
    ptr = ctypes.cast(func, ctypes.c_void_p).value
    pycapsule_new = ctypes.pythonapi.PyCapsule_New
    pycapsule_new.restype = ctypes.py_object
    pycapsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return pycapsule_new(ptr, b"xla._CUSTOM_CALL_TARGET", None)

def _load_lib():
    global _lib, _registered
    if _lib is not None:
        return _lib
    libname = "iir2d_jax"
    candidates = [
        os.path.join(os.path.dirname(__file__), libname + ".so"),
        os.path.join(os.path.dirname(__file__), "lib" + libname + ".so"),
        os.path.join(os.path.dirname(__file__), libname + ".dll"),
    ]
    last_error = None
    for p in candidates:
        if os.path.exists(p):
            try:
                _lib = ctypes.CDLL(p)
                break
            except OSError as exc:
                last_error = exc
    if _lib is None:
        if last_error is not None:
            raise RuntimeError(f"iir2d_jax shared library failed to load: {last_error}")
        raise RuntimeError("iir2d_jax shared library not found. Build it with CMake.")
    if not _registered:
        fn = _make_custom_call_capsule(_lib.iir2d_custom_call)
        xla_client.register_custom_call_target(
            b"iir2d_custom_call", fn, platform="gpu"
        )
        try:
            xla_client.register_custom_call_target(
                b"iir2d_custom_call", fn, platform="CUDA"
            )
        except Exception:
            pass
        _registered = True
    return _lib

def _iir2d_abstract(x, filter_id, border_mode, border_const, precision):
    return core.ShapedArray(x.shape, x.dtype)

def _iir2d_lowering(ctx, x, *, filter_id, border_mode, border_const, precision):
    _load_lib()
    x_type = ir.RankedTensorType(x.type)
    if len(x_type.shape) != 2:
        raise ValueError("iir2d: input must be 2D")
    h, w = x_type.shape
    opaque = struct.pack("<iiii fi", int(w), int(h), int(filter_id), int(border_mode), float(border_const), int(precision))
    call = custom_call(
        "iir2d_custom_call",
        result_types=[x.type],
        operands=[x],
        backend_config=opaque,
        operand_layouts=[(1, 0)],
        result_layouts=[(1, 0)],
    )
    return call.results

_iir2d_p = core.Primitive("iir2d")
_iir2d_p.def_impl(partial(xla.apply_primitive, _iir2d_p))
_iir2d_p.def_abstract_eval(_iir2d_abstract)
mlir.register_lowering(_iir2d_p, _iir2d_lowering, platform="gpu")
try:
    mlir.register_lowering(_iir2d_p, _iir2d_lowering, platform="cuda")
except Exception:
    pass

def _iir2d_batch(args, batch_dims, *, filter_id, border_mode, border_const, precision):
    (x,) = args
    (bdim,) = batch_dims
    if bdim is batching.not_mapped:
        return _iir2d_p.bind(
            x,
            filter_id=filter_id,
            border_mode=border_mode,
            border_const=border_const,
            precision=precision,
        ), batching.not_mapped
    x = jnp.moveaxis(x, bdim, 0)
    y = lax.map(
        lambda xi: _iir2d_p.bind(
            xi,
            filter_id=filter_id,
            border_mode=border_mode,
            border_const=border_const,
            precision=precision,
        ),
        x,
    )
    return y, 0

batching.primitive_batchers[_iir2d_p] = _iir2d_batch

def _iir2d_2d(x, filter_id, border_mode, border_const, precision):
    return _iir2d_p.bind(x, filter_id=filter_id, border_mode=border_mode,
                         border_const=border_const, precision=precision)

def iir2d(x, filter_id=1, border="mirror", border_const=0.0, precision="f32"):
    border_mode = {"clamp":0, "mirror":1, "wrap":2, "constant":3}.get(border, 1)
    prec_mode = {"f32":0, "mixed":1, "f64":2}.get(precision, 0)
    if prec_mode == 2 and x.dtype != jnp.float64:
        raise ValueError("precision f64 requires float64 input")
    if prec_mode != 2 and x.dtype != jnp.float32:
        raise ValueError("precision f32/mixed requires float32 input")
    if x.ndim == 2:
        return _iir2d_2d(x, filter_id, border_mode, border_const, prec_mode)
    f = _iir2d_2d
    for _ in range(x.ndim - 2):
        f = jax.vmap(f, in_axes=(0, None, None, None, None))
    return f(x, filter_id, border_mode, border_const, prec_mode)

@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
def iir2d_vjp(x, filter_id=1, border="mirror", border_const=0.0, precision="f32"):
    return iir2d(x, filter_id, border, border_const, precision)

def _iir2d_fwd(x, filter_id=1, border="mirror", border_const=0.0, precision="f32"):
    y = iir2d(x, filter_id, border, border_const, precision)
    return y, None

def _iir2d_bwd(filter_id, border, border_const, precision, res, g):
    del res
    return (iir2d(g, filter_id, border, border_const, precision),)

iir2d_vjp.defvjp(_iir2d_fwd, _iir2d_bwd)
