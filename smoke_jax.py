import sys

import jax
import jax.numpy as jnp
from iir2d_jax import iir2d


def main():
    print("jax", jax.__version__)
    try:
        import jaxlib  # noqa: F401
        print("jaxlib", jaxlib.__version__)
    except Exception as exc:
        print("jaxlib import failed:", exc)
        return 1

    devices = jax.devices()
    print("devices", devices)
    if not any(d.platform == "gpu" for d in devices):
        raise RuntimeError("No GPU device visible to JAX")

    x = jnp.linspace(0.0, 1.0, 64 * 64, dtype=jnp.float32).reshape(64, 64)
    y = jax.jit(iir2d, static_argnames=("filter_id", "border", "precision"))(
        x, filter_id=4, border="mirror", precision="f32"
    )
    y.block_until_ready()

    print("y stats", float(y.min()), float(y.max()), float(y.mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
