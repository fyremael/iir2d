from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
if not hasattr(jax.config, "define_bool_state"):
    def _define_bool_state(name: str, default: bool, help_str: str):
        del help_str
        setattr(jax.config, name, default)
        return default
    jax.config.define_bool_state = _define_bool_state
if not hasattr(jax, "linear_util"):
    from jax._src import linear_util as _linear_util
    jax.linear_util = _linear_util
import flax.linen as nn

from iir2d_jax import iir2d, iir2d_vjp


def _apply_iir_nhwc(
    x: jnp.ndarray,
    filter_id: int,
    border: str,
    border_const: float,
    precision: str,
    differentiable: bool,
) -> jnp.ndarray:
    """Apply the 2D IIR op channel-wise to an NHWC tensor."""
    if x.ndim != 4:
        raise ValueError(f"Expected NHWC rank-4 tensor, got shape {x.shape}")
    n, h, w, c = x.shape
    x2d = jnp.transpose(x, (0, 3, 1, 2)).reshape((n * c, h, w))
    op = iir2d_vjp if differentiable else iir2d
    y2d = op(
        x2d,
        filter_id=filter_id,
        border=border,
        border_const=border_const,
        precision=precision,
    )
    return jnp.transpose(y2d.reshape((n, c, h, w)), (0, 2, 3, 1))


class IIR2DResidual(nn.Module):
    """Single-filter residual block with learnable blend.

    y = x + sigmoid(alpha) * (IIR(x) - x)
    """

    filter_id: int = 4
    border: str = "mirror"
    border_const: float = 0.0
    precision: str = "f32"
    per_channel: bool = True
    differentiable: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        channels = x.shape[-1]
        alpha_shape = (channels,) if self.per_channel else (1,)
        alpha = self.param("alpha", nn.initializers.constant(0.0), alpha_shape)
        mix = jax.nn.sigmoid(alpha).reshape((1, 1, 1, -1))
        y = _apply_iir_nhwc(
            x,
            filter_id=self.filter_id,
            border=self.border,
            border_const=self.border_const,
            precision=self.precision,
            differentiable=self.differentiable,
        )
        return x + mix * (y - x)


class IIR2DBank(nn.Module):
    """Mixture-of-filters block over fixed filter IDs."""

    filter_ids: Sequence[int] = (1, 3, 4, 8)
    border: str = "mirror"
    border_const: float = 0.0
    precision: str = "f32"
    per_channel: bool = True
    differentiable: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if not self.filter_ids:
            raise ValueError("filter_ids must be non-empty")

        channels = x.shape[-1]
        gate_shape = (len(self.filter_ids), channels) if self.per_channel else (len(self.filter_ids), 1)
        logits = self.param("gate_logits", nn.initializers.zeros, gate_shape)
        weights = jax.nn.softmax(logits, axis=0).reshape((len(self.filter_ids), 1, 1, 1, -1))

        ys = []
        for fid in self.filter_ids:
            yi = _apply_iir_nhwc(
                x,
                filter_id=int(fid),
                border=self.border,
                border_const=self.border_const,
                precision=self.precision,
                differentiable=self.differentiable,
            )
            ys.append(yi)
        stacked = jnp.stack(ys, axis=0)
        mixed = jnp.sum(stacked * weights, axis=0)
        return mixed


class IIRDenoiseStem(nn.Module):
    """Small practical block: Conv -> IIR bank -> Conv + residual."""

    channels: int = 32
    filter_ids: Tuple[int, ...] = (1, 3, 4, 8)
    precision: str = "f32"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME", use_bias=False)(x)
        h = nn.gelu(h)
        h = IIR2DBank(
            filter_ids=self.filter_ids,
            border="mirror",
            precision=self.precision,
            per_channel=True,
            differentiable=True,
        )(h)
        h = nn.Conv(x.shape[-1], kernel_size=(3, 3), padding="SAME", use_bias=False)(h)
        return x + h
