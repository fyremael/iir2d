import importlib

import torch


def _load_ext():
    try:
        return importlib.import_module("iir2d_torch_ext")
    except Exception as e:
        raise RuntimeError("iir2d_torch_ext not built. Run `python setup.py install` in iir2d_op.") from e


class _IIR2DAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, filter_id, border_mode, border_const, prec_mode):
        ctx.filter_id = int(filter_id)
        ctx.border_mode = int(border_mode)
        ctx.border_const = float(border_const)
        ctx.prec_mode = int(prec_mode)
        return _load_ext().forward(x, ctx.filter_id, ctx.border_mode, ctx.border_const, ctx.prec_mode)

    @staticmethod
    def backward(ctx, grad_output):
        # Linear operator; use same filter as an approximate adjoint.
        grad_input = _load_ext().forward(grad_output, ctx.filter_id, ctx.border_mode, ctx.border_const, ctx.prec_mode)
        return grad_input, None, None, None, None

def iir2d(x, filter_id=1, border="mirror", border_const=0.0, precision="f32"):
    _load_ext()
    border_mode = {"clamp": 0, "mirror": 1, "wrap": 2, "constant": 3}.get(border, 1)
    prec_mode = {"f32": 0, "mixed": 1, "f64": 2}.get(precision, 0)
    if x.dim() == 2:
        return _IIR2DAutograd.apply(x, int(filter_id), int(border_mode), float(border_const), int(prec_mode))
    if x.dim() == 3:
        # CHW
        c, h, w = x.shape
        outs = []
        for i in range(c):
            outs.append(_IIR2DAutograd.apply(x[i], int(filter_id), int(border_mode), float(border_const), int(prec_mode)))
        return torch.stack(outs, dim=0)
    if x.dim() == 4:
        # NCHW
        n, c, h, w = x.shape
        outs = []
        for ni in range(n):
            outs_c = []
            for ci in range(c):
                outs_c.append(
                    _IIR2DAutograd.apply(x[ni, ci], int(filter_id), int(border_mode), float(border_const), int(prec_mode))
                )
            outs.append(torch.stack(outs_c, dim=0))
        return torch.stack(outs, dim=0)
    raise ValueError("iir2d expects 2D, 3D (CHW), or 4D (NCHW) tensor")


class IIR2D(torch.nn.Module):
    def __init__(self, filter_id=1, border="mirror", border_const=0.0, precision="f32"):
        super().__init__()
        self.filter_id = filter_id
        self.border = border
        self.border_const = border_const
        self.precision = precision

    def forward(self, x):
        return iir2d(x, self.filter_id, self.border, self.border_const, self.precision)
