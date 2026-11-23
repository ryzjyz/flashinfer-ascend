"""Map caller-prequantized QW tensors onto the shared 15-Tensor mixed ABI."""

from dataclasses import dataclass

from flashinfer.jit.tensor_adapter import package_tensor_runner

N_HEADS = 32
HEAD_DIM = 128
DIM_X = 5120
DIM_QR = 1280
DIM_Q = N_HEADS * HEAD_DIM


def admission(recipe, signature):
    import math

    names = recipe.tensor_parameter_names
    attrs = dict(signature.specializations)
    try:
        softmax_scale = float(attrs["attribute.softmax_scale"])
    except (KeyError, TypeError, ValueError):
        return ("prequantized QW requires a finite softmax_scale",)
    if not math.isfinite(softmax_scale):
        return ("prequantized QW requires a finite softmax_scale",)
    if len(signature.shapes) != 10 or len(signature.dtypes) != 10:
        return ("public prequantized QW signature must contain seven inputs and three outputs",)
    if len(names) != 15:
        return ("prequantized QW native ABI requires 15 Tensors",)
    tokens = signature.shapes[0][0]
    shapes = {
        "qr": signature.shapes[1],
        "activation_q": (tokens, DIM_QR),
        "weight_q": signature.shapes[2],
        "wqb": signature.shapes[2],
        "weight_scale": signature.shapes[3],
        "activation_scale": (tokens,),
        "qb": (tokens, DIM_Q),
        "x": signature.shapes[0],
        "ww": signature.shapes[4],
        "wb": (tokens, N_HEADS),
        "rope_sin": signature.shapes[5],
        "rope_cos": signature.shapes[6],
        "q": signature.shapes[7],
        "q_scale": signature.shapes[8],
        "w": signature.shapes[9],
    }
    dtypes = {
        "qr": "bfloat16", "activation_q": "int8", "weight_q": "int8",
        "wqb": "int8", "weight_scale": "float32", "activation_scale": "float32",
        "qb": "bfloat16", "x": "bfloat16", "ww": "bfloat16", "wb": "bfloat16",
        "rope_sin": "float32", "rope_cos": "float32", "q": "int8",
        "q_scale": "float16", "w": "float16",
    }
    if signature.physical_layouts[2] != "NZ":
        return ("prequantized QW requires a physical FRACTAL_NZ wqb_nz",)
    try:
        recipe.validate_tensor_metadata(shapes=shapes, dtypes=dtypes)
        recipe.validate_attributes({"w_scale": softmax_scale * (32 ** -0.5)})
    except (ValueError, TypeError) as error:
        return (str(error),)
    return ()


@dataclass(frozen=True)
class IndexerPrologueQwPrequantizedRunner:
    native: object

    def __call__(self, x, qr, wqb_nz, wqb_scale, ww, rope_sin, rope_cos,
                 q, q_scale, w, *, softmax_scale):
        import torch

        tokens = x.shape[0]
        device = x.device
        activation_q = torch.empty((tokens, DIM_QR), dtype=torch.int8, device=device)
        activation_scale = torch.empty((tokens,), dtype=torch.float32, device=device)
        qb = torch.empty((tokens, DIM_Q), dtype=torch.bfloat16, device=device)
        wb = torch.empty((tokens, N_HEADS), dtype=torch.bfloat16, device=device)
        return self.native(
            qr, activation_q, wqb_nz, wqb_nz, wqb_scale, activation_scale, qb,
            x, ww, wb, rope_sin, rope_cos, q, q_scale, w,
            w_scale=softmax_scale * (32 ** -0.5),
        )


def make_runner(module, spec, recipe):
    return IndexerPrologueQwPrequantizedRunner(
        package_tensor_runner(module, recipe, spec=spec))
