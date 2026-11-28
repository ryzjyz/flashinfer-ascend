"""Reviewed dense IndexerPrologueQW capabilities and wheel targets."""

from flashinfer.runtime import CapabilityCase, CapabilitySpec
from tests.inventory import ReviewedKernel


KERNELS = tuple(
    ReviewedKernel(
        "ascendc/attention/indexer_prologue_qw/native/op_package/" + soc,
        CapabilitySpec(
            op="attention.indexer_prologue_qw",
            soc_models=(soc,),
            dtypes=("bfloat16", "int8", "float16", "float32"),
            cases=(CapabilityCase(
                name="dense_w8a8_mixed",
                compute_dtypes=("int8",),
                accumulator_dtypes=("int32",),
                fusion_ops=("dynamic_quantize", "quantized_matmul", "rope", "dynamic_quantize"),
                auxiliary_outputs=("q_scale", "w"),
                specializations=(("quant.activation", ("per_token_int8",)),
                                 ("quant.weight", ("per_channel_int8",))),
            ),),
        ),
        backend="ascendc", native_recipe=("indexer_prologue_qw", "native"),
    )
    for soc in ("Ascend910B", "Ascend910C")
)
