"""Reviewed prequantized IndexerPrologueQW capabilities and wheel targets."""

from flashinfer.runtime import CapabilityCase, CapabilitySpec
from tests.inventory import ReviewedKernel


KERNELS = tuple(
    ReviewedKernel(
        "ascendc/attention/indexer_prologue_qw_prequantized/native/op_package/" + soc,
        CapabilitySpec(
            op="attention.indexer_prologue_qw_prequantized",
            soc_models=(soc,),
            dtypes=("bfloat16", "int8", "float16", "float32"),
            cases=(CapabilityCase(
                name="caller_prequantized_nz_mixed",
                compute_dtypes=("int8",),
                accumulator_dtypes=("int32",),
                fusion_ops=("quantized_matmul", "rope", "dynamic_quantize"),
                auxiliary_outputs=("q_scale", "w"),
            ),),
        ),
        backend="ascendc", native_recipe=("indexer_prologue_qw_prequantized", "native"),
    )
    for soc in ("Ascend910B", "Ascend910C")
)
