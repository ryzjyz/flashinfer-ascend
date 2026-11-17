/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dsv41_mla_prolog_def.cpp
 * \brief OpDef skeleton for FiDsv41MlaProlog (BF16 only, A3/arch22).
 *
 * This declares the public tensor order and dtypes from the standalone host
 * header. It does not prove that the current CATLASS-launched kernel can be
 * built or executed through the generated ACLNN adapter.
 */

#include "fi_host_checks.h"
#include "arch22/dsv41_mla_prolog_tiling.cpp"
#include "dsv41_mla_prolog_infershape.cpp"
#include "register/op_def_registry.h"

#define FI_STRING_INNER(x) #x
#define FI_STRING(x) FI_STRING_INNER(x)

namespace ops {
class FiDsv41MlaProlog : public OpDef {
public:
    explicit FiDsv41MlaProlog(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        this->Input("w_qa")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        this->Input("gamma_qa")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        this->Input("w_qb")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_FRACTAL_NZ});
        this->Input("w_qb_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        this->Input("w_kv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        this->Input("gamma_kv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        this->Input("rope_cos")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("rope_sin")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("slot_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});
        // kv_cache is an input whose addressed rows are updated in place.
        // OpDef cannot express input mutation; the ACLNN parameter is an input
        // pointer. The public mutation contract needs maintainer review.
        this->Input("kv_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        this->Output("q_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});

        SetInferShape(InferShapeDsv41MlaProlog).SetInferDataType(InferDataTypeDsv41MlaProlog);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig(FI_STRING(FI_COMPUTE_UNIT), aicore_config);
    }
};
OP_ADD(FiDsv41MlaProlog);
} // namespace ops
