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
 * \file dsv41_mla_prolog_tiling.h
 * \brief Host tiling data skeleton for the DS4.1 MLA prolog op package.
 *
 * The workspace offsets mirror the standalone kernel's MakeWorkspaceLayout.
 * The current op_kernel is a CATLASS DeviceGemm host launcher and does not read
 * this struct; the device entry / tiling consumption is unresolved. This is not
 * build or run evidence.
 */

#ifndef DSV41_MLA_PROLOG_TILING_H
#define DSV41_MLA_PROLOG_TILING_H

#include <cstdint>
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Dsv41MlaPrologTilingData)
TILING_DATA_FIELD_DEF(uint32_t, t);
TILING_DATA_FIELD_DEF(uint32_t, aicNum);
TILING_DATA_FIELD_DEF(uint32_t, qaOffset);
TILING_DATA_FIELD_DEF(uint32_t, qaBytes);
TILING_DATA_FIELD_DEF(uint32_t, kvOffset);
TILING_DATA_FIELD_DEF(uint32_t, kvBytes);
TILING_DATA_FIELD_DEF(uint32_t, kvNormOffset);
TILING_DATA_FIELD_DEF(uint32_t, kvNormBytes);
TILING_DATA_FIELD_DEF(uint32_t, qInt8Offset);
TILING_DATA_FIELD_DEF(uint32_t, qInt8Bytes);
TILING_DATA_FIELD_DEF(uint32_t, qScaleOffset);
TILING_DATA_FIELD_DEF(uint32_t, qScaleBytes);
TILING_DATA_FIELD_DEF(uint32_t, wScaleFpOffset);
TILING_DATA_FIELD_DEF(uint32_t, wScaleFpBytes);
TILING_DATA_FIELD_DEF(uint32_t, qmmOffset);
TILING_DATA_FIELD_DEF(uint32_t, qmmBytes);
TILING_DATA_FIELD_DEF(uint32_t, totalBytes);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FiDsv41MlaProlog, Dsv41MlaPrologTilingData)
} // namespace optiling

#endif // DSV41_MLA_PROLOG_TILING_H
