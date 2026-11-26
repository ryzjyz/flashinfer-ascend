/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_FOUNDATION_SYNC_SUBBLOCK_BARRIER_HPP
#define FLASHINFER_SPARSE_INDEXER_FOUNDATION_SYNC_SUBBLOCK_BARRIER_HPP

#include "catlass/detail/macros.hpp"
#include "kernel_operator.h"

namespace SparseIndexer::Common {

/**
 * C2 uses explicit per-subblock cross-core flags. In MIX 1:2, a cube core
 * owns two AIVs; a single block flag would let one AIV consume the other's
 * hand-off. The 16-id offset is the AIV0/AIV1 pair convention used by the
 * arch35 QLI/LI kernels.
 */
struct SubblockPairBlockBarrierSyncPolicy {
    static constexpr uint32_t kSyncMode = 4;
    static constexpr uint16_t kVecToCube = 0;
    static constexpr uint16_t kCubeToVec = 2;
    static constexpr uint16_t kAivPairOffset = 16;

    CATLASS_DEVICE static void AicWaitCubeReady(uint32_t pingpong = 0)
    {
        AscendC::CrossCoreWaitFlag<kSyncMode, PIPE_FIX>(kVecToCube + pingpong);
        AscendC::CrossCoreWaitFlag<kSyncMode, PIPE_FIX>(kVecToCube + pingpong + kAivPairOffset);
    }

    CATLASS_DEVICE static void AicArriveCubeDone(uint32_t pingpong = 0)
    {
        AscendC::CrossCoreSetFlag<kSyncMode, PIPE_FIX>(kCubeToVec + pingpong);
        AscendC::CrossCoreSetFlag<kSyncMode, PIPE_FIX>(kCubeToVec + pingpong + kAivPairOffset);
    }

    CATLASS_DEVICE static void AicFinalWait(uint32_t pingpong = 0)
    {
        AscendC::CrossCoreWaitFlag<kSyncMode, PIPE_FIX>(kVecToCube + pingpong);
        AscendC::CrossCoreWaitFlag<kSyncMode, PIPE_FIX>(kVecToCube + pingpong + kAivPairOffset);
    }

    CATLASS_DEVICE static void AivArriveVecReady(uint32_t pingpong = 0)
    {
        AscendC::CrossCoreSetFlag<kSyncMode, PIPE_V>(kVecToCube + pingpong);
    }

    CATLASS_DEVICE static void AivWaitCubeDone(uint32_t pingpong = 0)
    {
        AscendC::CrossCoreWaitFlag<kSyncMode, PIPE_V>(kCubeToVec + pingpong);
    }

    CATLASS_DEVICE static void GlobalBarrier()
    {
        AscendC::SyncAll<false>();
    }
};

} // namespace SparseIndexer::Common

#endif // FLASHINFER_SPARSE_INDEXER_FOUNDATION_SYNC_SUBBLOCK_BARRIER_HPP
