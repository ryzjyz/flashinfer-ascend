// 版权归华为技术有限公司（2026）；依 CANN 开源软件许可协议 2.0 授权。

// 独立分页算子仅共享 Vector 原语，分开保护以允许随后包含完整公共实现。
#ifndef INDEXER_REGION_SCORE_COMMON_VECTOR_H
#define INDEXER_REGION_SCORE_COMMON_VECTOR_H
#include "kernel_operator.h"
#include "detail/head_reduce_vf.hpp"
namespace IndexerRegionScoreVector {
using namespace AscendC;

constexpr uint32_t H4_HEADS = 4U;
constexpr uint32_t H4_ROWS_PER_AIV = 16U;
constexpr uint32_t H4_VECTOR_LANES = 64U;
constexpr uint32_t H4_NEG_INF_BITS = 0xFF800000U;

template <uint32_t SEGMENTS, uint32_t WEIGHT_ROW_STRIDE>
__simd_callee__ inline void AccumulateH4RemainingHeads(
    __ubuf__ float *src0, __ubuf__ float *src1, __ubuf__ float *weights,
    uint16_t row, uint32_t headStride, MicroAPI::MaskReg &mask,
    MicroAPI::RegTensor<float> &in00, MicroAPI::RegTensor<float> &in01,
    MicroAPI::RegTensor<float> &in10, MicroAPI::RegTensor<float> &in11,
    MicroAPI::RegTensor<float> &wt0, MicroAPI::RegTensor<float> &wt1,
    MicroAPI::RegTensor<float> &acc00, MicroAPI::RegTensor<float> &acc01,
    MicroAPI::RegTensor<float> &acc10, MicroAPI::RegTensor<float> &acc11)
{
    for (uint16_t head = 1U; head < H4_HEADS; ++head) {
        MicroAPI::LoadAlign(in00, src0 + head * headStride);
        MicroAPI::LoadAlign(in10, src1 + head * headStride);
        if constexpr (SEGMENTS == 2U) {
            MicroAPI::LoadAlign(in01, src0 + head * headStride + H4_VECTOR_LANES);
            MicroAPI::LoadAlign(in11, src1 + head * headStride + H4_VECTOR_LANES);
        }
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(
            wt0, weights + row * WEIGHT_ROW_STRIDE + head);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(
            wt1, weights + (row + 1U) * WEIGHT_ROW_STRIDE + head);
        MicroAPI::MulAddDst(acc00, in00, wt0, mask);
        MicroAPI::MulAddDst(acc10, in10, wt1, mask);
        if constexpr (SEGMENTS == 2U) {
            MicroAPI::MulAddDst(acc01, in01, wt0, mask);
            MicroAPI::MulAddDst(acc11, in11, wt1, mask);
        }
    }
}

template <uint32_t SEGMENTS, uint32_t WEIGHT_ROW_STRIDE = H4_HEADS,
          uint32_t ROWS = H4_ROWS_PER_AIV>
__simd_vf__ inline void WeightedReduceH4RowsVFImpl(
    __ubuf__ float *dst, __ubuf__ float *src, __ubuf__ float *weights,
    uint32_t headStride, uint32_t dstRowStride)
{
    static_assert(SEGMENTS == 1U || SEGMENTS == 2U,
                  "regular H4 VF supports one or two 64-lane segments");
    static_assert(ROWS > 0U && ROWS <= H4_ROWS_PER_AIV && ROWS % 2U == 0U,
                  "regular H4 VF requires complete row pairs");
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<float> in00, in01, in10, in11;
    MicroAPI::RegTensor<float> wt0, wt1;
    MicroAPI::RegTensor<float> acc00, acc01, acc10, acc11;

    for (uint16_t row = 0U; row < ROWS; row += 2U) {
        __ubuf__ float *src0 = src + row * H4_HEADS * headStride;
        __ubuf__ float *src1 = src0 + H4_HEADS * headStride;
        MicroAPI::LoadAlign(in00, src0);
        MicroAPI::LoadAlign(in10, src1);
        if constexpr (SEGMENTS == 2U) {
            MicroAPI::LoadAlign(in01, src0 + H4_VECTOR_LANES);
            MicroAPI::LoadAlign(in11, src1 + H4_VECTOR_LANES);
        }
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(
            wt0, weights + row * WEIGHT_ROW_STRIDE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(
            wt1, weights + (row + 1U) * WEIGHT_ROW_STRIDE);
        MicroAPI::Mul(acc00, in00, wt0, mask);
        MicroAPI::Mul(acc10, in10, wt1, mask);
        if constexpr (SEGMENTS == 2U) {
            MicroAPI::Mul(acc01, in01, wt0, mask);
            MicroAPI::Mul(acc11, in11, wt1, mask);
        }

        AccumulateH4RemainingHeads<SEGMENTS, WEIGHT_ROW_STRIDE>(
            src0, src1, weights, row, headStride, mask,
            in00, in01, in10, in11, wt0, wt1, acc00, acc01, acc10, acc11);

        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
            dst + row * dstRowStride, acc00, mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
            dst + (row + 1U) * dstRowStride, acc10, mask);
        if constexpr (SEGMENTS == 2U) {
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
                dst + row * dstRowStride + H4_VECTOR_LANES, acc01, mask);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
                dst + (row + 1U) * dstRowStride + H4_VECTOR_LANES, acc11, mask);
        }
    }
}

template <uint32_t SEGMENTS, uint32_t WEIGHT_ROW_STRIDE = H4_HEADS,
          uint32_t ROWS = H4_ROWS_PER_AIV>
__aicore__ inline void WeightedReduceH4RowsVF(
    const LocalTensor<float> &dst, const LocalTensor<float> &src,
    const LocalTensor<float> &weights, uint32_t weightOffset,
    uint32_t headStride, uint32_t dstRowStride)
{
    auto dstPtr = reinterpret_cast<__ubuf__ float *>(dst.GetPhyAddr());
    auto srcPtr = reinterpret_cast<__ubuf__ float *>(src.GetPhyAddr());
    auto weightPtr = reinterpret_cast<__ubuf__ float *>(weights.GetPhyAddr());
    asc_vf_call<WeightedReduceH4RowsVFImpl<SEGMENTS, WEIGHT_ROW_STRIDE, ROWS>>(
        dstPtr, srcPtr, weightPtr + weightOffset, headStride, dstRowStride);
}

template <uint32_t SEGMENTS, uint32_t WEIGHT_ROW_STRIDE = H4_HEADS,
          uint32_t ROWS = H4_ROWS_PER_AIV>
__simd_vf__ inline void WeightedReduceH4MaskedRowsVFImpl(
    __ubuf__ float *dst, __ubuf__ float *src, __ubuf__ float *weights,
    uint32_t headStride, uint32_t dstRowStride, __ubuf__ int32_t *visible, int32_t tileBegin)
{
    static_assert(SEGMENTS == 1U || SEGMENTS == 2U,
                  "regular H4 VF supports one or two 64-lane segments");
    static_assert(ROWS > 0U && ROWS <= H4_ROWS_PER_AIV && ROWS % 2U == 0U,
                  "regular H4 VF requires complete row pairs");
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<float> in00, in01, in10, in11;
    MicroAPI::RegTensor<float> wt0, wt1;
    MicroAPI::RegTensor<float> acc00, acc01, acc10, acc11;
    MicroAPI::RegTensor<int32_t> columns0, columns1, visible0, visible1;
    MicroAPI::RegTensor<uint32_t> negativeInfinityBits;
    MicroAPI::Duplicate(negativeInfinityBits, H4_NEG_INF_BITS, mask);
    MicroAPI::Arange(columns0, tileBegin);
    MicroAPI::Arange(columns1, tileBegin + H4_VECTOR_LANES);
    MicroAPI::MaskReg keep00, keep01, keep10, keep11;

    for (uint16_t row = 0U; row < ROWS; row += 2U) {
        __ubuf__ float *src0 = src + row * H4_HEADS * headStride;
        __ubuf__ float *src1 = src0 + H4_HEADS * headStride;
        MicroAPI::LoadAlign(in00, src0);
        MicroAPI::LoadAlign(in10, src1);
        if constexpr (SEGMENTS == 2U) {
            MicroAPI::LoadAlign(in01, src0 + H4_VECTOR_LANES);
            MicroAPI::LoadAlign(in11, src1 + H4_VECTOR_LANES);
        }
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(
            wt0, weights + row * WEIGHT_ROW_STRIDE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(
            wt1, weights + (row + 1U) * WEIGHT_ROW_STRIDE);
        MicroAPI::Mul(acc00, in00, wt0, mask);
        MicroAPI::Mul(acc10, in10, wt1, mask);
        if constexpr (SEGMENTS == 2U) {
            MicroAPI::Mul(acc01, in01, wt0, mask);
            MicroAPI::Mul(acc11, in11, wt1, mask);
        }

        AccumulateH4RemainingHeads<SEGMENTS, WEIGHT_ROW_STRIDE>(
            src0, src1, weights, row, headStride, mask,
            in00, in01, in10, in11, wt0, wt1, acc00, acc01, acc10, acc11);

        MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_BRC_B32>(visible0, visible + row);
        MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_BRC_B32>(visible1, visible + row + 1U);
        MicroAPI::Compare<int32_t, CMPMODE::LT>(keep00, columns0, visible0, mask);
        MicroAPI::Compare<int32_t, CMPMODE::LT>(keep10, columns0, visible1, mask);
        MicroAPI::Select(acc00, acc00, reinterpret_cast<MicroAPI::RegTensor<float> &>(negativeInfinityBits), keep00);
        MicroAPI::Select(acc10, acc10, reinterpret_cast<MicroAPI::RegTensor<float> &>(negativeInfinityBits), keep10);
        if constexpr (SEGMENTS == 2U) {
            MicroAPI::Compare<int32_t, CMPMODE::LT>(keep01, columns1, visible0, mask);
            MicroAPI::Compare<int32_t, CMPMODE::LT>(keep11, columns1, visible1, mask);
            MicroAPI::Select(acc01, acc01, reinterpret_cast<MicroAPI::RegTensor<float> &>(negativeInfinityBits), keep01);
            MicroAPI::Select(acc11, acc11, reinterpret_cast<MicroAPI::RegTensor<float> &>(negativeInfinityBits), keep11);
        }

        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
            dst + row * dstRowStride, acc00, mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
            dst + (row + 1U) * dstRowStride, acc10, mask);
        if constexpr (SEGMENTS == 2U) {
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
                dst + row * dstRowStride + H4_VECTOR_LANES, acc01, mask);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
                dst + (row + 1U) * dstRowStride + H4_VECTOR_LANES, acc11, mask);
        }
    }
}

template <uint32_t SEGMENTS, uint32_t WEIGHT_ROW_STRIDE = H4_HEADS,
          uint32_t ROWS = H4_ROWS_PER_AIV>
__aicore__ inline void WeightedReduceH4MaskedRowsVF(
    const LocalTensor<float> &dst, const LocalTensor<float> &src,
    const LocalTensor<float> &weights, uint32_t weightOffset,
    uint32_t headStride, uint32_t dstRowStride,
    const LocalTensor<int32_t> &visible, int32_t tileBegin)
{
    auto dstPtr = reinterpret_cast<__ubuf__ float *>(dst.GetPhyAddr());
    auto srcPtr = reinterpret_cast<__ubuf__ float *>(src.GetPhyAddr());
    auto weightPtr = reinterpret_cast<__ubuf__ float *>(weights.GetPhyAddr());
    asc_vf_call<WeightedReduceH4MaskedRowsVFImpl<SEGMENTS, WEIGHT_ROW_STRIDE, ROWS>>(
        dstPtr, srcPtr, weightPtr + weightOffset, headStride, dstRowStride,
        reinterpret_cast<__ubuf__ int32_t *>(visible.GetPhyAddr()), tileBegin);
}

} // 命名空间 IndexerRegionScoreVector
#endif // 结束保护 INDEXER_REGION_SCORE_COMMON_VECTOR_H

#ifndef INDEXER_REGION_SCORE_COMMON_VECTOR_ONLY
#ifndef IRS_WEIGHTED_PIPELINE_H
#define IRS_WEIGHTED_PIPELINE_H
#include "../../include/arch35/tiling_data_def.h"
#include "contracts/common.hpp"
#include "sync/subblock_barrier.hpp"
#include "block/key_loader.hpp"
#include "detail/request.hpp"
namespace IndexerRegionScoreWeighted {
using namespace IndexerRegionScorePaged;
using namespace AscendC;

} // 命名空间 IndexerRegionScoreWeighted

namespace IndexerRegionScoreWeighted {
using namespace AscendC;

constexpr uint32_t PAGED_H = 4U;
constexpr uint32_t PAGED_D = 256U;
constexpr uint32_t PAGED_MQ_MAX = 32U;
constexpr uint32_t PAGED_K_MACRO = kPagedKeyMacro;
constexpr uint32_t PAGED_NR = 128U;
constexpr uint32_t PAGED_M_MAX = 128U;
constexpr uint32_t PAGED_Q_ROWS_MAX = 256U;
constexpr uint32_t PAGED_PIPE_SLOTS = 2U;
// 16-bit operands occupy the complete 64 KiB L0A/B. One permit protects the
// shared operand slot; FP8 preserves its original two operand slots.
constexpr uint32_t PAGED_OPERAND_SLOTS = kQkElementBytes == 1U ? 2U : 1U;
static_assert(kQkElementBytes == 1U || kQkElementBytes == 2U, "unsupported Q/K element width");
static_assert(kPagedKeyL1Slots * kPagedKeyL1Bytes +
    PAGED_PIPE_SLOTS * PAGED_Q_ROWS_MAX * PAGED_D * kQkElementBytes <= 512U * 1024U,
    "Q/K resident buffers exceed A5 L1 capacity");
static_assert(PAGED_OPERAND_SLOTS * PAGED_M_MAX * PAGED_D * kQkElementBytes <= 64U * 1024U,
    "Q operand exceeds A5 L0A capacity");
static_assert(PAGED_OPERAND_SLOTS * PAGED_NR * PAGED_D * kQkElementBytes <= 64U * 1024U,
    "K operand exceeds A5 L0B capacity");
constexpr uint32_t PAGED_QUERIES_PER_AIV_MAX = PAGED_MQ_MAX / 2U;
constexpr uint32_t PAGED_RESULT_SLOT_ELEMENTS = PAGED_M_MAX * PAGED_NR;
constexpr uint32_t PAGED_OUTPUT_SLOT_ELEMENTS = PAGED_QUERIES_PER_AIV_MAX * PAGED_NR;
// 保留 144 元素 UB 预算，单个 AIV 的组融合权重最多需要 128 元素。
constexpr uint32_t PAGED_WEIGHT_FP32_ELEMENTS = 144U;
constexpr uint32_t PAGED_H4_WEIGHT_ROW_STRIDE = 32U / sizeof(float);
using RegionScoreSyncPolicy = SparseIndexer::Common::SubblockPairBlockBarrierSyncPolicy;
constexpr uint32_t PAGED_SYNC_MODE = RegionScoreSyncPolicy::kSyncMode;
constexpr uint32_t PAGED_C_FREE_EVENT = RegionScoreSyncPolicy::kVecToCube;
constexpr uint32_t PAGED_C_READY_EVENT = RegionScoreSyncPolicy::kCubeToVec;
constexpr uint32_t PAGED_WORK_DRAINED_EVENT = 10U;
constexpr uint32_t PAGED_AIV1_EVENT_OFFSET = RegionScoreSyncPolicy::kAivPairOffset;
constexpr FixpipeConfig PAGED_FIX_TO_UB_ROW_MAJOR = {CO2Layout::ROW_MAJOR, true};
__aicore__ inline uint32_t PagedMin(uint32_t left, uint32_t right)
{
    return left < right ? left : right;
}

__aicore__ inline uint32_t PagedMax(uint32_t left, uint32_t right)
{
    return left > right ? left : right;
}

__aicore__ inline uint32_t PagedAlignUp(uint32_t value, uint32_t alignment)
{
    return SparseIndexer::Common::AlignUp(value, alignment);
}

__aicore__ inline PagedTilingData NormalizeContinuousTiling(
    const IndexerRegionScore::RegionScoreTilingData &tilingData)
{
    PagedTilingData sharedTiling{};
    sharedTiling.totalTokens = tilingData.seqQ;
    sharedTiling.groups = tilingData.groups;
    sharedTiling.headsPerGroup = tilingData.headNum;
    sharedTiling.proxyDim = tilingData.headDim;
    sharedTiling.numRequests = tilingData.numRequests;
    sharedTiling.pageCols = 0U;
    sharedTiling.summariesPerPage = 1U;
    sharedTiling.seqK = static_cast<uint32_t>(tilingData.seqK);
    uint32_t groupTile = tilingData.groups <
            PAGED_M_MAX / tilingData.headNum
        ? tilingData.groups
        : PAGED_M_MAX / tilingData.headNum;
    for (uint32_t count = groupTile; count > 0U; --count) {
        if ((count * tilingData.headNum) % 16U == 0U &&
            tilingData.groups % count == 0U) {
            groupTile = count;
            break;
        }
    }
    sharedTiling.qTile = PAGED_M_MAX /
        (groupTile * tilingData.headNum);
    sharedTiling.kTile = PAGED_K_MACRO;
    sharedTiling.usedCubeCores = tilingData.usedCubeCores;
    sharedTiling.maskMode = tilingData.maskMode;
    sharedTiling.headScale = tilingData.headScale;
    sharedTiling.regionSize = tilingData.regionSize;
    sharedTiling.selectionConfig = tilingData.selectionConfig;
    sharedTiling.keyPagesPerMacro = PAGED_K_MACRO;
    sharedTiling.qShardCount = tilingData.qShardCount;
    sharedTiling.kShardCount = tilingData.kShardCount;
    sharedTiling.numPhysicalPages = 0U;
    // 连续编码为 0=BF16、1=FP16、2=FP32；共享编码为 0=FP32、1=BF16、2=FP16。
    sharedTiling.weightDtype = tilingData.weightDtype == 2U
        ? 0U : tilingData.weightDtype + 1U;
    sharedTiling.qStrideToken = tilingData.qStrideT;
    sharedTiling.qStrideGroup = tilingData.qStrideG;
    sharedTiling.qStrideHead = tilingData.headDim;
    sharedTiling.weightStrideToken = tilingData.weightStrideT;
    sharedTiling.weightStrideGroup = tilingData.weightStrideG;
    sharedTiling.cacheStridePage = 0U;
    sharedTiling.cacheStrideRegion = tilingData.headDim;
    sharedTiling.blockTableStrideRequest = 0U;
    sharedTiling.outputStrideGroup = tilingData.outStrideG;
    sharedTiling.outputStrideToken = tilingData.outStrideT;
    sharedTiling.batchTokens = tilingData.batchTokens;
    sharedTiling.batchBegin = tilingData.batchBegin;
    sharedTiling.batchCount = tilingData.batchCount;
    sharedTiling.scorePoolOffset = tilingData.scorePoolOffset;
    return sharedTiling;
}

} // namespace IndexerRegionScoreWeighted

#include "weighted/block_scheduler.hpp"
#include "weighted/score_plan_prescreen.hpp"
#include "block/block_mmad.hpp"
#include "block/block_reduce_head.hpp"

namespace IndexerRegionScoreWeighted {

template <typename KeyAdapter>
struct CausalH4BatchPolicy {
    static constexpr bool ENABLED = false;
};

template <bool PARTIAL_PAGES>
struct CausalH4BatchPolicy<ScalarPagedKeyResidentBringup<PARTIAL_PAGES>> {
    static constexpr bool ENABLED = kQkElementBytes == 1U;
};

__aicore__ inline bool UseScoreGroupFusion(const PagedTilingData &tiling)
{
    return tiling.groups != 0U && (tiling.groups % 4U) == 0U &&
           tiling.headsPerGroup >= 4U && tiling.headsPerGroup <= PAGED_M_MAX &&
           (tiling.headsPerGroup % 4U) == 0U;
}

__aicore__ inline uint32_t ScoreGroupTile(const PagedTilingData &tiling)
{
    const uint32_t heads = tiling.headsPerGroup;
    const uint32_t capacity = PagedMin(tiling.groups, PAGED_M_MAX / heads);
    for (uint32_t count = capacity; count > 0U; --count) {
        if ((count * heads) % 16U == 0U && tiling.groups % count == 0U) {
            return count;
        }
    }
    return capacity;
}

__aicore__ inline uint32_t ScoreQueryBand(const PagedTilingData &tiling)
{
    const uint32_t groups = UseScoreGroupFusion(tiling) ? ScoreGroupTile(tiling) : 1U;
    return (tiling.qTile * groups * tiling.headsPerGroup) % 16U == 0U
        ? 2U * tiling.qTile : tiling.qTile;
}

template <typename KeyAdapter, bool QUERY_RESIDENT, bool ALLOW_H4_TAIL = false>
class KeyResidentCompute final {
    using ComputePolicy = IndexerRegionScore::Block::ComputeStagePolicy<KeyAdapter, QUERY_RESIDENT>;
    using MmadBlock = IndexerRegionScore::Block::StageType<IndexerRegionScore::Block::Stage::Mmad, ComputePolicy>;
    using ReduceHeadBlock = IndexerRegionScore::Block::StageType<
        IndexerRegionScore::Block::Stage::ReduceHead, ComputePolicy>;

    friend struct NoMaskStrategy;
    friend struct RightDownCausalMaskStrategy;
    template <bool, bool>
    friend struct RegionScoreRequestPolicy;
public:
    __aicore__ inline void Init(GM_ADDR indexQ, GM_ADDR weights,
                                GM_ADDR actualKvLengths, GM_ADDR output,
                                const PagedTilingData *tiling, TPipe *pipe,
                                KeyAdapter *keyAdapter)
    {
        tiling_ = *tiling;
        pipe_ = pipe;
        keyAdapter_ = keyAdapter;
        indexQGm_.SetGlobalBuffer(reinterpret_cast<__gm__ QkInputType *>(indexQ));
        weightsFp32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weights));
        weightsBf16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(weights));
        weightsFp16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(weights));
        if (tiling_.maskMode == 3U) {
            actualKvLengthsGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t *>(actualKvLengths));
        }
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));

        if ASCEND_IS_AIC {
            InitCubeBuffers();
        } else {
            InitVectorBuffers();
        }
    }

    template <bool CAUSAL>
    __aicore__ inline void ConfigureDeviceSchedule(GM_ADDR cu, GM_ADDR kv, GM_ADDR plan)
    {
        scheduleCu_ = cu;
        balanceQBand_ = UseGroupFused() ? GroupFusedQMacroTile() : QMacroTile();
        LocalTensor<int32_t> metadata;
        if ASCEND_IS_AIV {
            metadata = acc_.template ReinterpretCast<int32_t>();
        }
        scorePartition_ = PrepareScorePartition<CAUSAL>(
            tiling_, cu, kv, plan, balanceQBand_, PairCore(), metadata);
    }

    __aicore__ inline void Release()
    {
        if ASCEND_IS_AIC {
            ReleaseCubeEvents();
        } else {
            DrainOutputStores();
            pipe_->ReleaseEventID<HardEvent::MTE2_V>(mte2ToVector_);
            pipe_->ReleaseEventID<HardEvent::V_S>(vectorToScalar_);
            pipe_->ReleaseEventID<HardEvent::S_V>(scalarToVector_);
            pipe_->ReleaseEventID<HardEvent::V_MTE2>(vectorToMte2_);
            pipe_->ReleaseEventID<HardEvent::V_MTE3>(vectorToMte3_);
            for (uint32_t slot = 0U; slot < PAGED_PIPE_SLOTS; ++slot) {
                pipe_->ReleaseEventID<HardEvent::MTE3_V>(outputStoreDone_[slot]);
            }
        }
    }

private:
    __aicore__ inline void InitCubeBuffers()
    {
        SetMMRowMajor();
        pipe_->InitBuffer(qL1Buf_, PAGED_PIPE_SLOTS * PAGED_Q_ROWS_MAX * PAGED_D * kQkElementBytes);
        pipe_->InitBuffer(qL0Buf_, PAGED_OPERAND_SLOTS * PAGED_M_MAX * PAGED_D * kQkElementBytes);
        pipe_->InitBuffer(kL0Buf_, PAGED_OPERAND_SLOTS * PAGED_NR * PAGED_D * kQkElementBytes);
        pipe_->InitBuffer(cL0Buf_,
                          PAGED_PIPE_SLOTS * PAGED_M_MAX * PAGED_NR * sizeof(float));
        for (uint32_t slot = 0U; slot < PAGED_PIPE_SLOTS; ++slot) {
            pipe_->InitBuffer(mmOutBuf_[slot],
                              PAGED_RESULT_SLOT_ELEMENTS * sizeof(float));
            mmOut_[slot] = mmOutBuf_[slot].Get<float>();
        }

        LocalTensor<QkInputType> qL1Pool = qL1Buf_.Get<QkInputType>();
        for (uint32_t slot = 0U; slot < PAGED_PIPE_SLOTS; ++slot) {
            qL1_[slot] = qL1Pool[slot * PAGED_Q_ROWS_MAX * PAGED_D];
            qReady_[slot] = pipe_->AllocEventID<HardEvent::MTE2_MTE1>();
            qFree_[slot] = pipe_->AllocEventID<HardEvent::MTE1_MTE2>();
            if (slot < PAGED_OPERAND_SLOTS) {
                mToMte1_[slot] = pipe_->AllocEventID<HardEvent::M_MTE1>();
                SetFlag<HardEvent::M_MTE1>(mToMte1_[slot]);
            }
            fixToM_[slot] = pipe_->AllocEventID<HardEvent::FIX_M>();
            SetFlag<HardEvent::MTE1_MTE2>(qFree_[slot]);
            SetFlag<HardEvent::FIX_M>(fixToM_[slot]);
        }
        qL0_ = qL0Buf_.Get<QkInputType>();
        kL0_ = kL0Buf_.Get<QkInputType>();
        cL0_ = cL0Buf_.Get<float>();
        mte1ToM_ = pipe_->AllocEventID<HardEvent::MTE1_M>();
        mToFix_ = pipe_->AllocEventID<HardEvent::M_FIX>();
    }

    __aicore__ inline void InitVectorBuffers()
    {
        for (uint32_t slot = 0U; slot < PAGED_PIPE_SLOTS; ++slot) {
            pipe_->InitBuffer(mmOutBuf_[slot],
                              PAGED_RESULT_SLOT_ELEMENTS * sizeof(float));
            mmOut_[slot] = mmOutBuf_[slot].Get<float>();
        }
        pipe_->InitBuffer(accBuf_,
                          PAGED_PIPE_SLOTS * PAGED_OUTPUT_SLOT_ELEMENTS * sizeof(float));
        // 跨步 GM 行分别搬入 32 字节对齐的 UB 行。
        pipe_->InitBuffer(weightUbBuf_, PAGED_WEIGHT_FP32_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(weightInputUbBuf_, PAGED_M_MAX * sizeof(float));
        pipe_->InitBuffer(cuSeqLensUbBuf_, 2U * sizeof(int32_t));
        pipe_->InitBuffer(visibleUbBuf_, PAGED_QUERIES_PER_AIV_MAX * sizeof(int32_t));
        acc_ = accBuf_.Get<float>();
        weightUb_ = weightUbBuf_.Get<float>();
        weightInputBf16Ub_ = weightInputUbBuf_.Get<bfloat16_t>();
        weightInputFp16Ub_ = weightInputUbBuf_.Get<half>();
        cuSeqLensUb_ = cuSeqLensUbBuf_.Get<int32_t>();
        visibleUb_ = visibleUbBuf_.Get<int32_t>();
        mte2ToVector_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
        vectorToScalar_ = pipe_->AllocEventID<HardEvent::V_S>();
        scalarToVector_ = pipe_->AllocEventID<HardEvent::S_V>();
        vectorToMte2_ = pipe_->AllocEventID<HardEvent::V_MTE2>();
        vectorToMte3_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
        for (uint32_t slot = 0U; slot < PAGED_PIPE_SLOTS; ++slot) {
            outputStoreDone_[slot] = pipe_->AllocEventID<HardEvent::MTE3_V>();
        }
    }

    __aicore__ inline LocalTensor<float> MmOut(uint32_t flatOffset)
    {
        const uint32_t slot = flatOffset / PAGED_RESULT_SLOT_ELEMENTS;
        return mmOut_[slot][flatOffset - slot * PAGED_RESULT_SLOT_ELEMENTS];
    }

    __aicore__ inline void ReleaseCubeEvents()
    {
        for (uint32_t slot = 0U; slot < PAGED_PIPE_SLOTS; ++slot) {
            WaitFlag<HardEvent::MTE1_MTE2>(qFree_[slot]);
            if (slot < PAGED_OPERAND_SLOTS) {
                WaitFlag<HardEvent::M_MTE1>(mToMte1_[slot]);
                pipe_->ReleaseEventID<HardEvent::M_MTE1>(mToMte1_[slot]);
            }
            WaitFlag<HardEvent::FIX_M>(fixToM_[slot]);
            pipe_->ReleaseEventID<HardEvent::MTE2_MTE1>(qReady_[slot]);
            pipe_->ReleaseEventID<HardEvent::MTE1_MTE2>(qFree_[slot]);
            pipe_->ReleaseEventID<HardEvent::FIX_M>(fixToM_[slot]);
        }
        pipe_->ReleaseEventID<HardEvent::MTE1_M>(mte1ToM_);
        pipe_->ReleaseEventID<HardEvent::M_FIX>(mToFix_);
        SetMMColumnMajor();
    }

    __aicore__ inline uint32_t PairCore() const
    {
        uint32_t pair = GetBlockIdx();
        if ASCEND_IS_AIV {
            pair /= GetTaskRation();
        }
        return pair;
    }

    __aicore__ inline uint32_t Heads() const
    {
        return tiling_.headsPerGroup;
    }

    __aicore__ inline bool UseGroupFused() const
    {
        return tiling_.groups != 0U && (tiling_.groups % 4U) == 0U &&
               tiling_.headsPerGroup >= 4U &&
               tiling_.headsPerGroup <= PAGED_M_MAX &&
               (tiling_.headsPerGroup % 4U) == 0U;
    }

    __aicore__ inline uint32_t GroupTile() const
    {
        // 与 Host 切分及连续缓存适配器的 Q tile 保持一致。
        const uint32_t capacity = PagedMin(tiling_.groups, PAGED_M_MAX / Heads());
        for (uint32_t count = capacity; count > 0U; --count) {
            if ((count * Heads()) % 16U == 0U && tiling_.groups % count == 0U) {
                return count;
            }
        }
        return capacity;
    }

    __aicore__ inline uint32_t GroupFusedQMicroTile() const
    {
        return tiling_.qTile;
    }

    __aicore__ inline uint32_t GroupFusedQMacroTile() const
    {
        // LoadData 按 16 行单位定位第二个微块。
        return (GroupFusedQMicroTile() * GroupTile() * Heads()) % 16U == 0U
                   ? 2U * GroupFusedQMicroTile() : GroupFusedQMicroTile();
    }

    __aicore__ inline uint32_t WeightFp32RowStride() const
    {
        return PagedAlignUp(Heads() * sizeof(float), 32U) / sizeof(float);
    }

    __aicore__ inline uint32_t WeightNativeRowStride() const
    {
        return PagedAlignUp(Heads() * sizeof(half), 32U) / sizeof(half);
    }

    __aicore__ inline uint32_t QMicroTile() const
    {
        return tiling_.qTile;
    }

    __aicore__ inline uint32_t QMacroTile() const
    {
        return (QMicroTile() * Heads()) % 16U == 0U
                   ? 2U * QMicroTile()
                   : QMicroTile();
    }

    __aicore__ inline void GetLocalQueryRange(uint32_t qCount, uint32_t subBlock,
                                               uint32_t &localQStart,
                                               uint32_t &localQCount) const
    {
        const uint32_t heads = Heads();
        const uint32_t queryAlignment = heads % 16U == 0U ? 1U :
                                        heads % 8U == 0U ? 2U : 4U;
        const uint32_t capacity = (PAGED_M_MAX / 2U) / heads;
        const uint32_t alignedCapacity = capacity / queryAlignment * queryAlignment;
        const uint32_t firstCount = qCount * heads <= PAGED_M_MAX / 2U || alignedCapacity == 0U
                                        ? qCount
                                        : PagedMin(qCount, alignedCapacity);
        localQStart = subBlock == 0U ? 0U : firstCount;
        localQCount = subBlock == 0U ? firstCount : qCount - firstCount;
    }

    template <bool CAUSAL>
    using RequestScheduler = SparseIndexer::Common::Block::BlockScheduler<RegionScoreRequestPolicy<CAUSAL, QUERY_RESIDENT>>;
    using KeyMacroScheduler = SparseIndexer::Common::Block::BlockScheduler<RegionScoreKeyMacroPolicy>;

    template <bool CAUSAL>
    __aicore__ inline RequestScheduler<CAUSAL> MakeRequestScheduler() const
    {
        using Policy = RegionScoreRequestPolicy<CAUSAL, QUERY_RESIDENT>;
        auto state = Policy::Init(tiling_, PairCore());
        if (scorePartition_.weighted) {
            Policy::UseWeightedRange(state, scorePartition_);
        } else if (scheduleCu_ != nullptr) {
            Policy::RestrictRequests(state, scheduleCu_);
        }
        return RequestScheduler<CAUSAL>(state);
    }

    __aicore__ inline void GatherQueryToL1(uint32_t slot, uint32_t group,
                                           uint32_t qStart, uint32_t qCount,
                                           uint32_t qRowsAlign)
    {
        WaitFlag<HardEvent::MTE1_MTE2>(qFree_[slot]);
        // 将 Q 的每个连续 [H,D] 切片视为 ND 矩阵，组合为与连续缓存相同的 NZ 矩阵。
        const uint64_t sourceOffset = static_cast<uint64_t>(qStart) * tiling_.qStrideToken +
                                      static_cast<uint64_t>(group) * tiling_.qStrideGroup;
        BlockKeyLoader<QkInputType>::CopyNdMatrices(
            qL1_[slot], indexQGm_[sourceOffset], Heads(), QkInputDim(tiling_), qRowsAlign,
            qCount, tiling_.qStrideToken, Heads() * kPagedInputC0);
        SetFlag<HardEvent::MTE2_MTE1>(qReady_[slot]);
    }

    __aicore__ inline void GatherGroupFusedQueryToL1(
        uint32_t slot, uint32_t groupStart, uint32_t groupCount,
        uint32_t qStart, uint32_t qCount)
    {
        WaitFlag<HardEvent::MTE1_MTE2>(qFree_[slot]);
        const uint32_t rowsPerToken = groupCount * Heads();
        const uint32_t qRowsAlign =
            PagedAlignUp(qCount * rowsPerToken, kPagedNAlignment);
        const uint64_t sourceOffset =
            static_cast<uint64_t>(qStart) * tiling_.qStrideToken +
            static_cast<uint64_t>(groupStart) * tiling_.qStrideGroup;
        BlockKeyLoader<QkInputType>::CopyNdMatrices(
            qL1_[slot], indexQGm_[sourceOffset], rowsPerToken, QkInputDim(tiling_), qRowsAlign,
            qCount, tiling_.qStrideToken, rowsPerToken * kPagedInputC0);
        SetFlag<HardEvent::MTE2_MTE1>(qReady_[slot]);
    }

    __aicore__ inline void GatherGroupFusedQueryMacro(
        uint32_t macro, uint32_t groupStart, uint32_t groupCount,
        uint32_t qBegin, uint32_t qEnd)
    {
        const uint32_t qMacroTile = GroupFusedQMacroTile();
        const uint32_t qStart = qBegin + macro * qMacroTile;
        const uint32_t qCount =
            PagedMin(qMacroTile, qEnd - qStart);
        const uint32_t slot = macro % PAGED_PIPE_SLOTS;
        GatherGroupFusedQueryToL1(
            slot, groupStart, groupCount, qStart, qCount);
    }

    __aicore__ inline void GatherFlatQueryMacro(uint64_t flatQ,
                                                uint32_t queryMacrosPerGroup,
                                                uint32_t qBegin, uint32_t qEnd)
    {
        const uint32_t group = static_cast<uint32_t>(flatQ / queryMacrosPerGroup);
        const uint32_t queryInGroup = static_cast<uint32_t>(flatQ % queryMacrosPerGroup);
        const uint32_t qStart = qBegin + queryInGroup * QMacroTile();
        const uint32_t qCount = PagedMin(QMacroTile(), qEnd - qStart);
        const uint32_t qRowsAlign = PagedAlignUp(qCount * Heads(), 16U);
        const uint32_t slot = static_cast<uint32_t>(flatQ % PAGED_PIPE_SLOTS);

        GatherQueryToL1(slot, group, qStart, qCount, qRowsAlign);
    }

    __aicore__ inline void LoadOperandsToL0(LocalTensor<QkInputType> keyL1,
                                             uint32_t qRowsAlign, uint32_t kRowsAlign,
                                             uint32_t qRowStart, uint32_t kRowStart,
                                             uint32_t mAlign, uint32_t nAlign,
                                             uint32_t resultSlot, uint32_t querySlot)
    {
        WaitFlag<HardEvent::M_MTE1>(mToMte1_[resultSlot % PAGED_OPERAND_SLOTS]);

        LoadData2DParamsV2 qParams;
        qParams.mStartPosition = qRowStart / 16U;
        qParams.kStartPosition = 0U;
        qParams.mStep = mAlign / 16U;
        qParams.kStep = QkInputDim(tiling_) / kPagedInputC0;
        qParams.srcStride = qRowsAlign / 16U;
        qParams.dstStride = mAlign / 16U;
        qParams.ifTranspose = false;
        LoadData(qL0_[(resultSlot % PAGED_OPERAND_SLOTS) * PAGED_M_MAX * PAGED_D], qL1_[querySlot], qParams);

        LoadData2DParamsV2 kParams;
        kParams.mStartPosition = kRowStart / 16U;
        kParams.kStartPosition = 0U;
        kParams.mStep = nAlign / 16U;
        kParams.kStep = QkInputDim(tiling_) / kPagedInputC0;
        kParams.srcStride = kRowsAlign / 16U;
        kParams.dstStride = nAlign / 16U;
        kParams.ifTranspose = false;
        LoadData(kL0_[(resultSlot % PAGED_OPERAND_SLOTS) * PAGED_NR * PAGED_D], keyL1, kParams);

        SetFlag<HardEvent::MTE1_M>(mte1ToM_);
        WaitFlag<HardEvent::MTE1_M>(mte1ToM_);
    }

    __aicore__ inline void RunTileMmad(
        uint32_t mAlign, uint32_t nAlign, uint32_t resultSlot)
    {
        MmadBlock::RunTileMmad(tiling_, cL0_, qL0_, kL0_, fixToM_[resultSlot],
            mToMte1_[resultSlot % PAGED_OPERAND_SLOTS], mToFix_, mAlign, nAlign, resultSlot);
    }

    __aicore__ inline void PublishCubeResult(
        uint32_t mAlign, uint32_t nAlign, uint32_t qCount, uint32_t kCount,
        uint32_t resultSlot)
    {
        MmadBlock::PublishCubeResult(Heads(), cL0_,
            MmOut(resultSlot * PAGED_RESULT_SLOT_ELEMENTS), fixToM_[resultSlot],
            mAlign, nAlign, qCount, kCount, resultSlot);
    }

    __aicore__ inline void ComputeOneTile(LocalTensor<QkInputType> keyL1,
                                           uint32_t qRowsAlign, uint32_t kRowsAlign,
                                           uint32_t qRowStart, uint32_t kRowStart,
                                           uint32_t qCount, uint32_t kCount,
                                           uint32_t resultSlot, uint32_t querySlot,
                                           bool fullyMasked)
    {
        // 全屏蔽 tile 仍保持双 AIV 的空闲/就绪协议，AIV 仅写负无穷。
        if (fullyMasked) {
            RegionScoreSyncPolicy::AicWaitCubeReady(resultSlot);
            RegionScoreSyncPolicy::AicArriveCubeDone(resultSlot);
            return;
        }
        const uint32_t mAlign = PagedAlignUp(qCount * Heads(), 16U);
        const uint32_t nAlign = PagedAlignUp(kCount, 16U);
        LoadOperandsToL0(keyL1, qRowsAlign, kRowsAlign, qRowStart, kRowStart,
                         mAlign, nAlign, resultSlot, querySlot);

        RunTileMmad(mAlign, nAlign, resultSlot);

        PublishCubeResult(mAlign, nAlign, qCount, kCount, resultSlot);
    }

    __aicore__ inline int32_t VisibleRegions(uint32_t request,
                                              uint32_t requestBegin,
                                              uint32_t token) const
    {
        const int64_t position = requestFirstPosition_ +
            static_cast<int64_t>(token - requestBegin);
        if (position <= 0) {
            return 0;
        }
        const uint64_t completed = static_cast<uint64_t>(position) / tiling_.regionSize;
        return static_cast<int32_t>(
            completed > tiling_.seqK ? tiling_.seqK : completed);
    }

    __aicore__ inline int32_t GetMaxVisibleLength(uint32_t request,
                                                   uint32_t requestBegin,
                                                   uint32_t qStart,
                                                   uint32_t qCount) const
    {
        return VisibleRegions(request, requestBegin, qStart + qCount - 1U);
    }

    __aicore__ inline void PrepareNextKeyMacro(
        uint32_t request, uint32_t kStart, uint32_t coreKEnd,
        uint32_t keyMacroIndex, bool &nextKeyPrepared)
    {
        if (!nextKeyPrepared && kStart + PAGED_K_MACRO < coreKEnd) {
            const uint32_t nextKStart = kStart + PAGED_K_MACRO;
            const uint32_t nextKCount =
                PagedMin(PAGED_K_MACRO, coreKEnd - nextKStart);
            const uint32_t nextKeySlot =
                (keyMacroIndex + 1U) % kPagedKeyL1Slots;
            keyAdapter_->PrepareKeyMacroToL1(
                request, nextKStart, nextKCount, nextKeySlot,
                PagedAlignUp(nextKCount, kPagedNAlignment));
            nextKeyPrepared = true;
        }
    }

    template <bool CAUSAL>
    __aicore__ inline void ComputeCubeQueryMicro(
        LocalTensor<QkInputType> keyL1, uint32_t qRowsAlign, uint32_t kRowsAlign,
        uint32_t qMicro, uint32_t qMicroCount, uint32_t request,
        uint32_t kStart, uint32_t kMacroCount, uint32_t querySlot,
        int32_t maxVisible, uint32_t &streamLoop, bool &nextKeyPrepared,
        uint32_t coreKEnd, uint32_t keyMacroIndex)
    {
        for (uint32_t kMicro = 0U; kMicro < kMacroCount;
             kMicro += PAGED_NR) {
            const uint32_t kMicroCount =
                PagedMin(PAGED_NR, kMacroCount - kMicro);
            const uint32_t resultSlot = streamLoop % PAGED_PIPE_SLOTS;
            bool fullyMasked = false;
            if constexpr (CAUSAL) {
                fullyMasked = qMicroCount == QMicroTile() &&
                    kMicroCount == PAGED_NR &&
                    static_cast<int32_t>(kStart + kMicro) >= maxVisible;
            }
            ComputeOneTile(keyL1, qRowsAlign, kRowsAlign,
                           qMicro * Heads(), kMicro,
                           qMicroCount, kMicroCount,
                           resultSlot, querySlot, fullyMasked);
            ++streamLoop;

            PrepareNextKeyMacro(request, kStart, coreKEnd, keyMacroIndex, nextKeyPrepared);
        }
    }

    template <bool CAUSAL>
    __aicore__ inline void ComputeCubeQueryMacro(
        LocalTensor<QkInputType> keyL1, uint32_t qRowsAlign, uint32_t kRowsAlign,
        uint32_t qMacroCount, uint32_t request, uint32_t requestQBegin, uint32_t group,
        uint32_t qStart, uint32_t kStart, uint32_t kMacroCount, uint32_t querySlot,
        uint32_t &streamLoop, bool &nextKeyPrepared, uint32_t coreKEnd, uint32_t keyMacroIndex)
    {
        for (uint32_t qMicro = 0U; qMicro < qMacroCount; qMicro += QMicroTile()) {
            const uint32_t qMicroCount =
                PagedMin(QMicroTile(), qMacroCount - qMicro);
            int32_t maxVisible = 0;
            if constexpr (CAUSAL) {
                if (qMicroCount == QMicroTile()) {
                    maxVisible = GetMaxVisibleLength(
                        request, requestQBegin,
                        qStart + qMicro, qMicroCount);
                }
            }
            ComputeCubeQueryMicro<CAUSAL>(keyL1, qRowsAlign, kRowsAlign, qMicro, qMicroCount,
                                request, kStart, kMacroCount, querySlot,
                                maxVisible, streamLoop, nextKeyPrepared, coreKEnd, keyMacroIndex);
        }
    }

    template <bool CAUSAL>
    __aicore__ inline void ConsumeCubeKeyMacro(
        LocalTensor<QkInputType> keyL1, uint32_t kRowsAlign, uint64_t flatQueryMacros,
        uint32_t queryMacrosPerGroup, uint32_t qBegin, uint32_t qEnd, uint32_t request,
        uint32_t requestQBegin, uint32_t kStart, uint32_t kMacroCount, uint32_t coreKEnd,
        uint32_t keyMacroIndex, uint32_t &streamLoop, bool &nextKeyPrepared, bool residentQueries)
    {
        for (uint64_t flatQ = 0U; flatQ < flatQueryMacros; ++flatQ) {
            const uint32_t group =
                static_cast<uint32_t>(flatQ / queryMacrosPerGroup);
            const uint32_t queryInGroup =
                static_cast<uint32_t>(flatQ % queryMacrosPerGroup);
            const uint32_t qStart = qBegin + queryInGroup * QMacroTile();
            const uint32_t qMacroCount = PagedMin(QMacroTile(), qEnd - qStart);
            const uint32_t querySlot =
                static_cast<uint32_t>(flatQ % PAGED_PIPE_SLOTS);
            const uint32_t qRowsAlign = PagedAlignUp(qMacroCount * Heads(), 16U);
            if (!residentQueries) {
                WaitFlag<HardEvent::MTE2_MTE1>(qReady_[querySlot]);
            }

            if (!residentQueries && flatQ + 1U < flatQueryMacros) {
                GatherFlatQueryMacro(flatQ + 1U, queryMacrosPerGroup, qBegin, qEnd);
            }
            ComputeCubeQueryMacro<CAUSAL>(keyL1, qRowsAlign, kRowsAlign, qMacroCount,
                                request, requestQBegin, group, qStart, kStart, kMacroCount,
                                querySlot, streamLoop, nextKeyPrepared, coreKEnd, keyMacroIndex);
            // 当前宏块全部 LoadData 完成后，生产者才可复用 Q 槽。
            if (!residentQueries) {
                SetFlag<HardEvent::MTE1_MTE2>(qFree_[querySlot]);
            }
        }
    }

    template <bool CAUSAL>
    __aicore__ inline void ProcessCubeRequestKeys(
        uint32_t request, uint32_t requestQBegin, uint32_t qBegin, uint32_t qEnd,
        uint32_t queryMacrosPerGroup, uint64_t flatQueryMacros,
        uint32_t coreKBegin, uint32_t coreKEnd, uint32_t firstKeyMacroIndex, uint32_t &streamLoop)
    {
        const bool residentQueries = QUERY_RESIDENT && flatQueryMacros <= PAGED_PIPE_SLOTS && coreKBegin < coreKEnd;
        if (residentQueries) {
            for (uint64_t flatQ = 0U; flatQ < flatQueryMacros; ++flatQ) {
                GatherFlatQueryMacro(flatQ, queryMacrosPerGroup, qBegin, qEnd);
                WaitFlag<HardEvent::MTE2_MTE1>(qReady_[static_cast<uint32_t>(flatQ)]);
            }
        }
        KeyMacroScheduler scheduler({coreKBegin, coreKEnd, firstKeyMacroIndex});
        KeyMacroScheduler::Coord macro;
        while (scheduler.Next(macro)) {
            const auto [kStart, kMacroCount, keyMacroIndex] = macro;
            const uint32_t keySlot = keyMacroIndex % kPagedKeyL1Slots;
            const uint32_t kRowsAlign = PagedAlignUp(kMacroCount, kPagedNAlignment);
            keyAdapter_->WaitKeyMacroReady(keySlot);

            LocalTensor<QkInputType> keyL1 = keyAdapter_->GetKeyL1(keySlot);

            if (!residentQueries) {
                GatherFlatQueryMacro(0U, queryMacrosPerGroup, qBegin, qEnd);
            }
            bool nextKeyPrepared = false;
            ConsumeCubeKeyMacro<CAUSAL>(keyL1, kRowsAlign, flatQueryMacros, queryMacrosPerGroup,
                                   qBegin, qEnd, request, requestQBegin, kStart, kMacroCount,
                                   coreKEnd, keyMacroIndex, streamLoop, nextKeyPrepared, residentQueries);
            // 当前 K 宏块须驻留至请求中所有组和 Q 宏块消费完成。
            keyAdapter_->ReleaseKeyMacroSlot(keySlot);
        }
        if (residentQueries) {
            for (uint32_t slot = 0U; slot < static_cast<uint32_t>(flatQueryMacros); ++slot) {
                SetFlag<HardEvent::MTE1_MTE2>(qFree_[slot]);
            }
        }
    }

    template <bool CAUSAL>
    __aicore__ inline void ProcessCubeRequests(uint32_t &streamLoop)
    {
        auto scheduler = MakeRequestScheduler<CAUSAL>();
        typename RequestScheduler<CAUSAL>::Coord work;
        while (scheduler.Next(work, *this)) {
            const auto [request, requestQBegin, qBegin, qEnd, requestKBegin, requestKEnd] = work;
            const uint32_t queryMacrosPerGroup =
                (qEnd - qBegin + QMacroTile() - 1U) / QMacroTile();
            const uint64_t flatQueryMacros =
                static_cast<uint64_t>(tiling_.groups) * queryMacrosPerGroup;
            if (flatQueryMacros == 0U) {
                continue;
            }

            const uint32_t firstKeyMacroIndex = requestKBegin / PAGED_K_MACRO;
            keyAdapter_->PrepareKeyMacroToL1(
                request, requestKBegin, PagedMin(PAGED_K_MACRO, requestKEnd - requestKBegin),
                firstKeyMacroIndex % kPagedKeyL1Slots,
                PagedAlignUp(PagedMin(PAGED_K_MACRO, requestKEnd - requestKBegin),
                                  kPagedNAlignment));

            ProcessCubeRequestKeys<CAUSAL>(request, requestQBegin, qBegin, qEnd,
                                          queryMacrosPerGroup, flatQueryMacros,
                                          requestKBegin, requestKEnd, firstKeyMacroIndex, streamLoop);
        }
    }

    template <bool CAUSAL>
    __aicore__ inline void ProcessCube()
    {
        uint32_t streamLoop = 0U;
        ProcessCubeRequests<CAUSAL>(streamLoop);

        // 耗尽两个 AIV 的结果槽许可，同时关闭未使用槽的末代许可。
        for (uint32_t slot = 0U; slot < PAGED_PIPE_SLOTS; ++slot) {
            RegionScoreSyncPolicy::AicFinalWait(slot);
        }
        CrossCoreSetFlag<PAGED_SYNC_MODE, PIPE_FIX>(PAGED_WORK_DRAINED_EVENT);
        CrossCoreSetFlag<PAGED_SYNC_MODE, PIPE_FIX>(
            PAGED_WORK_DRAINED_EVENT + PAGED_AIV1_EVENT_OFFSET);

    }

    __aicore__ inline int32_t GetGroupFusedMaxVisibleLength(
        uint32_t request, uint32_t requestBegin,
        uint32_t qStart, uint32_t qCount) const
    {
        return VisibleRegions(request, requestBegin, qStart + qCount - 1U);
    }

    template <bool CAUSAL>
    __aicore__ inline void ComputeGroupFusedQueryMicro(
        LocalTensor<QkInputType> keyL1, uint32_t qRowsAlign, uint32_t kRowsAlign,
        uint32_t qMicro, uint32_t rowsPerToken, uint32_t scoreRowCount, uint32_t request,
        uint32_t kStart, uint32_t kMacroCount, uint32_t querySlot, int32_t maxVisible,
        uint32_t &streamLoop, bool &nextKeyPrepared, uint32_t coreKEnd, uint32_t keyMacroIndex)
    {
        for (uint32_t kMicro = 0U;
             kMicro < kMacroCount; kMicro += PAGED_NR) {
            const uint32_t kMicroCount =
                PagedMin(PAGED_NR, kMacroCount - kMicro);
            const uint32_t resultSlot =
                streamLoop % PAGED_PIPE_SLOTS;
            bool fullyMasked = false;
            if constexpr (CAUSAL) {
                fullyMasked =
                    static_cast<int32_t>(kStart + kMicro) >=
                    maxVisible;
            }
            ComputeOneTile(
                keyL1, qRowsAlign, kRowsAlign,
                qMicro * rowsPerToken, kMicro,
                scoreRowCount, kMicroCount,
                resultSlot, querySlot, fullyMasked);
            ++streamLoop;

            PrepareNextKeyMacro(request, kStart, coreKEnd,
                                keyMacroIndex, nextKeyPrepared);
        }
    }

    template <bool CAUSAL>
    __aicore__ inline void ComputeGroupFusedQueryMacro(
        LocalTensor<QkInputType> keyL1, uint32_t qRowsAlign, uint32_t kRowsAlign,
        uint32_t qMacroCount, uint32_t qMicroTile, uint32_t groupCount, uint32_t rowsPerToken,
        uint32_t request, uint32_t requestQBegin, uint32_t qStart, uint32_t kStart,
        uint32_t kMacroCount, uint32_t querySlot, uint32_t &streamLoop,
        bool &nextKeyPrepared, uint32_t coreKEnd, uint32_t keyMacroIndex)
    {
        for (uint32_t qMicro = 0U; qMicro < qMacroCount;
             qMicro += qMicroTile) {
            const uint32_t qMicroCount = PagedMin(
                qMicroTile, qMacroCount - qMicro);
            const uint32_t scoreRowCount =
                qMicroCount * groupCount;
            int32_t maxVisible = 0;
            if constexpr (CAUSAL) {
                maxVisible = GetGroupFusedMaxVisibleLength(
                    request, requestQBegin,
                    qStart + qMicro, qMicroCount);
            }
            ComputeGroupFusedQueryMicro<CAUSAL>(keyL1, qRowsAlign, kRowsAlign, qMicro,
                                rowsPerToken, scoreRowCount, request, kStart, kMacroCount,
                                querySlot, maxVisible, streamLoop, nextKeyPrepared,
                                coreKEnd, keyMacroIndex);
        }
    }

    template <bool CAUSAL>
    __aicore__ inline void ConsumeGroupFusedQueries(
        LocalTensor<QkInputType> keyL1, uint32_t kRowsAlign, uint32_t queryMacroCount,
        uint32_t qMacroTile, uint32_t qMicroTile, uint32_t groupStart, uint32_t groupCount,
        uint32_t rowsPerToken, uint32_t qBegin, uint32_t qEnd, uint32_t request,
        uint32_t requestQBegin, uint32_t kStart, uint32_t kMacroCount, uint32_t &streamLoop,
        bool &nextKeyPrepared, uint32_t coreKEnd, uint32_t keyMacroIndex)
    {
        for (uint32_t queryMacro = 0U;
             queryMacro < queryMacroCount; ++queryMacro) {
            const uint32_t qStart =
                qBegin + queryMacro * qMacroTile;
            const uint32_t qMacroCount =
                PagedMin(qMacroTile, qEnd - qStart);
            const uint32_t querySlot =
                queryMacro % PAGED_PIPE_SLOTS;
            const uint32_t qRowsAlign = PagedAlignUp(
                qMacroCount * rowsPerToken, kPagedNAlignment);
            WaitFlag<HardEvent::MTE2_MTE1>(qReady_[querySlot]);
            if (queryMacro + 1U < queryMacroCount) {
                GatherGroupFusedQueryMacro(
                    queryMacro + 1U, groupStart, groupCount,
                    qBegin, qEnd);
            }

            ComputeGroupFusedQueryMacro<CAUSAL>(keyL1, qRowsAlign, kRowsAlign, qMacroCount,
                                qMicroTile, groupCount, rowsPerToken, request, requestQBegin,
                                qStart, kStart, kMacroCount, querySlot, streamLoop,
                                nextKeyPrepared, coreKEnd, keyMacroIndex);
            SetFlag<HardEvent::MTE1_MTE2>(qFree_[querySlot]);
        }
    }

    template <bool CAUSAL>
    __aicore__ inline void ConsumeGroupFusedKeyMacro(
        LocalTensor<QkInputType> keyL1, uint32_t kRowsAlign, uint32_t queryMacroCount,
        uint32_t groupTile, uint32_t qMicroTile, uint32_t qMacroTile, uint32_t qBegin,
        uint32_t qEnd, uint32_t request, uint32_t requestQBegin, uint32_t kStart,
        uint32_t kMacroCount, uint32_t &streamLoop, bool &nextKeyPrepared,
        uint32_t coreKEnd, uint32_t keyMacroIndex)
    {
        for (uint32_t groupStart = 0U;
             groupStart < tiling_.groups; groupStart += groupTile) {
            const uint32_t groupCount =
                PagedMin(groupTile, tiling_.groups - groupStart);
            const uint32_t rowsPerToken = groupCount * Heads();
            GatherGroupFusedQueryMacro(
                0U, groupStart, groupCount, qBegin, qEnd);
            ConsumeGroupFusedQueries<CAUSAL>(keyL1, kRowsAlign, queryMacroCount, qMacroTile,
                                    qMicroTile, groupStart, groupCount, rowsPerToken, qBegin, qEnd,
                                    request, requestQBegin, kStart, kMacroCount, streamLoop,
                                    nextKeyPrepared, coreKEnd, keyMacroIndex);
        }
    }

    template <bool CAUSAL>
    __aicore__ inline void ProcessGroupFusedRequestKeys(
        uint32_t request, uint32_t requestQBegin, uint32_t qBegin, uint32_t qEnd,
        uint32_t queryMacroCount, uint32_t groupTile, uint32_t qMicroTile, uint32_t qMacroTile,
        uint32_t coreKBegin, uint32_t coreKEnd, uint32_t firstKeyMacroIndex, uint32_t &streamLoop)
    {
        KeyMacroScheduler scheduler({coreKBegin, coreKEnd, firstKeyMacroIndex});
        KeyMacroScheduler::Coord macro;
        while (scheduler.Next(macro)) {
            const auto [kStart, kMacroCount, keyMacroIndex] = macro;
            const uint32_t keySlot =
                keyMacroIndex % kPagedKeyL1Slots;
            const uint32_t kRowsAlign =
                PagedAlignUp(kMacroCount, kPagedNAlignment);
            keyAdapter_->WaitKeyMacroReady(keySlot);
            LocalTensor<QkInputType> keyL1 =
                keyAdapter_->GetKeyL1(keySlot);

            bool nextKeyPrepared = false;
            ConsumeGroupFusedKeyMacro<CAUSAL>(keyL1, kRowsAlign, queryMacroCount, groupTile,
                                        qMicroTile, qMacroTile, qBegin, qEnd, request, requestQBegin,
                                        kStart, kMacroCount, streamLoop, nextKeyPrepared,
                                        coreKEnd, keyMacroIndex);
            keyAdapter_->ReleaseKeyMacroSlot(keySlot);
        }
    }

    template <bool CAUSAL>
    __aicore__ inline void ProcessGroupFusedCubeRequests(
        uint32_t groupTile, uint32_t qMicroTile, uint32_t qMacroTile, uint32_t &streamLoop)
    {
        auto scheduler = MakeRequestScheduler<CAUSAL>();
        typename RequestScheduler<CAUSAL>::Coord work;
        while (scheduler.Next(work, *this)) {
            const auto [request, requestQBegin, qBegin, qEnd, requestKBegin, requestKEnd] = work;
            const uint32_t queryMacroCount =
                (qEnd - qBegin + qMacroTile - 1U) / qMacroTile;
            if (queryMacroCount == 0U) {
                continue;
            }

            const uint32_t firstKeyMacroIndex = requestKBegin / PAGED_K_MACRO;
            keyAdapter_->PrepareKeyMacroToL1(
                request, requestKBegin,
                PagedMin(PAGED_K_MACRO, requestKEnd - requestKBegin),
                firstKeyMacroIndex % kPagedKeyL1Slots,
                PagedAlignUp(
                    PagedMin(PAGED_K_MACRO, requestKEnd - requestKBegin),
                    kPagedNAlignment));

            ProcessGroupFusedRequestKeys<CAUSAL>(request, requestQBegin, qBegin, qEnd,
                                                queryMacroCount, groupTile, qMicroTile, qMacroTile,
                                                requestKBegin, requestKEnd, firstKeyMacroIndex, streamLoop);
        }
    }

    template <bool CAUSAL>
    __aicore__ inline void ProcessCubeGroupFused()
    {
        uint32_t streamLoop = 0U;
        const uint32_t groupTile = GroupTile();
        const uint32_t qMicroTile = GroupFusedQMicroTile();
        const uint32_t qMacroTile = GroupFusedQMacroTile();

        ProcessGroupFusedCubeRequests<CAUSAL>(groupTile, qMicroTile, qMacroTile, streamLoop);

        for (uint32_t slot = 0U; slot < PAGED_PIPE_SLOTS; ++slot) {
            RegionScoreSyncPolicy::AicFinalWait(slot);
        }
        CrossCoreSetFlag<PAGED_SYNC_MODE, PIPE_FIX>(PAGED_WORK_DRAINED_EVENT);
        CrossCoreSetFlag<PAGED_SYNC_MODE, PIPE_FIX>(
            PAGED_WORK_DRAINED_EVENT + PAGED_AIV1_EVENT_OFFSET);
    }

    template <bool CAUSAL>
    __aicore__ inline bool LoadRequestSpan(uint32_t request,
                                            uint32_t &begin, uint32_t &end)
    {
        if (!keyAdapter_->GetRequestTokenSpan(request, begin, end)) {
            return false;
        }
        if constexpr (CAUSAL) {
            requestFirstPosition_ = RequestFirstTokenPosition(
                actualKvLengthsGm_.GetValue(request), begin, end);
        }
        return true;
    }

    __aicore__ inline void LoadWeights(uint32_t group, uint32_t qStart,
                                        uint32_t localQStart, uint32_t localQCount)
    {
        if (localQCount == 0U) {
            return;
        }
        const uint64_t sourceOffset =
            static_cast<uint64_t>(qStart + localQStart) * tiling_.weightStrideToken +
            static_cast<uint64_t>(group) * tiling_.weightStrideGroup;
        ReduceHeadBlock::LoadWeights(
            tiling_.weightDtype, Heads(), localQCount, tiling_.weightStrideToken,
            sourceOffset, weightUb_, weightInputBf16Ub_, weightInputFp16Ub_,
            weightsFp32Gm_, weightsBf16Gm_, weightsFp16Gm_);
        SetFlag<HardEvent::MTE2_V>(mte2ToVector_);
    }

    __aicore__ inline void LoadGroupFusedWeights(
        uint32_t qStart, uint32_t groupStart,
        uint32_t localScoreStart,
        uint32_t localScoreCount)
    {
        if (localScoreCount == 0U) {
            return;
        }
        const uint64_t sourceOffset =
            static_cast<uint64_t>(qStart) * tiling_.weightStrideToken +
            static_cast<uint64_t>(groupStart) * tiling_.weightStrideGroup +
            static_cast<uint64_t>(localScoreStart) * Heads();
        ReduceHeadBlock::LoadWeights(
            tiling_.weightDtype, Heads(), localScoreCount, Heads(),
            sourceOffset, weightUb_, weightInputBf16Ub_, weightInputFp16Ub_,
            weightsFp32Gm_, weightsBf16Gm_, weightsFp16Gm_);
        SetFlag<HardEvent::MTE2_V>(mte2ToVector_);
    }

    __aicore__ inline void LoadGroupFusedVisibleLengths(
        uint32_t request, uint32_t requestBegin,
        uint32_t qStart, uint32_t groupCount,
        uint32_t localScoreStart, uint32_t localScoreCount)
    {
        if (localScoreCount == 0U) {
            return;
        }
        for (uint32_t row = 0U; row < localScoreCount; ++row) {
            const uint32_t flatScore = localScoreStart + row;
            const uint32_t tokenOffset = flatScore / groupCount;
            visibleUb_.SetValue(
                row, VisibleRegions(request, requestBegin,
                                    qStart + tokenOffset));
        }
    }

    __aicore__ inline void FinishWeights(uint32_t localQCount)
    {
        WaitFlag<HardEvent::MTE2_V>(mte2ToVector_);
        ReduceHeadBlock::ConvertWeights(
            tiling_.weightDtype, Heads(), localQCount, tiling_.headScale,
            WeightFp32RowStride(), WeightNativeRowStride(),
            weightUb_, weightInputBf16Ub_, weightInputFp16Ub_);
        if (tiling_.weightDtype != 0U || tiling_.headScale != 1.0F) {
            SetFlag<HardEvent::V_S>(vectorToScalar_);
            WaitFlag<HardEvent::V_S>(vectorToScalar_);
        }
    }

    __aicore__ inline void LoadVisibleLengths(uint32_t request,
                                               uint32_t requestBegin,
                                               uint32_t qStart,
                                               uint32_t localQStart,
                                               uint32_t localQCount)
    {
        if (localQCount == 0U) {
            return;
        }
        for (uint32_t row = 0U; row < localQCount; ++row) {
            visibleUb_.SetValue(row, VisibleRegions(
                request, requestBegin, qStart + localQStart + row));
        }
    }

    __aicore__ inline void GetLocalQueryRange(uint32_t qCount,
                                               uint32_t &localQStart,
                                               uint32_t &localQCount) const
    {
        GetLocalQueryRange(qCount, GetSubBlockIdx(), localQStart, localQCount);
    }

    __aicore__ inline uint32_t AcquireOutputStore()
    {
        const uint32_t slot = outputStoreCount_ % PAGED_PIPE_SLOTS;
        if (outputStoreCount_ >= PAGED_PIPE_SLOTS) {
            WaitFlag<HardEvent::MTE3_V>(outputStoreDone_[slot]);
        }
        return slot;
    }

    __aicore__ inline void CommitOutputStore(uint32_t slot)
    {
        SetFlag<HardEvent::MTE3_V>(outputStoreDone_[slot]);
        ++outputStoreCount_;
    }

    __aicore__ inline void DrainOutputStores()
    {
        const uint32_t pending = PagedMin(outputStoreCount_, PAGED_PIPE_SLOTS);
        for (uint32_t slot = 0U; slot < pending; ++slot) {
            WaitFlag<HardEvent::MTE3_V>(outputStoreDone_[slot]);
        }
    }

    __aicore__ inline void StoreRows(uint32_t group, uint32_t firstQ,
                                      uint32_t kStart, uint32_t kCount,
                                      uint32_t rowCount, uint32_t rowStride,
                                      LocalTensor<float> source)
    {
        ReduceHeadBlock::StoreRows(
            tiling_, vectorToMte3_, outputGm_, group, firstQ, kStart, kCount, rowCount, rowStride, source);
    }

    template <bool CAUSAL>
    __aicore__ inline void ReduceVariableTile(
        LocalTensor<float> accSlot, LocalTensor<float> mmSlot,
        uint32_t rows, uint32_t cols, uint32_t firstColumn)
    {
        const int32_t tileBegin = static_cast<int32_t>(firstColumn);
        if constexpr (CAUSAL) {
            // Local score rows follow request token order, including group-fused rows.
            if (tileBegin >= visibleUb_.GetValue(rows - 1U)) {
                Duplicate(accSlot.ReinterpretCast<uint32_t>(),
                          IndexerRegionScoreVector::H4_NEG_INF_BITS, rows * PAGED_NR);
                return;
            }
            if (static_cast<int64_t>(firstColumn) + cols > visibleUb_.GetValue(0U)) {
                SetFlag<HardEvent::S_V>(scalarToVector_);
                WaitFlag<HardEvent::S_V>(scalarToVector_);
                IndexerRegionScoreVector::WeightedReduceTileVF<true>(
                    accSlot, mmSlot, weightUb_, visibleUb_, rows, Heads(),
                    WeightFp32RowStride(), cols, tileBegin);
                SetFlag<HardEvent::V_S>(vectorToScalar_);
                WaitFlag<HardEvent::V_S>(vectorToScalar_);
                return;
            }
        }
        IndexerRegionScoreVector::WeightedReduceTileVF<false>(
            accSlot, mmSlot, weightUb_, visibleUb_, rows, Heads(),
            WeightFp32RowStride(), cols, tileBegin);
    }

    template <bool CAUSAL>
    __aicore__ inline void StoreVariableTile(
        uint32_t group, uint32_t qStart, uint32_t localQStart, uint32_t localQCount,
        uint32_t kStart, uint32_t kCount, LocalTensor<float> mmSlot)
    {
        const uint32_t storeSlot = AcquireOutputStore();
        LocalTensor<float> accSlot = acc_[storeSlot * PAGED_OUTPUT_SLOT_ELEMENTS];
        ReduceVariableTile<CAUSAL>(accSlot, mmSlot, localQCount, kCount, kStart);
        StoreRows(group, qStart + localQStart, kStart, kCount,
                  localQCount, PAGED_NR, accSlot);
        CommitOutputStore(storeSlot);
    }

    __aicore__ inline void ReduceFullRows(
        const LocalTensor<float> &accSlot,
        const LocalTensor<float> &mmSlot)
    {
        // 单次 VF 对 16 行 H4 权重归约两个 64 列片段，搬运和事件由外层处理。
        IndexerRegionScoreVector::WeightedReduceH4RowsVF<
            2U, PAGED_H4_WEIGHT_ROW_STRIDE>(
            accSlot, mmSlot, weightUb_, 0U, PAGED_NR, PAGED_NR);
    }

    __aicore__ inline void PostprocessNoMaskTile(
        uint32_t group, uint32_t qStart, uint32_t qCount, uint32_t kStart,
        uint32_t kCount, uint32_t resultSlot)
    {
        uint32_t localQStart = 0U;
        uint32_t localQCount = 0U;
        GetLocalQueryRange(qCount, localQStart, localQCount);
        if (localQCount == 0U) {
            return;
        }
        LocalTensor<float> mmSlot = MmOut(resultSlot * PAGED_RESULT_SLOT_ELEMENTS);

        if (Heads() == PAGED_H && qCount == PAGED_MQ_MAX && kCount == PAGED_NR &&
            localQCount == PAGED_QUERIES_PER_AIV_MAX) {
            const uint32_t storeSlot = AcquireOutputStore();
            LocalTensor<float> accSlot = acc_[storeSlot * PAGED_OUTPUT_SLOT_ELEMENTS];
            ReduceFullRows(accSlot, mmSlot);
            StoreRows(group, qStart + localQStart, kStart, kCount,
                      localQCount, PAGED_NR, accSlot);
            CommitOutputStore(storeSlot);
            return;
        }

        StoreVariableTile<false>(group, qStart, localQStart, localQCount,
                                 kStart, kCount, mmSlot);
    }

    __aicore__ inline void StoreGroupFusedRows(
        uint32_t qStart, uint32_t groupStart, uint32_t groupCount,
        uint32_t localScoreStart,
        uint32_t localScoreCount, uint32_t kStart, uint32_t kCount,
        uint32_t rowStride, LocalTensor<float> source)
    {
        if ((localScoreStart % groupCount) == 0U &&
            (localScoreCount % groupCount) == 0U) {
            const uint32_t localTokenStart = localScoreStart / groupCount;
            const uint32_t localTokenCount = localScoreCount / groupCount;
            for (uint32_t groupOffset = 0U;
                 groupOffset < groupCount; ++groupOffset) {
                StoreRows(
                    groupStart + groupOffset,
                    qStart + localTokenStart, kStart, kCount,
                    localTokenCount, groupCount * rowStride,
                    source[groupOffset * rowStride]);
            }
            return;
        }
        for (uint32_t row = 0U; row < localScoreCount; ++row) {
            const uint32_t flatScore = localScoreStart + row;
            const uint32_t tokenOffset = flatScore / groupCount;
            const uint32_t group =
                groupStart + flatScore % groupCount;
            StoreRows(
                group, qStart + tokenOffset, kStart, kCount,
                1U, rowStride, source[row * rowStride]);
        }
    }

    template <bool CAUSAL>
    __aicore__ inline void StoreGroupFusedVariableTile(
        uint32_t qStart, uint32_t groupStart, uint32_t groupCount,
        uint32_t localScoreStart, uint32_t localScoreCount,
        uint32_t kStart, uint32_t kCount, LocalTensor<float> mmSlot)
    {
        const uint32_t storeSlot = AcquireOutputStore();
        LocalTensor<float> accSlot = acc_[storeSlot * PAGED_OUTPUT_SLOT_ELEMENTS];
        ReduceVariableTile<CAUSAL>(accSlot, mmSlot, localScoreCount, kCount, kStart);
        StoreGroupFusedRows(qStart, groupStart, groupCount, localScoreStart, localScoreCount,
                           kStart, kCount, PAGED_NR, accSlot);
        CommitOutputStore(storeSlot);
    }

    __aicore__ inline void PostprocessGroupFusedNoMaskTile(
        uint32_t qStart, uint32_t qCount,
        uint32_t groupStart, uint32_t groupCount, uint32_t kStart,
        uint32_t kCount, uint32_t resultSlot)
    {
        const uint32_t scoreRowCount = qCount * groupCount;
        uint32_t localScoreStart = 0U;
        uint32_t localScoreCount = 0U;
        GetLocalQueryRange(
            scoreRowCount, localScoreStart, localScoreCount);
        if (localScoreCount == 0U) {
            return;
        }
        LocalTensor<float> mmSlot =
            MmOut(resultSlot * PAGED_RESULT_SLOT_ELEMENTS);
        // 按当前 tile 尺寸选择完整块路径。
        if (Heads() == PAGED_H &&
            kCount == PAGED_NR &&
            localScoreCount == PAGED_QUERIES_PER_AIV_MAX) {
            const uint32_t storeSlot = AcquireOutputStore();
            LocalTensor<float> accSlot =
                acc_[storeSlot * PAGED_OUTPUT_SLOT_ELEMENTS];
            ReduceFullRows(accSlot, mmSlot);
            StoreGroupFusedRows(
                qStart, groupStart, groupCount,
                localScoreStart, localScoreCount,
                kStart, kCount, PAGED_NR, accSlot);
            CommitOutputStore(storeSlot);
            return;
        }

        StoreGroupFusedVariableTile<false>(qStart, groupStart, groupCount,
                                           localScoreStart, localScoreCount, kStart, kCount, mmSlot);
    }

    __aicore__ inline void StoreGroupFusedCausalFullTile(
        uint32_t qStart, uint32_t groupStart, uint32_t groupCount,
        uint32_t localScoreStart, uint32_t localScoreCount,
        uint32_t kStart, uint32_t kCount, LocalTensor<float> mmSlot)
    {
        int32_t minVisible = visibleUb_.GetValue(0U);
        int32_t maxVisible = minVisible;
        for (uint32_t row = 1U; row < localScoreCount; ++row) {
            const int32_t visible = visibleUb_.GetValue(row);
            minVisible = visible < minVisible ? visible : minVisible;
            maxVisible = visible > maxVisible ? visible : maxVisible;
        }
        const int32_t tileBegin = static_cast<int32_t>(kStart);
        const int32_t tileEnd = static_cast<int32_t>(kStart + kCount);
        const bool fullyVisible = tileEnd <= minVisible;
        const bool fullyMasked = tileBegin >= maxVisible;
        const uint32_t storeSlot = AcquireOutputStore();
        LocalTensor<float> accSlot =
            acc_[storeSlot * PAGED_OUTPUT_SLOT_ELEMENTS];
        if (fullyMasked) {
            Duplicate(
                accSlot.ReinterpretCast<uint32_t>(),
                IndexerRegionScoreVector::H4_NEG_INF_BITS,
                localScoreCount * PAGED_NR);
        }
        ReduceCausalFullTile(accSlot, mmSlot, localScoreCount, tileBegin,
                             fullyVisible, fullyMasked);
        StoreGroupFusedRows(
            qStart, groupStart, groupCount,
            localScoreStart, localScoreCount,
            kStart, kCount, PAGED_NR, accSlot);
        CommitOutputStore(storeSlot);
    }

    __aicore__ inline void PostprocessGroupFusedCausalTile(
        uint32_t qStart, uint32_t qCount,
        uint32_t groupStart, uint32_t groupCount, uint32_t kStart,
        uint32_t kCount, uint32_t resultSlot)
    {
        const uint32_t scoreRowCount = qCount * groupCount;
        uint32_t localScoreStart = 0U;
        uint32_t localScoreCount = 0U;
        GetLocalQueryRange(
            scoreRowCount, localScoreStart, localScoreCount);
        RegionScoreSyncPolicy::AivWaitCubeDone(resultSlot);
        if (localScoreCount == 0U) {
            return;
        }
        LocalTensor<float> mmSlot =
            MmOut(resultSlot * PAGED_RESULT_SLOT_ELEMENTS);
        // 完整 H4 tile 统一判断可见性，减少逐行分支并复用批量 VF 归约。
        if (Heads() == PAGED_H && kCount == PAGED_NR &&
            localScoreCount == PAGED_QUERIES_PER_AIV_MAX) {
            StoreGroupFusedCausalFullTile(qStart, groupStart, groupCount,
                                          localScoreStart, localScoreCount,
                                          kStart, kCount, mmSlot);
            return;
        }
        StoreGroupFusedVariableTile<true>(qStart, groupStart, groupCount,
                                          localScoreStart, localScoreCount, kStart, kCount, mmSlot);
    }

    __aicore__ inline void ReduceCausalFullTile(
        LocalTensor<float> accSlot, LocalTensor<float> mmSlot,
        uint32_t localQCount, int32_t tileBegin,
        bool fullyVisible, bool fullyMasked)
    {
        if constexpr (ALLOW_H4_TAIL) {
            if (fullyVisible) {
                ReduceFullRows(accSlot, mmSlot);
                return;
            }
        }
        if (fullyVisible) {
            if constexpr (CausalH4BatchPolicy<KeyAdapter>::ENABLED) {
                // This tile has a complete 128-column UB row even when logical R has a tail.
                ReduceFullRows(accSlot, mmSlot);
                return;
            } else if ((tiling_.seqK % PAGED_NR) == 0U) {
                ReduceFullRows(accSlot, mmSlot);
                return;
            }
        }
        if (fullyMasked) {
            return;
        }
        if (!fullyVisible) {
            SetFlag<HardEvent::S_V>(scalarToVector_);
            WaitFlag<HardEvent::S_V>(scalarToVector_);
            IndexerRegionScoreVector::WeightedReduceH4MaskedRowsVF<
                2U, PAGED_H4_WEIGHT_ROW_STRIDE>(
                accSlot, mmSlot, weightUb_, 0U, PAGED_NR, PAGED_NR,
                visibleUb_, tileBegin);
            // Scalar may replace visible lengths for the next Q macro only after the VF reads them.
            SetFlag<HardEvent::V_S>(vectorToScalar_);
            WaitFlag<HardEvent::V_S>(vectorToScalar_);
            return;
        }
        IndexerRegionScoreVector::WeightedReduceTileVF<false>(
            accSlot, mmSlot, weightUb_, visibleUb_, localQCount, PAGED_H,
            PAGED_H4_WEIGHT_ROW_STRIDE, PAGED_NR, tileBegin);
    }

    __aicore__ inline void StoreCausalFullTile(
        uint32_t group, uint32_t qStart, uint32_t localQStart, uint32_t localQCount,
        uint32_t kStart, uint32_t kCount, uint32_t resultSlot, LocalTensor<float> mmSlot)
    {
        int32_t minVisible = visibleUb_.GetValue(0U);
        int32_t maxVisible = minVisible;
        for (uint32_t row = 1U; row < localQCount; ++row) {
            const int32_t visible = visibleUb_.GetValue(row);
            minVisible = visible < minVisible ? visible : minVisible;
            maxVisible = visible > maxVisible ? visible : maxVisible;
        }
        const int32_t tileBegin = static_cast<int32_t>(kStart);
        const int32_t tileEnd = static_cast<int32_t>(kStart + kCount);
        const bool fullyVisible = tileEnd <= minVisible;
        const bool fullyMasked = tileBegin >= maxVisible;
        const uint32_t storeSlot = AcquireOutputStore();
        LocalTensor<float> accSlot = acc_[storeSlot * PAGED_OUTPUT_SLOT_ELEMENTS];
        if (fullyMasked) {
            Duplicate(accSlot.ReinterpretCast<uint32_t>(),
                      IndexerRegionScoreVector::H4_NEG_INF_BITS,
                      localQCount * PAGED_NR);
        }
        RegionScoreSyncPolicy::AivWaitCubeDone(resultSlot);
        ReduceCausalFullTile(accSlot, mmSlot, localQCount, tileBegin,
                             fullyVisible, fullyMasked);
        StoreRows(group, qStart + localQStart, kStart, kCount,
                  localQCount, PAGED_NR, accSlot);
        CommitOutputStore(storeSlot);
    }

    __aicore__ inline void PostprocessCausalTile(
        uint32_t group, uint32_t qStart, uint32_t qCount, uint32_t kStart,
        uint32_t kCount, uint32_t resultSlot)
    {
        uint32_t localQStart = 0U;
        uint32_t localQCount = 0U;
        GetLocalQueryRange(qCount, localQStart, localQCount);
        if (localQCount == 0U) {
            RegionScoreSyncPolicy::AivWaitCubeDone(resultSlot);
            return;
        }
        LocalTensor<float> mmSlot = MmOut(resultSlot * PAGED_RESULT_SLOT_ELEMENTS);
        if (Heads() != PAGED_H) {
            RegionScoreSyncPolicy::AivWaitCubeDone(resultSlot);
            StoreVariableTile<true>(group, qStart, localQStart, localQCount,
                                    kStart, kCount, mmSlot);
            return;
        }

        if constexpr (CausalH4BatchPolicy<KeyAdapter>::ENABLED) {
            // Retain the Q4 decode geometry and exclude completely masked tiles.
            if (tiling_.groups == 1U && tiling_.qTile == 4U &&
                qCount == 4U && localQCount == 4U && kCount == PAGED_NR &&
                static_cast<int64_t>(kStart) < visibleUb_.GetValue(3U) &&
                static_cast<int64_t>(kStart) + PAGED_NR > visibleUb_.GetValue(0U)) {
                const uint32_t storeSlot = AcquireOutputStore();
                LocalTensor<float> accSlot = acc_[storeSlot * PAGED_OUTPUT_SLOT_ELEMENTS];
                RegionScoreSyncPolicy::AivWaitCubeDone(resultSlot);
                SetFlag<HardEvent::S_V>(scalarToVector_);
                WaitFlag<HardEvent::S_V>(scalarToVector_);
                IndexerRegionScoreVector::WeightedReduceH4MaskedRowsVF<
                    2U, PAGED_H4_WEIGHT_ROW_STRIDE, 4U>(
                    accSlot, mmSlot, weightUb_, 0U, PAGED_NR, PAGED_NR,
                    visibleUb_, static_cast<int32_t>(kStart));
                SetFlag<HardEvent::V_S>(vectorToScalar_);
                WaitFlag<HardEvent::V_S>(vectorToScalar_);
                StoreRows(group, qStart + localQStart, kStart, kCount,
                          4U, PAGED_NR, accSlot);
                CommitOutputStore(storeSlot);
                return;
            }
            // A request's causal positions increase with its query rows.
            if (tiling_.groups == 1U && tiling_.qTile == 4U &&
                qCount == 4U && localQCount == 4U && kCount == PAGED_NR &&
                static_cast<int64_t>(kStart) + PAGED_NR <= visibleUb_.GetValue(0U)) {
                const uint32_t storeSlot = AcquireOutputStore();
                LocalTensor<float> accSlot = acc_[storeSlot * PAGED_OUTPUT_SLOT_ELEMENTS];
                RegionScoreSyncPolicy::AivWaitCubeDone(resultSlot);
                IndexerRegionScoreVector::WeightedReduceH4RowsVF<
                    2U, PAGED_H4_WEIGHT_ROW_STRIDE, 4U>(
                    accSlot, mmSlot, weightUb_, 0U, PAGED_NR, PAGED_NR);
                StoreRows(group, qStart + localQStart, kStart, kCount,
                          4U, PAGED_NR, accSlot);
                CommitOutputStore(storeSlot);
                return;
            }
        }

        if (qCount == PAGED_MQ_MAX && kCount == PAGED_NR &&
            localQCount == PAGED_QUERIES_PER_AIV_MAX) {
            StoreCausalFullTile(group, qStart, localQStart, localQCount,
                                kStart, kCount, resultSlot, mmSlot);
            return;
        }

        RegionScoreSyncPolicy::AivWaitCubeDone(resultSlot);
        StoreVariableTile<true>(group, qStart, localQStart, localQCount,
                                kStart, kCount, mmSlot);
    }

    template <bool CAUSAL>
    __aicore__ inline void ConsumeVectorResult(
        uint32_t group, uint32_t qStart, uint32_t qMicro, uint32_t qMicroCount,
        uint32_t kStart, uint32_t kMicro, uint32_t kMicroCount, uint32_t resultSlot,
        uint32_t request, uint32_t localQStart, uint32_t localQCount, uint32_t &streamLoop)
    {

        if constexpr (!CAUSAL) {
            RegionScoreSyncPolicy::AivWaitCubeDone(resultSlot);
        }

        if constexpr (CAUSAL) {
            PostprocessCausalTile(
                group, qStart + qMicro, qMicroCount,
                kStart + kMicro, kMicroCount, resultSlot);
        } else {
            PostprocessNoMaskTile(
                group, qStart + qMicro, qMicroCount,
                kStart + kMicro, kMicroCount, resultSlot);
        }
        PipeBarrier<PIPE_V>();
        RegionScoreSyncPolicy::AivArriveVecReady(resultSlot);

        ++streamLoop;
    }

    template <bool CAUSAL>
    __aicore__ inline void ProcessVectorQueryMacro(
        uint32_t request, uint32_t requestQBegin, uint32_t group, uint32_t qStart,
        uint32_t qMacroCount, uint32_t kStart, uint32_t kMacroCount, uint32_t &streamLoop)
    {
        for (uint32_t qMicro = 0U; qMicro < qMacroCount;
             qMicro += QMicroTile()) {
            const uint32_t qMicroCount =
                PagedMin(QMicroTile(), qMacroCount - qMicro);
            uint32_t localQStart = 0U;
            uint32_t localQCount = 0U;
            GetLocalQueryRange(qMicroCount, localQStart, localQCount);
            if (localQCount != 0U) {
                LoadWeights(group, qStart + qMicro,
                            localQStart, localQCount);
            }
            if constexpr (CAUSAL) {
                LoadVisibleLengths(request, requestQBegin,
                                   qStart + qMicro,
                                   localQStart, localQCount);
            }
            if (localQCount != 0U) {
                FinishWeights(localQCount);
            }
            for (uint32_t kMicro = 0U; kMicro < kMacroCount;
                 kMicro += PAGED_NR) {
                const uint32_t kMicroCount =
                    PagedMin(PAGED_NR, kMacroCount - kMicro);
                const uint32_t resultSlot = streamLoop % PAGED_PIPE_SLOTS;
                ConsumeVectorResult<CAUSAL>(group, qStart, qMicro, qMicroCount, kStart, kMicro,
                        kMicroCount, resultSlot, request, localQStart, localQCount, streamLoop);
            }
            // 权重 UB 仅一份，必须等本代所有 VF 消费完成才可发起下一次搬运。
            if (localQCount != 0U) {
                SetFlag<HardEvent::V_MTE2>(vectorToMte2_);
                WaitFlag<HardEvent::V_MTE2>(vectorToMte2_);
            }
        }
    }

    template <bool CAUSAL>
    __aicore__ inline void ConsumeVectorKeyMacro(
        uint32_t request, uint32_t requestQBegin, uint32_t qBegin, uint32_t qEnd,
        uint32_t kStart, uint32_t kMacroCount, uint32_t &streamLoop)
    {
        for (uint32_t group = 0U; group < tiling_.groups; ++group) {
            for (uint32_t qStart = qBegin; qStart < qEnd; qStart += QMacroTile()) {
                const uint32_t qMacroCount =
                    PagedMin(QMacroTile(), qEnd - qStart);
                ProcessVectorQueryMacro<CAUSAL>(request, requestQBegin, group, qStart,
                                      qMacroCount, kStart, kMacroCount, streamLoop);
            }
        }
    }

    template <bool CAUSAL>
    __aicore__ inline void ProcessVectorRequests(uint32_t &streamLoop)
    {
        auto scheduler = MakeRequestScheduler<CAUSAL>();
        typename RequestScheduler<CAUSAL>::Coord work;
        while (scheduler.Next(work, *this)) {
            const auto [request, requestQBegin, qBegin, qEnd, requestKBegin, requestKEnd] = work;

            KeyMacroScheduler macroScheduler({requestKBegin, requestKEnd, requestKBegin / PAGED_K_MACRO});
            KeyMacroScheduler::Coord macro;
            while (macroScheduler.Next(macro)) {
                const uint32_t kStart = macro.start;
                const uint32_t kMacroCount = macro.count;
                ConsumeVectorKeyMacro<CAUSAL>(request, requestQBegin, qBegin, qEnd,
                                          kStart, kMacroCount, streamLoop);
            }
        }
    }

    template <bool CAUSAL>
    __aicore__ inline void ProcessVector()
    {
        for (uint32_t slot = 0U; slot < PAGED_PIPE_SLOTS; ++slot) {
            RegionScoreSyncPolicy::AivArriveVecReady(slot);
        }

        uint32_t streamLoop = 0U;
        ProcessVectorRequests<CAUSAL>(streamLoop);
        CrossCoreWaitFlag<PAGED_SYNC_MODE, PIPE_V>(PAGED_WORK_DRAINED_EVENT);

    }

    template <bool CAUSAL>
    __aicore__ inline void ConsumeGroupFusedVectorResults(
        uint32_t qStart, uint32_t qMicro, uint32_t qMicroCount, uint32_t groupStart,
        uint32_t groupCount, uint32_t kStart, uint32_t kMacroCount, uint32_t &streamLoop)
    {
        for (uint32_t kMicro = 0U;
             kMicro < kMacroCount; kMicro += PAGED_NR) {
            const uint32_t kMicroCount =
                PagedMin(PAGED_NR, kMacroCount - kMicro);
            const uint32_t resultSlot =
                streamLoop % PAGED_PIPE_SLOTS;
            if constexpr (CAUSAL) {
                PostprocessGroupFusedCausalTile(
                    qStart + qMicro, qMicroCount,
                    groupStart, groupCount,
                    kStart + kMicro, kMicroCount,
                    resultSlot);
            } else {
                RegionScoreSyncPolicy::AivWaitCubeDone(resultSlot);
                PostprocessGroupFusedNoMaskTile(
                    qStart + qMicro, qMicroCount,
                    groupStart, groupCount,
                    kStart + kMicro, kMicroCount,
                    resultSlot);
            }
            PipeBarrier<PIPE_V>();
            RegionScoreSyncPolicy::AivArriveVecReady(resultSlot);
            ++streamLoop;
        }
    }

    template <bool CAUSAL>
    __aicore__ inline void ProcessGroupFusedVectorQueryMacro(
        uint32_t request, uint32_t requestQBegin, uint32_t qStart, uint32_t qMacroCount,
        uint32_t qMicroTile, uint32_t groupStart, uint32_t groupCount, uint32_t kStart,
        uint32_t kMacroCount, uint32_t &streamLoop)
    {
        for (uint32_t qMicro = 0U; qMicro < qMacroCount;
             qMicro += qMicroTile) {
            const uint32_t qMicroCount = PagedMin(
                qMicroTile, qMacroCount - qMicro);
            const uint32_t scoreRowCount =
                qMicroCount * groupCount;
            uint32_t localScoreStart = 0U;
            uint32_t localScoreCount = 0U;
            GetLocalQueryRange(
                scoreRowCount,
                localScoreStart, localScoreCount);
            if (localScoreCount != 0U) {
                LoadGroupFusedWeights(
                    qStart + qMicro, groupStart,
                    localScoreStart, localScoreCount);
                if constexpr (CAUSAL) {
                    LoadGroupFusedVisibleLengths(
                        request, requestQBegin,
                        qStart + qMicro, groupCount,
                        localScoreStart, localScoreCount);
                }
                FinishWeights(localScoreCount);
            }

            ConsumeGroupFusedVectorResults<CAUSAL>(qStart, qMicro, qMicroCount, groupStart,
                                groupCount, kStart, kMacroCount, streamLoop);
            if (localScoreCount != 0U) {
                SetFlag<HardEvent::V_MTE2>(vectorToMte2_);
                WaitFlag<HardEvent::V_MTE2>(vectorToMte2_);
            }
        }
    }

    template <bool CAUSAL>
    __aicore__ inline void ConsumeGroupFusedVectorKeyMacro(
        uint32_t request, uint32_t requestQBegin, uint32_t qBegin, uint32_t qEnd,
        uint32_t groupTile, uint32_t qMacroTile, uint32_t qMicroTile, uint32_t kStart,
        uint32_t kMacroCount, uint32_t &streamLoop)
    {
        for (uint32_t groupStart = 0U;
             groupStart < tiling_.groups; groupStart += groupTile) {
            const uint32_t groupCount =
                PagedMin(groupTile, tiling_.groups - groupStart);
            for (uint32_t qStart = qBegin; qStart < qEnd;
                 qStart += qMacroTile) {
                const uint32_t qMacroCount =
                    PagedMin(qMacroTile, qEnd - qStart);
                ProcessGroupFusedVectorQueryMacro<CAUSAL>(request, requestQBegin, qStart, qMacroCount,
                                            qMicroTile, groupStart, groupCount, kStart,
                                            kMacroCount, streamLoop);
            }
        }
    }

    template <bool CAUSAL>
    __aicore__ inline void ProcessGroupFusedVectorRequests(
        uint32_t groupTile, uint32_t qMacroTile, uint32_t qMicroTile, uint32_t &streamLoop)
    {
        auto scheduler = MakeRequestScheduler<CAUSAL>();
        typename RequestScheduler<CAUSAL>::Coord work;
        while (scheduler.Next(work, *this)) {
            const auto [request, requestQBegin, qBegin, qEnd, requestKBegin, requestKEnd] = work;

            KeyMacroScheduler macroScheduler({requestKBegin, requestKEnd, requestKBegin / PAGED_K_MACRO});
            KeyMacroScheduler::Coord macro;
            while (macroScheduler.Next(macro)) {
                const uint32_t kStart = macro.start;
                const uint32_t kMacroCount = macro.count;
                ConsumeGroupFusedVectorKeyMacro<CAUSAL>(request, requestQBegin, qBegin, qEnd,
                                                 groupTile, qMacroTile, qMicroTile, kStart,
                                                 kMacroCount, streamLoop);
            }
        }
    }

    template <bool CAUSAL>
    __aicore__ inline void ProcessVectorGroupFused()
    {
        for (uint32_t slot = 0U; slot < PAGED_PIPE_SLOTS; ++slot) {
            RegionScoreSyncPolicy::AivArriveVecReady(slot);
        }

        uint32_t streamLoop = 0U;
        const uint32_t groupTile = GroupTile();
        const uint32_t qMicroTile = GroupFusedQMicroTile();
        const uint32_t qMacroTile = GroupFusedQMacroTile();

        ProcessGroupFusedVectorRequests<CAUSAL>(groupTile, qMacroTile, qMicroTile, streamLoop);
        CrossCoreWaitFlag<PAGED_SYNC_MODE, PIPE_V>(PAGED_WORK_DRAINED_EVENT);
    }

    PagedTilingData tiling_{};
    TPipe *pipe_ = nullptr;
    KeyAdapter *keyAdapter_ = nullptr;
    int64_t requestFirstPosition_ = 0;

    GlobalTensor<QkInputType> indexQGm_;
    GlobalTensor<float> weightsFp32Gm_;
    GlobalTensor<bfloat16_t> weightsBf16Gm_;
    GlobalTensor<half> weightsFp16Gm_;
    GlobalTensor<int32_t> actualKvLengthsGm_;
    GlobalTensor<float> outputGm_;

    TEventID qReady_[PAGED_PIPE_SLOTS];
    TEventID qFree_[PAGED_PIPE_SLOTS];
    TEventID mToMte1_[PAGED_PIPE_SLOTS];
    TEventID fixToM_[PAGED_PIPE_SLOTS];
    TEventID mte1ToM_;
    TEventID mToFix_;
    TEventID mte2ToVector_;
    TEventID vectorToScalar_;
    TEventID scalarToVector_;
    TEventID vectorToMte2_;
    TEventID vectorToMte3_;
    TEventID outputStoreDone_[PAGED_PIPE_SLOTS];

    TBuf<TPosition::A1> qL1Buf_;
    TBuf<TPosition::A2> qL0Buf_;
    TBuf<TPosition::B2> kL0Buf_;
    TBuf<TPosition::CO1> cL0Buf_;
    TBuf<TPosition::VECCALC> mmOutBuf_[PAGED_PIPE_SLOTS];
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> weightUbBuf_;
    TBuf<TPosition::VECCALC> weightInputUbBuf_;
    TBuf<TPosition::VECCALC> cuSeqLensUbBuf_;
    TBuf<TPosition::VECCALC> visibleUbBuf_;

    LocalTensor<QkInputType> qL1_[PAGED_PIPE_SLOTS];
    LocalTensor<QkInputType> qL0_;
    LocalTensor<QkInputType> kL0_;
    LocalTensor<float> cL0_;
    LocalTensor<float> mmOut_[PAGED_PIPE_SLOTS];
    LocalTensor<float> acc_;
    LocalTensor<float> weightUb_;
    LocalTensor<bfloat16_t> weightInputBf16Ub_;
    LocalTensor<half> weightInputFp16Ub_;
    LocalTensor<int32_t> cuSeqLensUb_;
    LocalTensor<int32_t> visibleUb_;
    GM_ADDR scheduleCu_ = nullptr;
    ScorePartition scorePartition_{};
    uint32_t balanceQBand_ = 0U;
    uint32_t outputStoreCount_ = 0U;
};

} // 命名空间 IndexerRegionScoreWeighted

namespace IndexerRegionScoreWeighted {
// All launched CV participants execute every batch, including empty Q/K shards.
// Compute/adapter Release drain L1, local events and MTE3 before selection's
// SyncAll/Reset. The final barrier publishes IDs and returns the single pool.
template <typename CacheStrategy, typename MaskStrategy, typename SelectionStrategy,
          typename Compute>
__aicore__ inline void RunBoundedScoreBatches(
    GM_ADDR query, GM_ADDR weights, GM_ADDR keySummary, GM_ADDR actualKvLengths,
    GM_ADDR blockTable, GM_ADDR cuSeqLens, GM_ADDR selectedRegionIds, GM_ADDR workspace,
    const PagedTilingData &tiling, uint64_t scratchOffset, AscendC::TPipe *pipe, GM_ADDR scorePlan)
{
    PagedTilingData batch = tiling;
    GM_ADDR scorePool = workspace + tiling.scorePoolOffset;
    using BatchScheduler = SparseIndexer::Common::Block::BlockScheduler<RegionScoreBatchPolicy>;
    BatchScheduler scheduler({tiling.totalTokens, tiling.batchTokens});
    BatchScheduler::Coord work;
    while (scheduler.Next(work)) {
        const uint32_t begin = work.begin;
        batch.batchBegin = begin;
        batch.batchCount = work.count;

        typename CacheStrategy::Adapter keyAdapter;
        Compute compute;
        compute.Init(query, weights, actualKvLengths, scorePool, &batch, pipe, &keyAdapter);
        CacheStrategy::InitAdapter(keyAdapter, keySummary, blockTable, cuSeqLens, &batch, pipe);
        GM_ADDR planCu = (blockTable == nullptr && batch.numRequests == 1U) ? nullptr : cuSeqLens;
        compute.template ConfigureDeviceSchedule<MaskStrategy::CAUSAL>(planCu, actualKvLengths, scorePlan);
        MaskStrategy::Process(compute);
        compute.Release();
        keyAdapter.Release();

        SelectionStrategy::Run(scorePool, actualKvLengths, cuSeqLens, selectedRegionIds,
                               workspace, batch, scratchOffset, pipe);

        // Selector stores wait for MTE3 completion, including private native paths.
        // All AIC/AIVs participate even when their current shard owns no rows.
        RegionScoreSyncPolicy::GlobalBarrier();

        pipe->Reset();
        if ASCEND_IS_AIC {
            // DAV3510 TPipe::Init reserves and seeds M_MTE1 IDs 0/1/2 for
            // its final Destroy. Reset clears event allocation bookkeeping,
            // but preserves those pending hardware permits. Reserve the IDs
            // again so the next compute uses private events 3 and above.
            // Do not Set or Wait here: Destroy must consume the original
            // three permits exactly once after the last batch.
            (void)pipe->AllocEventID<HardEvent::M_MTE1>();
            (void)pipe->AllocEventID<HardEvent::M_MTE1>();
            (void)pipe->AllocEventID<HardEvent::M_MTE1>();
        }
    }
}

// 策略模式在编译期绑定行为，执行器管理资源生命周期。
template <typename CacheStrategy, typename MaskStrategy, typename SelectionStrategy, bool QUERY_RESIDENT = false,
          bool ALLOW_H4_TAIL = false>
class RegionScoreRunner final {
public:
    __aicore__ inline void Run(GM_ADDR query, GM_ADDR weights, GM_ADDR keySummary,
        GM_ADDR actualKvLengths, GM_ADDR blockTable, GM_ADDR cuSeqLens,
        GM_ADDR output, GM_ADDR selectedRegionIds, GM_ADDR workspace,
        const PagedTilingData &tiling, uint64_t scratchOffset, AscendC::TPipe *pipe)
    {
        GM_ADDR scorePlan = workspace;
        workspace += DeviceScorePlanBytes(tiling.numRequests, tiling.usedCubeCores, tiling.totalTokens, tiling.batchTokens);
        if (tiling.batchTokens != 0U) {
            RunBoundedScoreBatches<CacheStrategy, MaskStrategy, SelectionStrategy,
                KeyResidentCompute<typename CacheStrategy::Adapter, QUERY_RESIDENT, ALLOW_H4_TAIL>>(
                query, weights, keySummary, actualKvLengths, blockTable, cuSeqLens,
                selectedRegionIds, workspace, tiling, scratchOffset, pipe, scorePlan);
            return;
        }
        typename CacheStrategy::Adapter keyAdapter;
        KeyResidentCompute<typename CacheStrategy::Adapter, QUERY_RESIDENT, ALLOW_H4_TAIL> compute;
        compute.Init(query, weights, actualKvLengths, output, &tiling, pipe, &keyAdapter);
        CacheStrategy::InitAdapter(keyAdapter, keySummary, blockTable, cuSeqLens, &tiling, pipe);
        GM_ADDR planCu = (blockTable == nullptr && tiling.numRequests == 1U) ? nullptr : cuSeqLens;
        compute.template ConfigureDeviceSchedule<MaskStrategy::CAUSAL>(planCu, actualKvLengths, scorePlan);
        MaskStrategy::Process(compute);
        compute.Release();
        keyAdapter.Release();
        SelectionStrategy::Run(output, actualKvLengths, cuSeqLens, selectedRegionIds,
                               workspace, tiling, scratchOffset, pipe);
    }
};
} // 命名空间 IndexerRegionScoreWeighted

// 版权所有 (c) 2026 华为技术有限公司，遵循 CANN 开源软件许可协议 2.0。

namespace IndexerRegionScoreWeighted {
template <typename CacheStrategy, typename MaskStrategy, typename SelectionStrategy, bool ALLOW_H4_TAIL = false>
using D128QueryResidentRunner = RegionScoreRunner<CacheStrategy, MaskStrategy, SelectionStrategy, true, ALLOW_H4_TAIL>;
} // namespace IndexerRegionScoreWeighted

// 版权所有 (c) 2026 华为技术有限公司，遵循 CANN 开源软件许可协议 2.0。

namespace IndexerRegionScoreWeighted {
struct NoMaskStrategy {
    static constexpr bool CAUSAL = false;
    template <typename Compute>
    __aicore__ static inline void Process(Compute &compute)
    {

        // 输入布局为 [T,G,H,D]，按硬件块处理分组以保证各布局连续访问。
        if (compute.UseGroupFused()) {
            if ASCEND_IS_AIC {
                compute.template ProcessCubeGroupFused<false>();
            } else {
                compute.template ProcessVectorGroupFused<false>();
            }
        } else if ASCEND_IS_AIC {
            compute.template ProcessCube<false>();
        } else {
            compute.template ProcessVector<false>();
        }
    }
};
} // 命名空间 IndexerRegionScoreWeighted

// 版权所有 (c) 2026 华为技术有限公司，遵循 CANN 开源软件许可协议 2.0。

namespace IndexerRegionScoreWeighted {
struct RightDownCausalMaskStrategy {
    static constexpr bool CAUSAL = true;
    template <typename Compute>
    __aicore__ static inline void Process(Compute &compute)
    {
        if (compute.UseGroupFused()) {
            if ASCEND_IS_AIC {
                compute.template ProcessCubeGroupFused<true>();
            } else {
                compute.template ProcessVectorGroupFused<true>();
            }
        } else if ASCEND_IS_AIC {
            compute.template ProcessCube<true>();
        } else {
            compute.template ProcessVector<true>();
        }
    }
};
} // 命名空间 IndexerRegionScoreWeighted

#endif // 结束保护 IRS_WEIGHTED_PIPELINE_H
#endif // 结束保护 INDEXER_REGION_SCORE_COMMON_VECTOR_ONLY
