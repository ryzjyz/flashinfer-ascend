#ifndef INDEXER_REGION_SCORE_STAGE_REGISTRY_HPP
#define INDEXER_REGION_SCORE_STAGE_REGISTRY_HPP

namespace IndexerRegionScore::Block {

enum class Stage { KeyCache, Mmad, ReduceHead, TopK };
enum class KeyCacheKind { Continuous, Paged, PagedD128 };

template <Stage, class Policy>
struct StageImplementation;

template <Stage StageId, class Policy>
using StageType = typename StageImplementation<StageId, Policy>::Type;

template <KeyCacheKind Kind, bool PartialPages = false>
struct KeyCachePolicy {};

template <class KeyAdapter, bool QueryResident>
struct ComputeStagePolicy {};

template <class SelectionPolicy, auto Compact>
struct SelectionStagePolicy {};

} // namespace IndexerRegionScore::Block

#endif
