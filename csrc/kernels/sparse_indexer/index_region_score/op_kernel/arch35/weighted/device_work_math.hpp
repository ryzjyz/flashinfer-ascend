#ifndef IRS_WEIGHTED_DEVICE_WORK_MATH_HPP
#define IRS_WEIGHTED_DEVICE_WORK_MATH_HPP

#include <cstdint>

namespace IndexerRegionScoreWeighted {

// firstPosition is the causal position at qBegin. Position plus Q extent must
// fit int64_t, and the weighted work total must fit uint64_t.
struct DeviceWorkShape {
    uint32_t qBegin;
    uint32_t qEnd;
    uint32_t qBand;
    uint32_t kMacro;
    uint32_t seqK;
    uint32_t regionSize;
    int64_t firstPosition;
    bool causal;
    uint32_t heads = 0U;
};

__aicore__ inline uint32_t BandCount(const DeviceWorkShape &shape)
{
    if (shape.qBand == 0U || shape.qEnd <= shape.qBegin) {
        return 0U;
    }
    const uint32_t length = shape.qEnd - shape.qBegin;
    return length / shape.qBand + (length % shape.qBand != 0U);
}

__aicore__ inline uint32_t BandKEnd(const DeviceWorkShape &shape, uint32_t band)
{
    if (shape.kMacro == 0U || band >= BandCount(shape)) {
        return 0U;
    }
    uint64_t visible = shape.seqK;
    if (shape.causal) {
        if (shape.regionSize == 0U) {
            return 0U;
        }
        const uint64_t length = shape.qEnd - shape.qBegin;
        const uint64_t bandEnd = (static_cast<uint64_t>(band) + 1U) * shape.qBand;
        const int64_t position = shape.firstPosition +
            static_cast<int64_t>(bandEnd < length ? bandEnd : length) - 1;
        visible = position > 0 ? static_cast<uint64_t>(position) / shape.regionSize : 0U;
        visible = visible < shape.seqK ? visible : shape.seqK;
    }
    return static_cast<uint32_t>(visible / shape.kMacro + (visible % shape.kMacro != 0U));
}

__aicore__ inline uint32_t KMacroCount(const DeviceWorkShape &shape)
{
    const uint32_t bands = BandCount(shape);
    return bands == 0U ? 0U : BandKEnd(shape, bands - 1U);
}

__aicore__ inline uint32_t FirstActiveBand(const DeviceWorkShape &shape, uint32_t kIndex)
{
    const uint32_t bands = BandCount(shape);
    if (bands == 0U || kIndex >= KMacroCount(shape)) {
        return bands;
    }
    if (!shape.causal) {
        return 0U;
    }
    const uint64_t threshold = static_cast<uint64_t>(shape.regionSize) *
        (static_cast<uint64_t>(kIndex) * shape.kMacro + 1U);
    // KMacroCount has already established that the last band reaches threshold.
    uint64_t distance;
    if (shape.firstPosition >= 0) {
        const uint64_t first = static_cast<uint64_t>(shape.firstPosition);
        if (threshold <= first) {
            return 0U;
        }
        distance = threshold - first;
    } else {
        const uint64_t magnitude = static_cast<uint64_t>(-(shape.firstPosition + 1)) + 1U;
        distance = threshold + magnitude;
    }
    return static_cast<uint32_t>(distance / shape.qBand);
}

// The hardware issues N128 microtiles even when the resident K macro is wider.
__aicore__ inline uint32_t KeyMicrotiles(const DeviceWorkShape &shape, uint32_t macro)
{
    constexpr uint32_t microtile = 128U;
    const uint32_t remaining = shape.seqK - macro * shape.kMacro;
    const uint32_t rows = remaining < shape.kMacro ? remaining : shape.kMacro;
    return rows / microtile + (rows % microtile != 0U);
}

// The last Q band may have fewer physical M16 blocks than full bands.
__aicore__ inline uint32_t TailBandCost(const DeviceWorkShape &shape, uint32_t tileCost)
{
    if (shape.heads == 0U || (shape.qEnd - shape.qBegin) % shape.qBand == 0U) {
        return tileCost;
    }
    const uint32_t fullM16 = (shape.qBand * shape.heads + 15U) / 16U;
    const uint32_t remainder = (shape.qEnd - shape.qBegin) % shape.qBand;
    const uint32_t tailM16 = (remainder * shape.heads + 15U) / 16U;
    return (tileCost / fullM16) * tailM16;
}

// Sum floor((a*i+b)/m), 0 <= i < n. Euclidean iteration, logarithmic in m.
__aicore__ inline uint64_t WorkFloorSum(uint64_t n, uint64_t m, uint64_t a, uint64_t b)
{
    uint64_t sum = 0U;
    while (true) {
        if (a >= m) {
            const uint64_t pairs = (n & 1U) == 0U ? (n / 2U) * (n - 1U) : n * ((n - 1U) / 2U);
            sum += pairs * (a / m);
            a %= m;
        }
        if (b >= m) {
            sum += n * (b / m);
            b %= m;
        }
        const uint64_t product = a * n;
        uint64_t quotient = product / m;
        const uint64_t remainder = product % m;
        uint64_t nextB;
        if (remainder >= m - b) {
            ++quotient;
            nextB = remainder - (m - b);
        } else {
            nextB = remainder + b;
        }
        if (quotient == 0U) {
            return sum;
        }
        n = quotient;
        b = nextB;
        const uint64_t previousM = m;
        m = a;
        a = previousM;
    }
}

__aicore__ inline uint64_t WorkPrefix(
    const DeviceWorkShape &shape, uint32_t kLimit, uint32_t loadCost, uint32_t tileCost)
{
    const uint32_t totalK = KMacroCount(shape);
    const uint32_t limit = kLimit < totalK ? kLimit : totalK;
    if (limit == 0U) {
        return 0U;
    }
    const uint32_t bands = BandCount(shape);
    uint64_t tileCount = static_cast<uint64_t>(bands) * limit;
    if (shape.causal && bands > 1U && BandKEnd(shape, 0U) < limit) {
        // Positive bands use ceil(floor(position/regionSize)/kMacro).
        // Full-band last positions form an arithmetic progression. Only the
        // final Q tail needs a separate term. Clip zero and saturated bands.
        const uint32_t length = shape.qEnd - shape.qBegin;
        const uint32_t fullBands = length / shape.qBand;
        const uint32_t firstPositive = FirstActiveBand(shape, 0U);
        const uint32_t firstSaturated = FirstActiveBand(shape, limit - 1U);
        const uint32_t begin = firstPositive < fullBands ? firstPositive : fullBands;
        const uint32_t end = firstSaturated < fullBands ? firstSaturated : fullBands;
        tileCount = static_cast<uint64_t>(fullBands - end) * limit;
        if (end > begin) {
            const uint64_t position = static_cast<uint64_t>(shape.firstPosition +
                static_cast<int64_t>((static_cast<uint64_t>(begin) + 1U) * shape.qBand) - 1);
            const uint64_t divisor = static_cast<uint64_t>(shape.regionSize) * shape.kMacro;
            const uint64_t adjustment = static_cast<uint64_t>(shape.regionSize) * (shape.kMacro - 1U);
            uint64_t base = position / divisor;
            uint64_t remainder = position % divisor;
            if (remainder >= divisor - adjustment) {
                ++base;
                remainder -= divisor - adjustment;
            } else {
                remainder += adjustment;
            }
            const uint64_t count = end - begin;
            tileCount += count * base + WorkFloorSum(count, divisor, shape.qBand, remainder);
        }
        if (length % shape.qBand != 0U) {
            const uint32_t tail = BandKEnd(shape, fullBands);
            tileCount += tail < limit ? tail : limit;
        }
    }
    const uint32_t fullWidth = KeyMicrotiles(shape, 0U);
    const uint32_t tailWidth = KeyMicrotiles(shape, limit - 1U);
    const uint32_t widthDifference = fullWidth - tailWidth;
    const uint64_t loadedMicrotiles = static_cast<uint64_t>(limit) * fullWidth - widthDifference;
    const uint64_t activeTailBands = bands - FirstActiveBand(shape, limit - 1U);
    const uint64_t computedMicrotiles = tileCount * fullWidth - activeTailBands * widthDifference;
    return loadedMicrotiles * loadCost + computedMicrotiles * tileCost -
        loadedMicrotiles * (tileCost - TailBandCost(shape, tileCost));
}

// Each K macro starts with a load-cost prefix, followed by one tileCost for
// every active Q band. Return the first band whose start is at or after target.
// A cut inside a band's cost rounds forward. A cut in a load prefix returns
// that macro's first active band. End is (KMacroCount(shape), 0).
__aicore__ inline void SeekCut(
    const DeviceWorkShape &shape, uint64_t target, uint32_t loadCost, uint32_t tileCost,
    uint32_t &kIndex, uint32_t &band)
{
    const uint32_t totalK = KMacroCount(shape);
    kIndex = totalK;
    band = 0U;
    if (target >= WorkPrefix(shape, totalK, loadCost, tileCost)) {
        return;
    }
    uint32_t low = 0U;
    uint64_t prefix = 0U;
    const uint32_t bands = BandCount(shape);
    if (bands == 1U || !shape.causal || BandKEnd(shape, 0U) == totalK) {
        const uint64_t fullMacroCost = KeyMicrotiles(shape, 0U) *
            (loadCost + static_cast<uint64_t>(bands - 1U) * tileCost + TailBandCost(shape, tileCost));
        if (fullMacroCost != 0U) {
            low = static_cast<uint32_t>(target / fullMacroCost);
            prefix = low * fullMacroCost;
        }
    } else {
        uint32_t high = totalK;
        while (low < high) {
            const uint32_t middle = low + (high - low) / 2U;
            if (WorkPrefix(shape, middle + 1U, loadCost, tileCost) <= target) {
                low = middle + 1U;
            } else {
                high = middle;
            }
        }
        prefix = WorkPrefix(shape, low, loadCost, tileCost);
    }
    kIndex = low;
    band = FirstActiveBand(shape, low);
    const uint64_t offset = target - prefix;
    const uint32_t width = KeyMicrotiles(shape, low);
    loadCost *= width;
    tileCost *= width;
    if (offset > loadCost && tileCost != 0U) {
        const uint64_t remaining = offset - loadCost;
        band += static_cast<uint32_t>(remaining / tileCost + (remaining % tileCost != 0U));
    }
    if (band >= BandCount(shape)) {
        ++kIndex;
        band = kIndex < totalK ? FirstActiveBand(shape, kIndex) : 0U;
    }
}

} // namespace IndexerRegionScoreWeighted
#endif
