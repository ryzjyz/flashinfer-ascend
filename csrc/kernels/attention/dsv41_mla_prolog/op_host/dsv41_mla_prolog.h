#ifndef DSV41_MLA_PROLOG_H
#define DSV41_MLA_PROLOG_H

#include <acl/acl.h>
#include <cstdint>

struct Dsv41MlaPrologArguments {
    const uint16_t *x;
    const uint16_t *wQa;
    const uint16_t *gammaQa;
    const int8_t *wQb;
    const uint16_t *wQbScale;
    const uint16_t *wKv;
    const uint16_t *gammaKv;
    const float *ropeCos;
    const float *ropeSin;
    const int32_t *slotIndices;
    uint16_t *kvCache;
    uint16_t *qOutput;
    void *workspace;
    uint64_t workspaceSize;
    int32_t t;
};

uint64_t GetDsv41MlaPrologWorkspaceSize(const Dsv41MlaPrologArguments &args);

aclError Dsv41MlaProlog(
    const Dsv41MlaPrologArguments &args,
    aclrtStream stream);

#endif
