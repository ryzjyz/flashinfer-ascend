#ifndef MLA_PROLOG_H
#define MLA_PROLOG_H

/*
 * Frozen standalone host ABI names (Dsv41MlaPrologArguments / Dsv41MlaProlog)
 * are retained only for traceability of the verified standalone runner. They
 * are not the public repository operator name; the public name is mla_prolog.
 */

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

#endif // MLA_PROLOG_H
