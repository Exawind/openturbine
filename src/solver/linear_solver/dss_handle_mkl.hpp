#pragma once

#include <vector>

#include <mkl.h>
#include <mkl_pardiso.h>

namespace openturbine {

template <>
class DSSHandle<DSSAlgorithm::MKL> {
    struct mklDssHandleType {
        MKL_INT iparm[64];
        void* pt[64];
        MKL_INT mtype;
        MKL_INT msglvl;
        MKL_INT maxfct;
        MKL_INT nrhs;
        MKL_INT mnum;
        std::vector<MKL_INT> perm;

        mklDssHandleType() {
            mtype = 11;
            nrhs = 1;
            msglvl = 0;
            maxfct = 1;
            mnum = 1;
            pardisoinit(pt, &mtype, iparm);

            iparm[1] = 3;
            iparm[12] = 1;
            iparm[23] = 0;
            iparm[24] = 1;

            iparm[27] = 0;  // double
            iparm[34] = 1;  // zero-based
        }

        ~mklDssHandleType() {
            MKL_INT error;
            const MKL_INT finish_phase = -1;
            const auto num_rows = static_cast<MKL_INT>(perm.size());
            pardiso(
                pt, &maxfct, &mnum, &mtype, &finish_phase, &num_rows, nullptr, nullptr, nullptr,
                perm.data(), &nrhs, iparm, &msglvl, nullptr, nullptr, &error
            );
        }
    };
    std::shared_ptr<mklDssHandleType> mkl_dss_handle;

public:
    DSSHandle() : mkl_dss_handle(std::make_shared<mklDssHandleType>()) {}

    void** get_handle() { return mkl_dss_handle->pt; }

    MKL_INT* get_iparm() { return mkl_dss_handle->iparm; }

    MKL_INT& get_mtype() { return mkl_dss_handle->mtype; }

    MKL_INT& get_msglvl() { return mkl_dss_handle->msglvl; }

    MKL_INT& get_maxfct() { return mkl_dss_handle->maxfct; }

    MKL_INT& get_nrhs() { return mkl_dss_handle->nrhs; }

    MKL_INT& get_mnum() { return mkl_dss_handle->mnum; }

    std::vector<MKL_INT>& get_perm() { return mkl_dss_handle->perm; }
};
}  // namespace openturbine
