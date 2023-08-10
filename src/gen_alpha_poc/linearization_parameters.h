#pragma once

#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gen_alpha_solver {

/// Abstract base class to provide problem-specific residual vector and iteration matrix
class LinearizationParameters {
public:
    virtual ~LinearizationParameters() = default;

    virtual HostView1D ResidualVector(
        const HostView1D, const HostView1D, const HostView1D, const HostView1D
    ) = 0;

    virtual HostView2D IterationMatrix(
        const double&, const double&, const double&, const HostView1D, const HostView1D,
        const HostView1D, const HostView1D, const HostView1D
    ) = 0;
};

/// Defines a unity residual vector and identity iteration matrix
class UnityLinearizationParameters : public LinearizationParameters {
public:
    UnityLinearizationParameters(){};

    virtual HostView1D ResidualVector(
        const HostView1D, const HostView1D, const HostView1D, const HostView1D
    ) override;

    virtual HostView2D IterationMatrix(
        const double&, const double&, const double&, const HostView1D, const HostView1D,
        const HostView1D, const HostView1D, const HostView1D
    ) override;
};

}  // namespace openturbine::gen_alpha_solver
