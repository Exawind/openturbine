#pragma once

namespace openturbine {

struct StepParameters {
    bool is_dynamic_solve;
    size_t max_iter;
    double h;
    double alpha_m;
    double alpha_f;
    double gamma;
    double beta;
    double gamma_prime;
    double beta_prime;
    double conditioner;
    double absolute_convergence_tol;
    double relative_convergence_tol;

    StepParameters(
        bool is_dynamic_solve_, size_t max_iter_, double h_, double rho_inf, double a_tol = 1e-7,
        double r_tol = 1e-5
    )
        : is_dynamic_solve(is_dynamic_solve_),
          max_iter(max_iter_),
          h(h_),
          alpha_m((2. * rho_inf - 1.) / (rho_inf + 1.)),
          alpha_f(rho_inf / (rho_inf + 1.)),
          gamma(0.5 + alpha_f - alpha_m),
          beta(0.25 * (gamma + 0.5) * (gamma + 0.5)),
          gamma_prime((is_dynamic_solve) ? gamma / (h * beta) : 0.),
          beta_prime((is_dynamic_solve) ? (1. - alpha_m) / (h * h * beta * (1. - alpha_f)) : 0.),
          conditioner(beta * h * h),
          absolute_convergence_tol(a_tol),
          relative_convergence_tol(r_tol) {}
};

}  // namespace openturbine
