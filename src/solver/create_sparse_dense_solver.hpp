#pragma once

#include <string>

#include <Amesos2.hpp>

namespace openturbine {

template <typename GlobalCrsMatrixType, typename GlobalMultiVectorType>
[[nodiscard]] inline Teuchos::RCP<Amesos2::Solver<GlobalCrsMatrixType, GlobalMultiVectorType>>
CreateSparseDenseSolver(
    Teuchos::RCP<GlobalCrsMatrixType>& A, Teuchos::RCP<GlobalMultiVectorType>& x_mv,
    Teuchos::RCP<GlobalMultiVectorType>& b
) {
    const auto solver_name = std::string{"klu2"};
    auto amesos_solver =
        Amesos2::create<GlobalCrsMatrixType, GlobalMultiVectorType>(solver_name, A, x_mv, b);
    amesos_solver->symbolicFactorization();
    return amesos_solver;
}

}  // namespace openturbine
