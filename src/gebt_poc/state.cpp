#include "src/gebt_poc/state.h"

#include <KokkosBlas.hpp>

#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

State::State()
    : generalized_coords_("generalized_coordinates", 1),
      velocity_("velocities", 1),
      acceleration_("accelerations", 1),
      algorithmic_acceleration_("algorithmic_accelerations", 1) {
}

State::State(
    LieGroupFieldView gen_coords, LieAlgebraFieldView velocity, LieAlgebraFieldView accln,
    LieAlgebraFieldView algo_accln
)
    : generalized_coords_("generalized_coordinates", gen_coords.extent(0)),
      velocity_("velocities", velocity.extent(0)),
      acceleration_("accelerations", accln.extent(0)),
      algorithmic_acceleration_("algorithmic_accelerations", algo_accln.extent(0)) {
    auto n_gen_coords = gen_coords.extent(0);
    auto n_velocities = velocity.extent(0);
    auto n_accelerations = accln.extent(0);
    auto n_algo_accelerations = algo_accln.extent(0);

    if (n_gen_coords != n_velocities || n_gen_coords != n_accelerations ||
        n_gen_coords != n_algo_accelerations) {
        throw std::invalid_argument(
            "The number rows (i.e. nodes) of generalized coordinates, velocities, accelerations, "
            "and algorithmic accelerations in the initial state must be the same"
        );
    }

    Kokkos::deep_copy(generalized_coords_, gen_coords);
    Kokkos::deep_copy(velocity_, velocity);
    Kokkos::deep_copy(acceleration_, accln);
    Kokkos::deep_copy(algorithmic_acceleration_, algo_accln);
}
}  // namespace openturbine::gebt_poc
