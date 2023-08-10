#include "src/gen_alpha_poc/state.h"

namespace openturbine::gen_alpha_solver {

State::State()
    : generalized_coords_("generalized_coordinates", 1),
      velocity_("velocities", 1),
      acceleration_("accelerations", 1),
      algorithmic_acceleration_("algorithmic_accelerations", 1) {
}

State::State(HostView1D q, HostView1D v, HostView1D v_dot, HostView1D a)
    : generalized_coords_("generalized_coordinates", q.size()),
      velocity_("velocities", v.size()),
      acceleration_("accelerations", v_dot.size()),
      algorithmic_acceleration_("algorithmic_accelerations", a.size()) {
    Kokkos::deep_copy(generalized_coords_, q);
    Kokkos::deep_copy(velocity_, v);
    Kokkos::deep_copy(acceleration_, v_dot);
    Kokkos::deep_copy(algorithmic_acceleration_, a);
}

MassMatrix::MassMatrix(double mass, Vector J) : mass_(mass), principal_moment_of_inertia_(J) {
    if (mass <= 0.) {
        throw std::invalid_argument("Mass must be positive");
    }
    if (J.GetXComponent() <= 0. || J.GetYComponent() <= 0. || J.GetZComponent() <= 0.) {
        throw std::invalid_argument("Moment of inertia must be positive");
    }

    auto mass_matrix = std::vector<std::vector<double>>{
        {mass, 0., 0., 0., 0., 0.},               // row 1
        {0., mass, 0., 0., 0., 0.},               // row 2
        {0., 0., mass, 0., 0., 0.},               // row 3
        {0., 0., 0., J.GetXComponent(), 0., 0.},  // row 4
        {0., 0., 0., 0., J.GetYComponent(), 0.},  // row 5
        {0., 0., 0., 0., 0., J.GetZComponent()}   // row 6
    };
    this->mass_matrix_ = create_matrix(mass_matrix);
}

MassMatrix::MassMatrix(double mass, double moment_of_inertia)
    : MassMatrix(mass, {moment_of_inertia, moment_of_inertia, moment_of_inertia}) {
}

MassMatrix::MassMatrix(HostView2D mass_matrix)
    : mass_matrix_("mass_matrix", mass_matrix.extent(0), mass_matrix.extent(1)) {
    if (mass_matrix_.extent(0) != 6 || mass_matrix_.extent(1) != 6) {
        throw std::invalid_argument("Mass matrix must be 6 x 6");
    }
    Kokkos::deep_copy(mass_matrix_, mass_matrix);
    this->mass_ = mass_matrix_(0, 0);
    this->principal_moment_of_inertia_ =
        Vector(mass_matrix_(3, 3), mass_matrix_(4, 4), mass_matrix_(5, 5));
}

HostView2D MassMatrix::GetMomentOfInertiaMatrix() const {
    HostView2D moment_of_inertia_matrix("Moment of inertia matrix", 3, 3);

    constexpr int numComponents = 3;
    double J[numComponents] = {
        this->principal_moment_of_inertia_.GetXComponent(),
        this->principal_moment_of_inertia_.GetYComponent(),
        this->principal_moment_of_inertia_.GetZComponent()};

    Kokkos::parallel_for(
        "Moment of inertia matrix", numComponents,
        KOKKOS_LAMBDA(const int i) { moment_of_inertia_matrix(i, i) = J[i]; }
    );

    return moment_of_inertia_matrix;
}

GeneralizedForces::GeneralizedForces(const Vector& forces, const Vector& moments)
    : forces_(forces), moments_(moments) {
    this->generalized_forces_ = create_vector({
        forces.GetXComponent(),   // row 1
        forces.GetYComponent(),   // row 2
        forces.GetZComponent(),   // row 3
        moments.GetXComponent(),  // row 4
        moments.GetYComponent(),  // row 5
        moments.GetZComponent()   // row 6
    });
}

GeneralizedForces::GeneralizedForces(HostView1D generalized_forces)
    : generalized_forces_("generalized_forces_vector", generalized_forces.size()) {
    if (generalized_forces_.extent(0) != 6) {
        throw std::invalid_argument("Generalized forces must be 6 x 1");
    }
    Kokkos::deep_copy(generalized_forces_, generalized_forces);
    this->forces_ = Vector(generalized_forces_(0), generalized_forces_(1), generalized_forces_(2));
    this->moments_ = Vector(generalized_forces_(3), generalized_forces_(4), generalized_forces_(5));
}

}  // namespace openturbine::gen_alpha_solver
