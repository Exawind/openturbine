#include <gtest/gtest.h>

#include "src/gebt_poc/mesh.h"
#include "src/gebt_poc/field_data.h"

#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc {

class LinearizationParameters {
public:
};

class UnityLinearizationParameters : public LinearizationParameters{

};

class GeneralizedAlphaStepper {
public:
  bool Step(Mesh& mesh, FieldData& field_data, std::size_t lagrange_multipliers) {
    return true;
  }


  friend GeneralizedAlphaStepper CreateBasicStepper();

protected:
  GeneralizedAlphaStepper() = default;

  void SetParameters(double alpha_f, double alpha_m, double beta, double gamma) {
   alphaF_ = alpha_f;
   alphaM_ = alpha_m;
   beta_ = beta;
   gamma_ = gamma;
  }

  void SetPreconditioner(bool is_preconditioned) {
    is_preconditioned_ = is_preconditioned;
  }

  void SetSystemAssembler(std::shared_ptr<LinearizationParameters> assembler) {
    assembler_ = assembler;
  }

  std::shared_ptr<LinearizationParameters> assembler_;

  double alphaF_;
  double alphaM_;
  double beta_;
  double gamma_;
  bool is_preconditioned_;
};

GeneralizedAlphaStepper CreateBasicStepper() {
  GeneralizedAlphaStepper stepper;
  stepper.SetParameters(0., 0., .5, 1.);
  stepper.SetPreconditioner(true);

  stepper.SetSystemAssembler(std::make_shared<UnityLinearizationParameters>());

  return stepper;
}

}


TEST(GEBT_TimeIntegratorTest, AlphaStepSolutionAfterOneIncWithNonZeroInitialState) {
    using namespace openturbine::gebt_poc;
    auto stepper = CreateBasicStepper();

    auto mesh = Create1DMesh(1, 1);
    auto field_data = FieldData(mesh, 1);
    constexpr auto lie_group_size = 7;
    constexpr auto lie_algebra_size = 6;

std::cout << 1 << std::endl;
    Kokkos::parallel_for(mesh.GetNumberOfNodes(), KOKKOS_LAMBDA(int node) {
      auto coordinates = field_data.GetNodalData<Field::Coordinates>(node);
      for(int i = 0; i < lie_group_size; ++i) {
        coordinates(i) = 0.;
      }

      auto velocity = field_data.GetNodalData<Field::Velocity>(node);
      auto acceleration = field_data.GetNodalData<Field::Acceleration>(node);
      auto algo_acceleration = field_data.GetNodalData<Field::AlgorithmicAcceleration>(node);
      for(int i = 0; i < lie_algebra_size; ++i)
      {
        velocity(i) = i + 1.;
        acceleration(i) = i + 1.;
        algo_acceleration(i) = i + 1.;
      }
    });

std::cout << 2 << std::endl;
    size_t n_lagrange_mults{0};

std::cout << 3 << std::endl;
    bool step_converged = stepper.Step(mesh, field_data, n_lagrange_mults);

std::cout << 4 << std::endl;
    EXPECT_TRUE(step_converged);
std::cout << 5 << std::endl;
    using openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal;
    auto coordinates = Kokkos::View<double*>("coordinates", lie_group_size);
std::cout << 5.1 << std::endl;
    auto c = field_data.GetNodalData<Field::Coordinates>(0);
std::cout << 5.2 << std::endl;
std::cout << c.extent(0) << std::endl;
    Kokkos::deep_copy(coordinates, c);
std::cout << 5.5 << std::endl;
    expect_kokkos_view_1D_equal(coordinates, {1., 2., 3., 0., 0., 0., 0.});
std::cout << 6 << std::endl;
    auto velocity = Kokkos::View<double*>("velocity", lie_algebra_size);
    Kokkos::deep_copy(velocity, field_data.GetNodalData<Field::Velocity>(0));
    expect_kokkos_view_1D_equal(velocity, {-1., 0., 1., 2., 3., 4.});
std::cout << 7 << std::endl;
    auto acceleration = Kokkos::View<double*>("acceleration", lie_algebra_size);
    Kokkos::deep_copy(acceleration, field_data.GetNodalData<Field::Acceleration>(0));
    expect_kokkos_view_1D_equal(acceleration, {-2., -2., -2., -2., -2., -2.});
std::cout << 8 << std::endl;
    auto algo_acceleration = Kokkos::View<double*>("algo acceleration", lie_algebra_size);
    Kokkos::deep_copy(algo_acceleration, field_data.GetNodalData<Field::AlgorithmicAcceleration>(0));
    expect_kokkos_view_1D_equal(algo_acceleration, {-2., -2., -2., -2., -2., -2.});
}
