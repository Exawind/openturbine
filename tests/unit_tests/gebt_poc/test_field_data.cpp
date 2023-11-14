#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/gebt_poc/field_data.h"

template <openturbine::gebt_poc::Field field>
void initializeNodalData(
    const openturbine::gebt_poc::Mesh& mesh, openturbine::gebt_poc::FieldData& field_data,
    double multiplier
) {
    Kokkos::parallel_for(
        mesh.GetNumberOfNodes(),
        KOKKOS_LAMBDA(int node) {
            auto nodal_values = field_data.GetNodalData<field>(node);
            for (std::size_t component = 0; component < nodal_values.extent(0); ++component) {
                nodal_values(component) = multiplier * (node * nodal_values.extent(0) + component);
            }
        }
    );
}

template <openturbine::gebt_poc::Field field>
void readNodalData(
    const openturbine::gebt_poc::Mesh& mesh, openturbine::gebt_poc::FieldData& field_data,
    double multiplier
) {
    for (int node = 0; node < mesh.GetNumberOfNodes(); ++node) {
        auto nodal_values = field_data.ReadNodalData<field>(node);
        auto host_values = Kokkos::create_mirror(nodal_values);
        Kokkos::deep_copy(host_values, nodal_values);
        for (std::size_t component = 0; component < host_values.extent(0); ++component) {
            EXPECT_EQ(
                host_values(component), multiplier * (node * host_values.extent(0) + component)
            );
        }
    }
}

TEST(FieldDataTest, CreateNodalDataAndAccess) {
    int number_of_elements = 2;
    int nodes_per_element = 3;
    auto mesh = openturbine::gebt_poc::Create1DMesh(number_of_elements, nodes_per_element);
    auto field_data = openturbine::gebt_poc::FieldData(mesh, 1);
    using openturbine::gebt_poc::Field;

    initializeNodalData<Field::Coordinates>(mesh, field_data, 1.);
    initializeNodalData<Field::Velocity>(mesh, field_data, 2.);
    initializeNodalData<Field::Acceleration>(mesh, field_data, 3.);
    initializeNodalData<Field::AlgorithmicAcceleration>(mesh, field_data, 4.);
    initializeNodalData<Field::CoordinatesNext>(mesh, field_data, 5.);
    initializeNodalData<Field::AlgorithmicAccelerationNext>(mesh, field_data, 6.);
    initializeNodalData<Field::DeltaCoordinates>(mesh, field_data, 7.);

    readNodalData<Field::Coordinates>(mesh, field_data, 1.);
    readNodalData<Field::Velocity>(mesh, field_data, 2.);
    readNodalData<Field::Acceleration>(mesh, field_data, 3.);
    readNodalData<Field::AlgorithmicAcceleration>(mesh, field_data, 4.);
    readNodalData<Field::CoordinatesNext>(mesh, field_data, 5.);
    readNodalData<Field::AlgorithmicAccelerationNext>(mesh, field_data, 6.);
    readNodalData<Field::DeltaCoordinates>(mesh, field_data, 7.);
}

struct SetElementWeights {
  openturbine::gebt_poc::FieldData field_data;

  KOKKOS_FUNCTION
  void operator()(int element) {
    auto weights = field_data.GetElementData<openturbine::gebt_poc::Field::Weight>(element);
    weights(0) = element * 4.;
    weights(1) = element * 4. + 1.;
    weights(2) = element * 4. + 2.;
    weights(3) = element * 4. + 3.;
  }
};

struct SetStiffnessMatrix {
  openturbine::gebt_poc::FieldData field_data;

  KOKKOS_FUNCTION
  void operator()(int element) {
    auto stiffness = field_data.GetElementData<openturbine::gebt_poc::Field::StiffnessMatrix>(element);
    for (int point = 0; point < stiffness.extent(0); ++point) {
      for (int row = 0; row < 6; ++row) {
        for (int column = 0; column < 6; ++column) {
          stiffness(point, row, column) = point * row * column;
        }
      }
    }
  }
};

TEST(FieldDataTest, CreateElementDataAndAccess) {
    int number_of_elements = 2;
    int nodes_per_element = 3;
    int quadrature_points_per_element = 4;
    auto mesh = openturbine::gebt_poc::Create1DMesh(number_of_elements, nodes_per_element);
    auto field_data = openturbine::gebt_poc::FieldData(mesh, quadrature_points_per_element);
    using openturbine::gebt_poc::Field;

    Kokkos::parallel_for(mesh.GetNumberOfElements(), SetElementWeights{field_data});

    for (int element = 0; element < number_of_elements; ++element) {
        auto weights = field_data.ReadElementData<Field::Weight>(element);
        EXPECT_EQ(weights.extent(0), 4);
        auto host_weights = Kokkos::create_mirror(weights);
        Kokkos::deep_copy(host_weights, weights);
        EXPECT_EQ(host_weights(0), element * 4.);
        EXPECT_EQ(host_weights(1), element * 4. + 1.);
        EXPECT_EQ(host_weights(2), element * 4. + 2.);
        EXPECT_EQ(host_weights(3), element * 4. + 3.);
    }

    Kokkos::parallel_for(mesh.GetNumberOfElements(), SetStiffnessMatrix{field_data});

    for (int element = 0; element < number_of_elements; ++element) {
        auto stiffness = field_data.ReadElementData<Field::StiffnessMatrix>(element);
        EXPECT_EQ(stiffness.extent(0), 4);
        EXPECT_EQ(stiffness.extent(1), 6);
        EXPECT_EQ(stiffness.extent(2), 6);
        auto host_stiffness = Kokkos::create_mirror(stiffness);
        Kokkos::deep_copy(host_stiffness, stiffness);
        for (int point = 0; point < quadrature_points_per_element; ++point) {
            for (int row = 0; row < 6; ++row) {
                for (int column = 0; column < 6; ++column) {
                    EXPECT_EQ(host_stiffness(point, row, column), point * row * column);
                }
            }
        }
    }
}
