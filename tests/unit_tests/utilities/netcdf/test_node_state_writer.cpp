#include <filesystem>

#include <gtest/gtest.h>

#include "utilities/netcdf/node_state_writer.hpp"

namespace openturbine::tests {

class NodeStateWriterTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_file = "test_node_state.nc";
        std::filesystem::remove(test_file);
        num_nodes = 3;
    }

    void TearDown() override { std::filesystem::remove(test_file); }

    std::string test_file;
    size_t num_nodes;
};

TEST_F(NodeStateWriterTest, ConstructorCreatesExpectedDimensionsAndVariables) {
    util::NodeStateWriter writer(test_file, true, num_nodes);
    const auto& file = writer.GetFile();

    EXPECT_EQ(writer.GetNumNodes(), num_nodes);

    EXPECT_NO_THROW({
        // dimensions
        EXPECT_GE(file.GetDimensionId("time"), 0);
        EXPECT_GE(file.GetDimensionId("nodes"), 0);

        // position, x
        EXPECT_GE(file.GetVariableId("x_x"), 0);
        EXPECT_GE(file.GetVariableId("x_y"), 0);
        EXPECT_GE(file.GetVariableId("x_z"), 0);
        EXPECT_GE(file.GetVariableId("x_i"), 0);
        EXPECT_GE(file.GetVariableId("x_j"), 0);
        EXPECT_GE(file.GetVariableId("x_k"), 0);
        EXPECT_GE(file.GetVariableId("x_w"), 0);

        // displacement, u
        EXPECT_GE(file.GetVariableId("u_x"), 0);
        EXPECT_GE(file.GetVariableId("u_y"), 0);
        EXPECT_GE(file.GetVariableId("u_z"), 0);
        EXPECT_GE(file.GetVariableId("u_i"), 0);
        EXPECT_GE(file.GetVariableId("u_j"), 0);
        EXPECT_GE(file.GetVariableId("u_k"), 0);
        EXPECT_GE(file.GetVariableId("u_w"), 0);

        // velocity, v
        EXPECT_GE(file.GetVariableId("v_x"), 0);
        EXPECT_GE(file.GetVariableId("v_y"), 0);
        EXPECT_GE(file.GetVariableId("v_z"), 0);
        EXPECT_GE(file.GetVariableId("v_i"), 0);
        EXPECT_GE(file.GetVariableId("v_j"), 0);
        EXPECT_GE(file.GetVariableId("v_k"), 0);

        // acceleration, a
        EXPECT_GE(file.GetVariableId("a_x"), 0);
        EXPECT_GE(file.GetVariableId("a_y"), 0);
        EXPECT_GE(file.GetVariableId("a_z"), 0);
        EXPECT_GE(file.GetVariableId("a_i"), 0);
        EXPECT_GE(file.GetVariableId("a_j"), 0);
        EXPECT_GE(file.GetVariableId("a_k"), 0);

        // force, f
        EXPECT_GE(file.GetVariableId("f_x"), 0);
        EXPECT_GE(file.GetVariableId("f_y"), 0);
        EXPECT_GE(file.GetVariableId("f_z"), 0);
        EXPECT_GE(file.GetVariableId("f_i"), 0);
        EXPECT_GE(file.GetVariableId("f_j"), 0);
        EXPECT_GE(file.GetVariableId("f_k"), 0);
    });
}

TEST_F(NodeStateWriterTest, WriteStateDataForPosition) {
    util::NodeStateWriter writer(test_file, true, num_nodes);

    std::vector<double> x = {1., 2., 3.};
    std::vector<double> y = {4., 5., 6.};
    std::vector<double> z = {7., 8., 9.};
    std::vector<double> i = {0.1, 0.2, 0.3};
    std::vector<double> j = {0.4, 0.5, 0.6};
    std::vector<double> k = {0.7, 0.8, 0.9};
    std::vector<double> w = {1., 1., 1.};

    EXPECT_NO_THROW(writer.WriteStateData(0, "x", x, y, z, i, j, k, w));
}

TEST_F(NodeStateWriterTest, WriteStateDataForVelocity) {
    util::NodeStateWriter writer(test_file, true, num_nodes);

    std::vector<double> x = {1., 2., 3.};
    std::vector<double> y = {4., 5., 6.};
    std::vector<double> z = {7., 8., 9.};
    std::vector<double> i = {0.1, 0.2, 0.3};
    std::vector<double> j = {0.4, 0.5, 0.6};
    std::vector<double> k = {0.7, 0.8, 0.9};

    EXPECT_NO_THROW(writer.WriteStateData(0, "v", x, y, z, i, j, k));
}

TEST_F(NodeStateWriterTest, ThrowsOnInvalidComponentPrefix) {
    util::NodeStateWriter writer(test_file, true, num_nodes);
    std::vector<double> data(num_nodes, 1.);

    EXPECT_THROW(
        writer.WriteStateData(0, "invalid_prefix", data, data, data, data, data, data),
        std::invalid_argument
    );
}

TEST_F(NodeStateWriterTest, ThrowsOnMismatchedVectorSizes) {
    util::NodeStateWriter writer(test_file, true, num_nodes);

    std::vector<double> correct_size(num_nodes, 1.);    // 3
    std::vector<double> wrong_size(num_nodes + 1, 1.);  // 4

    EXPECT_THROW(
        writer.WriteStateData(
            0, "position", correct_size, wrong_size, correct_size, correct_size, correct_size,
            correct_size, correct_size
        ),
        std::invalid_argument
    );
}

}  // namespace openturbine::tests
