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
    size_t num_nodes{0};
};

TEST_F(NodeStateWriterTest, ConstructorCreatesExpectedDimensionsAndVariables) {
    const util::NodeStateWriter writer(test_file, true, num_nodes);
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

TEST_F(NodeStateWriterTest, WriteStateDataAtTimestepForPosition) {
    util::NodeStateWriter writer(test_file, true, num_nodes);

    const std::vector<double> x = {1., 2., 3.};
    const std::vector<double> y = {4., 5., 6.};
    const std::vector<double> z = {7., 8., 9.};
    const std::vector<double> i = {0.1, 0.2, 0.3};
    const std::vector<double> j = {0.4, 0.5, 0.6};
    const std::vector<double> k = {0.7, 0.8, 0.9};
    const std::vector<double> w = {1., 1., 1.};

    EXPECT_NO_THROW(writer.WriteStateDataAtTimestep(0, "x", x, y, z, i, j, k, w));

    const auto& file = writer.GetFile();
    std::vector<double> read_data(num_nodes);

    file.ReadVariable("x_x", read_data.data());
    EXPECT_EQ(read_data, x);

    file.ReadVariable("x_y", read_data.data());
    EXPECT_EQ(read_data, y);

    file.ReadVariable("x_z", read_data.data());
    EXPECT_EQ(read_data, z);

    file.ReadVariable("x_i", read_data.data());
    EXPECT_EQ(read_data, i);

    file.ReadVariable("x_j", read_data.data());
    EXPECT_EQ(read_data, j);

    file.ReadVariable("x_k", read_data.data());
    EXPECT_EQ(read_data, k);

    file.ReadVariable("x_w", read_data.data());
    EXPECT_EQ(read_data, w);
}

TEST_F(NodeStateWriterTest, WriteStateDataAtTimestepForVelocity) {
    util::NodeStateWriter writer(test_file, true, num_nodes);

    const std::vector<double> x = {1., 2., 3.};
    const std::vector<double> y = {4., 5., 6.};
    const std::vector<double> z = {7., 8., 9.};
    const std::vector<double> i = {0.1, 0.2, 0.3};
    const std::vector<double> j = {0.4, 0.5, 0.6};
    const std::vector<double> k = {0.7, 0.8, 0.9};

    EXPECT_NO_THROW(writer.WriteStateDataAtTimestep(0, "v", x, y, z, i, j, k));

    const auto& file = writer.GetFile();
    std::vector<double> read_data(num_nodes);
    const std::vector<size_t> start = {0, 0};
    const std::vector<size_t> count = {1, num_nodes};

    file.ReadVariableAt("v_x", start, count, read_data.data());
    EXPECT_EQ(read_data, x);

    file.ReadVariableAt("v_y", start, count, read_data.data());
    EXPECT_EQ(read_data, y);

    file.ReadVariableAt("v_z", start, count, read_data.data());
    EXPECT_EQ(read_data, z);

    file.ReadVariableAt("v_i", start, count, read_data.data());
    EXPECT_EQ(read_data, i);

    file.ReadVariableAt("v_j", start, count, read_data.data());
    EXPECT_EQ(read_data, j);

    file.ReadVariableAt("v_k", start, count, read_data.data());
    EXPECT_EQ(read_data, k);
}

TEST_F(NodeStateWriterTest, ThrowsOnInvalidComponentPrefix) {
    util::NodeStateWriter writer(test_file, true, num_nodes);
    const std::vector<double> data(num_nodes, 1.);

    EXPECT_THROW(
        writer.WriteStateDataAtTimestep(0, "invalid_prefix", data, data, data, data, data, data),
        std::invalid_argument
    );
}

TEST_F(NodeStateWriterTest, ThrowsOnMismatchedVectorSizes) {
    util::NodeStateWriter writer(test_file, true, num_nodes);

    const std::vector<double> correct_size(num_nodes, 1.);    // write data to 3 nodes
    const std::vector<double> wrong_size(num_nodes + 1, 1.);  // write data to 4 nodes

    EXPECT_THROW(
        writer.WriteStateDataAtTimestep(
            0, "position", correct_size, wrong_size, correct_size, correct_size, correct_size,
            correct_size, correct_size
        ),
        std::invalid_argument
    );
}

}  // namespace openturbine::tests
