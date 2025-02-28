#include <filesystem>

#include <gtest/gtest.h>

#include "utilities/netcdf/netcdf_file.hpp"

namespace openturbine::tests {

class NetCDFFileTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_file = "test_output.nc";
        std::filesystem::remove(test_file);
    }

    void TearDown() override { std::filesystem::remove(test_file); }

    std::string test_file;
};

TEST_F(NetCDFFileTest, CreateAndCloseFile) {
    EXPECT_NO_THROW({
        util::NetCDFFile file(test_file);
        EXPECT_NE(file.GetNetCDFId(), -1);
    });
    EXPECT_TRUE(std::filesystem::exists(test_file));
}

TEST_F(NetCDFFileTest, AddDimension) {
    util::NetCDFFile file(test_file);
    EXPECT_NO_THROW({
        int dim_id = file.AddDimension("test_dim", 10);
        EXPECT_GE(dim_id, 0);
        EXPECT_EQ(file.GetDimensionId("test_dim"), dim_id);
    });
}

TEST_F(NetCDFFileTest, AddAndWriteVariableWithTypeDouble) {
    util::NetCDFFile file(test_file);
    int dim_id = file.AddDimension("time", 5);
    std::vector<int> dim_ids = {dim_id};

    file.AddVariable<double>("position", dim_ids);
    std::vector<double> data = {1.1, 2.2, 3.3, 4.4, 5.5};
    EXPECT_NO_THROW(file.WriteVariable("position", data));
}

TEST_F(NetCDFFileTest, AddAndWriteVariableWithTypeFloat) {
    util::NetCDFFile file(test_file);
    int dim_id = file.AddDimension("time", 5);
    std::vector<int> dim_ids = {dim_id};

    file.AddVariable<float>("velocity", dim_ids);
    std::vector<float> data = {1.f, 2.f, 3.f, 4.f, 5.f};
    EXPECT_NO_THROW(file.WriteVariable("velocity", data));
}

TEST_F(NetCDFFileTest, AddAndWriteVariableWithTypeInt) {
    util::NetCDFFile file(test_file);
    int dim_id = file.AddDimension("time", 5);
    std::vector<int> dim_ids = {dim_id};

    file.AddVariable<int>("count", dim_ids);
    std::vector<int> data = {1, 2, 3, 4, 5};
    EXPECT_NO_THROW(file.WriteVariable("count", data));
}

TEST_F(NetCDFFileTest, AddAndWriteVariableWithTypeString) {
    util::NetCDFFile file(test_file);
    int dim_id = file.AddDimension("time", 5);
    std::vector<int> dim_ids = {dim_id};

    file.AddVariable<std::string>("labels", dim_ids);
    std::vector<std::string> data = {"one", "two", "three", "four", "five"};
    EXPECT_NO_THROW(file.WriteVariable("labels", data));
}

TEST_F(NetCDFFileTest, AddAndWriteVariableAtWithTypeDouble) {
    util::NetCDFFile file(test_file);
    int dim_id = file.AddDimension("time", 10);
    std::vector<int> dim_ids = {dim_id};
    file.AddVariable<double>("position", dim_ids);

    std::vector<double> data = {1.1, 2.2, 3.3};
    std::vector<size_t> start = {2};
    std::vector<size_t> count = {3};
    EXPECT_NO_THROW(file.WriteVariableAt("position", start, count, data));
}

TEST_F(NetCDFFileTest, AddAndWriteVariableAtWithTypeFloat) {
    util::NetCDFFile file(test_file);
    int dim_id = file.AddDimension("time", 10);
    std::vector<int> dim_ids = {dim_id};
    file.AddVariable<float>("velocity", dim_ids);

    std::vector<float> data = {1.f, 2.f, 3.f};
    std::vector<size_t> start = {4};
    std::vector<size_t> count = {3};
    EXPECT_NO_THROW(file.WriteVariableAt("velocity", start, count, data));
}

TEST_F(NetCDFFileTest, AddAndWriteVariableAtWithTypeInt) {
    util::NetCDFFile file(test_file);
    int dim_id = file.AddDimension("time", 10);
    std::vector<int> dim_ids = {dim_id};
    file.AddVariable<int>("count", dim_ids);
    std::vector<int> data = {1, 2, 3};
    std::vector<size_t> start = {6};
    std::vector<size_t> count = {3};
    EXPECT_NO_THROW(file.WriteVariableAt("count", start, count, data));
}

TEST_F(NetCDFFileTest, AddAndWriteVariableAtWithTypeString) {
    util::NetCDFFile file(test_file);
    int dim_id = file.AddDimension("time", 10);
    std::vector<int> dim_ids = {dim_id};
    file.AddVariable<std::string>("labels", dim_ids);

    std::vector<std::string> data = {"one", "two", "three"};
    std::vector<size_t> start = {0};
    std::vector<size_t> count = {3};
    EXPECT_NO_THROW(file.WriteVariableAt("labels", start, count, data));
}

TEST_F(NetCDFFileTest, AddAttribute) {
    util::NetCDFFile file(test_file);
    int dim_id = file.AddDimension("time", 5);
    std::vector<int> dim_ids = {dim_id};
    EXPECT_NO_THROW(file.AddVariable<double>("position", dim_ids));

    EXPECT_NO_THROW(file.AddAttribute("position", "units", std::string("meters")));
}

TEST_F(NetCDFFileTest, OpenExistingFile) {
    {
        util::NetCDFFile file(test_file);
        file.AddDimension("time", 5);
        file.AddVariable<double>("position", {file.GetDimensionId("time")});
    }

    EXPECT_NO_THROW({
        util::NetCDFFile file(test_file, false);
        EXPECT_NE(file.GetNetCDFId(), -1);
        EXPECT_NO_THROW(file.GetDimensionId("time"));
        EXPECT_NO_THROW(file.GetVariableId("position"));
    });
}

TEST_F(NetCDFFileTest, GetNumberOfDimensions) {
    util::NetCDFFile file(test_file);

    int time_dim = file.AddDimension("time", 5);
    int space_dim = file.AddDimension("nodes", 3);
    std::vector<int> dim_ids = {time_dim, space_dim};
    file.AddVariable<double>("position_1D", {time_dim});
    file.AddVariable<double>("position_2D", dim_ids);

    EXPECT_EQ(file.GetNumberOfDimensions("position_1D"), 1);
    EXPECT_EQ(file.GetNumberOfDimensions("position_2D"), 2);
}

TEST_F(NetCDFFileTest, GetDimensionLength) {
    util::NetCDFFile file(test_file);

    int time_dim = file.AddDimension("time", 5);
    int space_dim = file.AddDimension("nodes", 3);

    EXPECT_EQ(file.GetDimensionLength(time_dim), 5);
    EXPECT_EQ(file.GetDimensionLength(space_dim), 3);

    EXPECT_EQ(file.GetDimensionLength("time"), 5);
    EXPECT_EQ(file.GetDimensionLength("nodes"), 3);
}

TEST_F(NetCDFFileTest, GetShape) {
    util::NetCDFFile file(test_file);

    int time_dim = file.AddDimension("time", 5);
    int space_dim = file.AddDimension("nodes", 3);

    file.AddVariable<double>("position_1D", {time_dim});
    std::vector<size_t> expected_shape_1D = {5};
    EXPECT_EQ(file.GetShape("position_1D"), expected_shape_1D);

    std::vector<int> dim_ids_2D = {time_dim, space_dim};
    file.AddVariable<double>("position_2D", dim_ids_2D);
    std::vector<size_t> expected_shape_2D = {5, 3};
    EXPECT_EQ(file.GetShape("position_2D"), expected_shape_2D);
}

TEST_F(NetCDFFileTest, ReadVariableWithTypeDouble) {
    util::NetCDFFile file(test_file);
    int dim_id = file.AddDimension("time", 5);
    std::vector<int> dim_ids = {dim_id};

    file.AddVariable<double>("position", dim_ids);
    std::vector<double> write_data = {1.1, 2.2, 3.3, 4.4, 5.5};
    file.WriteVariable("position", write_data);

    std::vector<double> read_data(5);
    EXPECT_NO_THROW(file.ReadVariable("position", read_data.data()));
    EXPECT_EQ(read_data, write_data);
}

TEST_F(NetCDFFileTest, ReadVariableWithTypeFloat) {
    util::NetCDFFile file(test_file);
    int dim_id = file.AddDimension("time", 5);
    std::vector<int> dim_ids = {dim_id};

    file.AddVariable<float>("velocity", dim_ids);
    std::vector<float> write_data = {1.f, 2.f, 3.f, 4.f, 5.f};
    file.WriteVariable("velocity", write_data);

    std::vector<float> read_data(5);
    EXPECT_NO_THROW(file.ReadVariable("velocity", read_data.data()));
    EXPECT_EQ(read_data, write_data);
}

TEST_F(NetCDFFileTest, ReadVariableWithTypeInt) {
    util::NetCDFFile file(test_file);
    int dim_id = file.AddDimension("time", 5);
    std::vector<int> dim_ids = {dim_id};

    file.AddVariable<int>("count", dim_ids);
    std::vector<int> write_data = {1, 2, 3, 4, 5};
    file.WriteVariable("count", write_data);

    std::vector<int> read_data(5);
    EXPECT_NO_THROW(file.ReadVariable("count", read_data.data()));
    EXPECT_EQ(read_data, write_data);
}

TEST_F(NetCDFFileTest, ReadVariableAtWithTypeDouble) {
    util::NetCDFFile file(test_file);
    int dim_id = file.AddDimension("time", 10);
    std::vector<int> dim_ids = {dim_id};
    file.AddVariable<double>("position", dim_ids);

    // Write full data
    std::vector<double> write_data = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.};
    file.WriteVariable("position", write_data);

    // Read partial data
    std::vector<double> read_data(3);
    std::vector<size_t> start = {2};
    std::vector<size_t> count = {3};
    EXPECT_NO_THROW(file.ReadVariableAt("position", start, count, read_data.data()));
    EXPECT_EQ(read_data, std::vector<double>({3.3, 4.4, 5.5}));
}

TEST_F(NetCDFFileTest, ReadVariableAtWithTypeFloat) {
    util::NetCDFFile file(test_file);
    int dim_id = file.AddDimension("time", 10);
    std::vector<int> dim_ids = {dim_id};
    file.AddVariable<float>("velocity", dim_ids);

    // Write full data
    std::vector<float> write_data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f};
    file.WriteVariable("velocity", write_data);

    // Read partial data
    std::vector<float> read_data(3);
    std::vector<size_t> start = {4};
    std::vector<size_t> count = {3};
    EXPECT_NO_THROW(file.ReadVariableAt("velocity", start, count, read_data.data()));
    EXPECT_EQ(read_data, std::vector<float>({5.f, 6.f, 7.f}));
}

TEST_F(NetCDFFileTest, ReadVariableAtWithTypeInt) {
    util::NetCDFFile file(test_file);
    int dim_id = file.AddDimension("time", 10);
    std::vector<int> dim_ids = {dim_id};
    file.AddVariable<int>("count", dim_ids);

    // Write full data
    std::vector<int> write_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    file.WriteVariable("count", write_data);

    // Read partial data
    std::vector<int> read_data(3);
    std::vector<size_t> start = {6};
    std::vector<size_t> count = {3};
    EXPECT_NO_THROW(file.ReadVariableAt("count", start, count, read_data.data()));
    EXPECT_EQ(read_data, std::vector<int>({7, 8, 9}));
}

TEST_F(NetCDFFileTest, ReadVariableWithStrideTypeDouble) {
    util::NetCDFFile file(test_file);
    int dim_id = file.AddDimension("time", 10);
    std::vector<int> dim_ids = {dim_id};
    file.AddVariable<double>("position", dim_ids);

    // Write full data
    std::vector<double> write_data = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.};
    file.WriteVariable("position", write_data);

    // Read every other element starting from index 1
    std::vector<double> read_data(3);
    std::vector<size_t> start = {1};
    std::vector<size_t> count = {3};
    std::vector<ptrdiff_t> stride = {2};
    EXPECT_NO_THROW(file.ReadVariableWithStride("position", start, count, stride, read_data.data()));
    EXPECT_EQ(read_data, std::vector<double>({2.2, 4.4, 6.6}));
}

TEST_F(NetCDFFileTest, ReadVariableWithStrideTypeFloat) {
    util::NetCDFFile file(test_file);
    int dim_id = file.AddDimension("time", 10);
    std::vector<int> dim_ids = {dim_id};
    file.AddVariable<float>("velocity", dim_ids);

    // Write full data
    std::vector<float> write_data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f};
    file.WriteVariable("velocity", write_data);

    // Read every third element starting from index 0
    std::vector<float> read_data(3);
    std::vector<size_t> start = {0};
    std::vector<size_t> count = {3};
    std::vector<ptrdiff_t> stride = {3};
    EXPECT_NO_THROW(file.ReadVariableWithStride("velocity", start, count, stride, read_data.data()));
    EXPECT_EQ(read_data, std::vector<float>({1.f, 4.f, 7.f}));
}

TEST_F(NetCDFFileTest, ReadVariableWithStrideTypeInt) {
    util::NetCDFFile file(test_file);
    int dim_id = file.AddDimension("time", 10);
    std::vector<int> dim_ids = {dim_id};
    file.AddVariable<int>("count", dim_ids);

    // Write full data
    std::vector<int> write_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    file.WriteVariable("count", write_data);

    // Read every fourth element starting from index 2
    std::vector<int> read_data(2);
    std::vector<size_t> start = {2};
    std::vector<size_t> count = {2};
    std::vector<ptrdiff_t> stride = {4};
    EXPECT_NO_THROW(file.ReadVariableWithStride("count", start, count, stride, read_data.data()));
    EXPECT_EQ(read_data, std::vector<int>({3, 7}));
}

}  // namespace openturbine::tests
