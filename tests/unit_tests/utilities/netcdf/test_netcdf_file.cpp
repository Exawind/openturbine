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

}  // namespace openturbine::tests
