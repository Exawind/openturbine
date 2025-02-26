#include <filesystem>

#include <gtest/gtest.h>

#include "utilities/outputs/netcdf_utilities.hpp"

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
        NetCDFFile file(test_file);
        EXPECT_NE(file.GetNetCDFId(), -1);
    });
    EXPECT_TRUE(std::filesystem::exists(test_file));
}

TEST_F(NetCDFFileTest, AddDimension) {
    NetCDFFile file(test_file);
    EXPECT_NO_THROW({
        int dim_id = file.AddDimension("test_dim", 10);
        EXPECT_GE(dim_id, 0);
        EXPECT_EQ(file.GetDimensionId("test_dim"), dim_id);
    });
}

TEST_F(NetCDFFileTest, AddAndWriteVariable) {
    NetCDFFile file(test_file);

    int dim_id = file.AddDimension("time", 5);
    std::vector<int> dim_ids = {dim_id};
    EXPECT_NO_THROW(file.AddVariable<double>("position", dim_ids));

    std::vector<double> data = {1., 2., 3., 4., 5.};
    EXPECT_NO_THROW(file.WriteVariable("position", data));
}

TEST_F(NetCDFFileTest, AddAndWriteVariableAt) {
    NetCDFFile file(test_file);

    int dim_id = file.AddDimension("time", 10);
    std::vector<int> dim_ids = {dim_id};
    EXPECT_NO_THROW(file.AddVariable<double>("position", dim_ids));

    std::vector<double> data = {1., 2., 3.};  // Add data for only 3 elements
    std::vector<size_t> start = {2};          // Start at index 2
    std::vector<size_t> count = {3};          // Write 3 elements
    EXPECT_NO_THROW(file.WriteVariableAt("position", start, count, data));
}

TEST_F(NetCDFFileTest, AddAttribute) {
    NetCDFFile file(test_file);

    int dim_id = file.AddDimension("time", 5);
    std::vector<int> dim_ids = {dim_id};
    EXPECT_NO_THROW(file.AddVariable<double>("position", dim_ids));

    EXPECT_NO_THROW(file.AddAttribute("position", "units", std::string("meters")));
}

TEST_F(NetCDFFileTest, OpenExistingFile) {
    {
        NetCDFFile file(test_file);
        file.AddDimension("time", 5);
        file.AddVariable<double>("position", {file.GetDimensionId("time")});
    }

    EXPECT_NO_THROW({
        NetCDFFile file(test_file, false);
        EXPECT_NE(file.GetNetCDFId(), -1);
        EXPECT_NO_THROW(file.GetDimensionId("time"));
        EXPECT_NO_THROW(file.GetVariableId("position"));
    });
}

}  // namespace openturbine::tests
