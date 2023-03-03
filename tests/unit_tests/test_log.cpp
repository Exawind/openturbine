#include "gtest/gtest.h"

#include "src/utilities/log.h"

namespace oturb_tests {

using namespace openturbine::util;

TEST(SeverityLevelTest, DefaultValue) {
    // Test that the default value of a SeverityLevel object is kNone
    SeverityLevel level{};
    EXPECT_EQ(level, SeverityLevel::kNone);
}

TEST(SeverityLevelTest, Comparison) {
    // Test that different severity levels can be compared
    EXPECT_LT(SeverityLevel::kNone, SeverityLevel::kError);
    EXPECT_LT(SeverityLevel::kError, SeverityLevel::kWarning);
    EXPECT_LT(SeverityLevel::kWarning, SeverityLevel::kInfo);
    EXPECT_LT(SeverityLevel::kInfo, SeverityLevel::kDebug);
    EXPECT_EQ(SeverityLevel::kError, SeverityLevel::kError);
    EXPECT_NE(SeverityLevel::kError, SeverityLevel::kWarning);
}

TEST(SeverityLevelTest, CastToInt) {
    // Test that the underlying integer value of each severity level is correct
    EXPECT_EQ(static_cast<int>(SeverityLevel::kNone), 0);
    EXPECT_EQ(static_cast<int>(SeverityLevel::kError), 1);
    EXPECT_EQ(static_cast<int>(SeverityLevel::kWarning), 2);
    EXPECT_EQ(static_cast<int>(SeverityLevel::kInfo), 3);
    EXPECT_EQ(static_cast<int>(SeverityLevel::kDebug), 4);
}

TEST(SeverityLevelToStringTest, Debug) {
    // Test that the function returns "DEBUG" for the kDebug level
    SeverityLevel level = SeverityLevel::kDebug;
    std::string expected = "DEBUG";
    std::string actual = SeverityLevelToString(level);
    EXPECT_EQ(actual, expected);
}

TEST(SeverityLevelToStringTest, Error) {
    // Test that the function returns "ERROR" for the kError level
    SeverityLevel level = SeverityLevel::kError;
    std::string expected = "ERROR";
    std::string actual = SeverityLevelToString(level);
    EXPECT_EQ(actual, expected);
}

TEST(SeverityLevelToStringTest, Info) {
    // Test that the function returns "INFO" for the kInfo level
    SeverityLevel level = SeverityLevel::kInfo;
    std::string expected = "INFO";
    std::string actual = SeverityLevelToString(level);
    EXPECT_EQ(actual, expected);
}

TEST(SeverityLevelToStringTest, Warning) {
    // Test that the function returns "WARNING" for the kWarning level
    SeverityLevel level = SeverityLevel::kWarning;
    std::string expected = "WARNING";
    std::string actual = SeverityLevelToString(level);
    EXPECT_EQ(actual, expected);
}

TEST(SeverityLevelToStringTest, None) {
    // Test that the function returns "NONE" for an invalid level
    SeverityLevel level = static_cast<SeverityLevel>(-1);  // Invalid level
    std::string expected = "NONE";
    std::string actual = SeverityLevelToString(level);
    EXPECT_EQ(actual, expected);
}

TEST(OutputTypeTest, Comparison) {
    // Test that different severity levels can be compared
    EXPECT_LT(OutputType::kConsole, OutputType::kFile);
    EXPECT_LT(OutputType::kFile, OutputType::kConsoleAndFile);
    EXPECT_LT(OutputType::kConsole, OutputType::kConsoleAndFile);
    EXPECT_EQ(OutputType::kConsoleAndFile, OutputType::kConsoleAndFile);
    EXPECT_NE(OutputType::kConsoleAndFile, OutputType::kFile);
}

TEST(OutputTypeTest, CastToInt) {
    // Test that the underlying integer value of each severity level is correct
    EXPECT_EQ(static_cast<int>(OutputType::kConsole), 0);
    EXPECT_EQ(static_cast<int>(OutputType::kFile), 1);
    EXPECT_EQ(static_cast<int>(OutputType::kConsoleAndFile), 2);
}

TEST(LogTest, GetInstance) {
    // get a log instance with default parameters
    Log* log_instance = Log::Get();
    EXPECT_TRUE(log_instance != nullptr);
    EXPECT_EQ(log_instance->GetMaxSeverityLevel(), SeverityLevel::kDebug);
    EXPECT_EQ(log_instance->GetOutputType(), OutputType::kConsoleAndFile);
    EXPECT_EQ(log_instance->GetOutputFileName(), "log.txt");

    // get the same instance on a second call
    Log* second_log_instance = Log::Get();
    EXPECT_TRUE(second_log_instance != nullptr);
    EXPECT_EQ(second_log_instance, log_instance);
}

}  // namespace oturb_tests
