#include <gtest/gtest.h>

#include "src/rigid_pendulum_poc/time_stepper.h"

namespace openturbine::rigid_pendulum::tests {

TEST(TimeStepperTest, CreateDefaultTimeStepper) {
    auto time_stepper = TimeStepper();

    EXPECT_EQ(time_stepper.GetInitialTime(), 0.);
    EXPECT_EQ(time_stepper.GetCurrentTime(), 0.);
    EXPECT_EQ(time_stepper.GetTimeStep(), 1.);
    EXPECT_EQ(time_stepper.GetNumberOfSteps(), 1);
}

TEST(TimeStepperTest, CreateTimeStepper) {
    auto time_stepper = TimeStepper(1., 0.01, 10);

    EXPECT_EQ(time_stepper.GetInitialTime(), 1.);
    EXPECT_EQ(time_stepper.GetCurrentTime(), 1.);
    EXPECT_EQ(time_stepper.GetTimeStep(), 0.01);
    EXPECT_EQ(time_stepper.GetNumberOfSteps(), 10);
}

TEST(TimeStepperTest, ConstructorWithInvalidNumberOfSteps) {
    EXPECT_THROW(TimeStepper(1., 0.01, 1, 0), std::invalid_argument);
}

TEST(TimeStepperTest, AdvanceAnalysisTime) {
    auto time_stepper = TimeStepper();

    EXPECT_EQ(time_stepper.GetCurrentTime(), 0.);

    time_stepper.AdvanceTimeStep();
    EXPECT_EQ(time_stepper.GetCurrentTime(), 1.);
}

TEST(TimeStepperTest, GetNumberOfIterations) {
    auto time_stepper = TimeStepper();

    EXPECT_EQ(time_stepper.GetNumberOfIterations(), 0);
}

TEST(TimeStepperTest, SetNumberOfIterations) {
    auto time_stepper = TimeStepper();

    time_stepper.SetNumberOfIterations(10);
    EXPECT_EQ(time_stepper.GetNumberOfIterations(), 10);
}

TEST(TimeStepperTest, IncrementNumberOfIterations) {
    auto time_stepper = TimeStepper();

    EXPECT_EQ(time_stepper.GetNumberOfIterations(), 0);

    time_stepper.IncrementNumberOfIterations();
    EXPECT_EQ(time_stepper.GetNumberOfIterations(), 1);
}

TEST(TimeStepperTest, GetTotalNumberOfIterations) {
    auto time_stepper = TimeStepper();

    EXPECT_EQ(time_stepper.GetTotalNumberOfIterations(), 0);
}

TEST(TimeStepperTest, IncrementTotalNumberOfIterations) {
    auto time_stepper = TimeStepper();

    EXPECT_EQ(time_stepper.GetTotalNumberOfIterations(), 0);

    time_stepper.IncrementTotalNumberOfIterations(10);
    EXPECT_EQ(time_stepper.GetTotalNumberOfIterations(), 10);
}

TEST(TimeStepperTest, GetMaximumNumberOfIterations) {
    auto time_stepper = TimeStepper();

    EXPECT_EQ(time_stepper.GetMaximumNumberOfIterations(), 10);
}

}  // namespace openturbine::rigid_pendulum::tests
