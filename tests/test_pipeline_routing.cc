#include <gtest/gtest.h>
#include "pipeline_routing.h"

TEST(IntPredicateRouting, UnifiedGoesToSP) {
  EXPECT_EQ(resolve_int_predicate_target(true), PipelineTarget::SP);
}

TEST(IntPredicateRouting, SeparateGoesToINT) {
  EXPECT_EQ(resolve_int_predicate_target(false), PipelineTarget::INT);
}

TEST(SpOpRouting, DefaultGoesToSP) {
  EXPECT_EQ(resolve_sp_op_target(false, false, false), PipelineTarget::SP);
}

TEST(SpOpRouting, FP32AllowedAndIntAvailableGoesToINT) {
  EXPECT_EQ(resolve_sp_op_target(true, true, false), PipelineTarget::INT);
}

TEST(SpOpRouting, FP32AllowedButIntBusyGoesToSP) {
  EXPECT_EQ(resolve_sp_op_target(true, false, false), PipelineTarget::SP);
}

TEST(SpOpRouting, IMADAlwaysGoesToSP) {
  EXPECT_EQ(resolve_sp_op_target(true, true, true), PipelineTarget::SP);
}

TEST(SpOpRouting, FP32NotAllowedGoesToSP) {
  EXPECT_EQ(resolve_sp_op_target(false, true, false), PipelineTarget::SP);
}
