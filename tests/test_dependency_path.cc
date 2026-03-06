#include <gtest/gtest.h>
#include "dependency_path.h"

TEST(UsesControlBitDependency, TruePathReturnsTrue) {
  EXPECT_TRUE(uses_control_bit_dependency(true, true, false));
}

TEST(UsesControlBitDependency, NonBinaryReturnsFalse) {
  EXPECT_FALSE(uses_control_bit_dependency(true, false, false));
}

TEST(UsesControlBitDependency, ScoreboardEnabledReturnsFalse) {
  EXPECT_FALSE(uses_control_bit_dependency(true, true, true));
}

TEST(UsesControlBitDependency, ExecutionDrivenReturnsFalse) {
  EXPECT_FALSE(uses_control_bit_dependency(false, true, false));
}

TEST(UsesControlBitDependency, AllFalseReturnsFalse) {
  EXPECT_FALSE(uses_control_bit_dependency(false, false, false));
}

TEST(UsesTraceModeScoreboard, TruePathReturnsFalse) {
  EXPECT_FALSE(uses_trace_mode_scoreboard(true, true, false));
}

TEST(UsesTraceModeScoreboard, NonBinaryReturnsTrue) {
  EXPECT_TRUE(uses_trace_mode_scoreboard(true, false, false));
}

TEST(UsesTraceModeScoreboard, ScoreboardEnabledReturnsTrue) {
  EXPECT_TRUE(uses_trace_mode_scoreboard(true, true, true));
}

TEST(UsesTraceModeScoreboard, ExecutionDrivenReturnsFalse) {
  EXPECT_FALSE(uses_trace_mode_scoreboard(false, true, false));
}

TEST(UsesTraceModeScoreboard, MutualExclusivity) {
  bool modes[][3] = {
    {true, true, false},
    {true, false, false},
    {true, true, true},
    {false, true, false},
    {false, false, true},
    {true, false, true},
    {false, false, false},
    {false, true, true},
  };
  for (auto &m : modes) {
    bool cb = uses_control_bit_dependency(m[0], m[1], m[2]);
    bool sb = uses_trace_mode_scoreboard(m[0], m[1], m[2]);
    if (m[0]) {
      EXPECT_NE(cb, sb) << "trace_mode=true: control_bit and scoreboard must be mutually exclusive"
                         << " for captured=" << m[1] << " remod_sb=" << m[2];
    } else {
      EXPECT_FALSE(cb);
      EXPECT_FALSE(sb);
    }
  }
}
