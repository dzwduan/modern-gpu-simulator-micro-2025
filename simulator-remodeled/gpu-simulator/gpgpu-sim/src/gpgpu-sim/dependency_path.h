#ifndef DEPENDENCY_PATH_H
#define DEPENDENCY_PATH_H

inline bool uses_control_bit_dependency(bool is_trace_mode,
                                        bool is_captured_from_binary,
                                        bool is_remodeling_scoreboarding_enabled) {
  return is_trace_mode && is_captured_from_binary &&
         !is_remodeling_scoreboarding_enabled;
}

inline bool uses_trace_mode_scoreboard(bool is_trace_mode,
                                       bool is_captured_from_binary,
                                       bool is_remodeling_scoreboarding_enabled) {
  return is_trace_mode &&
         (!is_captured_from_binary || is_remodeling_scoreboarding_enabled);
}

#endif
