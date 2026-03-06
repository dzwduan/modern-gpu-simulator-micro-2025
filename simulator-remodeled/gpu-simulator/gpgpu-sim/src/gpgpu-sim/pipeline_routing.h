#ifndef PIPELINE_ROUTING_H
#define PIPELINE_ROUTING_H

enum class PipelineTarget {
  SP,
  INT,
  DP,
  SFU,
  TENSOR,
  UNIFORM,
  BRANCH,
  MISC_QUEUE,
  MISC_NO_QUEUE,
  MEMORY
};

inline PipelineTarget resolve_int_predicate_target(bool is_unified) {
  return is_unified ? PipelineTarget::SP : PipelineTarget::INT;
}

inline PipelineTarget resolve_sp_op_target(bool is_fp32ops_allowed_in_int,
                                           bool int_can_issue, bool is_imad) {
  if (is_fp32ops_allowed_in_int && int_can_issue && !is_imad) {
    return PipelineTarget::INT;
  }
  return PipelineTarget::SP;
}

#endif
