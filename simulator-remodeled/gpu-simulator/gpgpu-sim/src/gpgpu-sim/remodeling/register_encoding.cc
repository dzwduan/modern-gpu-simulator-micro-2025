// Copyright (c) 2023-2025, Rodrigo Huerta, Mojtaba Abaie Shoushtary, Josep-Llorenç Cruz, Antonio González
// Universitat Politecnica de Catalunya
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The Universitat Politecnica de Catalunya nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "register_encoding.h"

#include <string>

namespace remodel {

unsigned int translate_warp_id_of_sm_to_subcore(unsigned int warp_id,
                                                unsigned int num_subcores) {
  return warp_id / num_subcores;
}

TraceEnhancedOperandType get_reg_type_eval(traced_operand& op) {
  TraceEnhancedOperandType reg_type = op.get_operand_type();
  if( (reg_type == TraceEnhancedOperandType::MREF) || (reg_type == TraceEnhancedOperandType::CBANK) || (reg_type == TraceEnhancedOperandType::DESC)) {
    if(op.get_operand_string().find("UR") != std::string::npos) {
      reg_type = TraceEnhancedOperandType::UREG;
    }else if(op.get_operand_string().find("R") != std::string::npos) {
      reg_type = TraceEnhancedOperandType::REG;
    }
  }
  return reg_type;
}

bool check_is_reserved_regs_remodeling(int reg, TraceEnhancedOperandType reg_type, bool is_trace_mode) {
  bool res = false;
  if(is_trace_mode) {
    if(reg_type == TraceEnhancedOperandType::REG) {
      res = (reg == RESERVED_REG_NUMBER);
    }else if(reg_type == TraceEnhancedOperandType::UREG) {
      res = (reg == RESERVED_UREG_NUMBER);
    }else if(reg_type == TraceEnhancedOperandType::PRED) {
      res = (reg == RESERVED_PRED_NUMBER);
    }else if(reg_type == TraceEnhancedOperandType::UPRED) {
      res = (reg == RESERVED_UPRED_NUMBER);
    }
  }
  return res;
}

unsigned int translate_reg_to_global_id(int reg, TraceEnhancedOperandType reg_type) {
  unsigned int global_id = 0;
  if(reg_type == TraceEnhancedOperandType::REG) {
    global_id = reg;
  }else if(reg_type == TraceEnhancedOperandType::UREG) {
    global_id = GLOBAL_ID_BASE_UREG + reg;
  }else if(reg_type == TraceEnhancedOperandType::PRED) {
    global_id = GLOBAL_ID_BASE_PRED;
    if(reg != PR) {
      global_id += reg;
    }
  }else if(reg_type == TraceEnhancedOperandType::UPRED) {
    global_id = GLOBAL_ID_BASE_UPRED;
    if(reg != UPR) {
      global_id += reg;
    }
  }
  return global_id;
}

} // namespace remodel
