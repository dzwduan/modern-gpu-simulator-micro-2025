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

#pragma once

// Pure encoding helpers between the trace operand representation and the
// simulator's flat register/warp numbering. They depend only on the trace
// operand types, so they carry no SM, configuration or simulator state.

#include "../../../../../util/traces_enhanced/src/traced_operand.h"

namespace remodel {

unsigned int translate_warp_id_of_sm_to_subcore(unsigned int warp_id, unsigned int num_subcores);

TraceEnhancedOperandType get_reg_type_eval(traced_operand& op);

// Trace encodings of the architectural discard registers RZ, URZ, PT and UPT.
// Reads of them return a constant and writes to them are dropped, so they carry
// no dependence and are excluded from scoreboard tracking.
constexpr int RESERVED_REG_NUMBER = 255;
constexpr int RESERVED_UREG_NUMBER = 63;
constexpr int RESERVED_PRED_NUMBER = 7;
constexpr int RESERVED_UPRED_NUMBER = 7;

bool check_is_reserved_regs_remodeling(int reg, TraceEnhancedOperandType reg_type, bool is_trace_mode);

// Bases of the flat register id space shared by all four register files:
// regular registers occupy [0, 256), uniform registers start at 256,
// predicates at 512 and uniform predicates at 520.
constexpr unsigned int GLOBAL_ID_BASE_UREG = 256;
constexpr unsigned int GLOBAL_ID_BASE_PRED = 512;
constexpr unsigned int GLOBAL_ID_BASE_UPRED = 520;

unsigned int translate_reg_to_global_id(int reg, TraceEnhancedOperandType reg_type);

} // namespace remodel
