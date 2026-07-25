// Unit tests for remodel::register_encoding.
//
// The functions under test are linked from the object file that the simulator
// build produces, so these cases pin the behaviour of the production code
// itself rather than of a copy.

#include <gtest/gtest.h>

#include "register_encoding.h"

namespace {

using remodel::check_is_reserved_regs_remodeling;
using remodel::get_reg_type_eval;
using remodel::translate_reg_to_global_id;
using remodel::translate_warp_id_of_sm_to_subcore;

using OperandType = TraceEnhancedOperandType;

TraceEnhancedOperandType eval(const char *operand_string) {
  traced_operand op(operand_string);
  return get_reg_type_eval(op);
}

// --- translate_warp_id_of_sm_to_subcore --------------------------------------
//
// Warps are dealt to subcores by `warp_id % num_subcores`; this helper returns
// the position the warp occupies inside its own subcore's slice.

TEST(TranslateWarpIdOfSmToSubcore, IndexesTheWarpWithinItsSubcoreSlice) {
  EXPECT_EQ(0u, translate_warp_id_of_sm_to_subcore(0, 4));
  EXPECT_EQ(0u, translate_warp_id_of_sm_to_subcore(3, 4));
  EXPECT_EQ(1u, translate_warp_id_of_sm_to_subcore(4, 4));
  EXPECT_EQ(1u, translate_warp_id_of_sm_to_subcore(7, 4));
  EXPECT_EQ(11u, translate_warp_id_of_sm_to_subcore(47, 4));
}

TEST(TranslateWarpIdOfSmToSubcore, IsIdentityForASingleSubcore) {
  EXPECT_EQ(0u, translate_warp_id_of_sm_to_subcore(0, 1));
  EXPECT_EQ(17u, translate_warp_id_of_sm_to_subcore(17, 1));
}

TEST(TranslateWarpIdOfSmToSubcore, TruncatesWhenTheWarpCountIsNotDivisible) {
  // Six warps over four subcores: the last slice is partial, and the helper
  // still divides, so the two leftover warps land at index 1 of subcores 0/1.
  const unsigned int expected[6] = {0, 0, 0, 0, 1, 1};
  for (unsigned int warp_id = 0; warp_id < 6; warp_id++) {
    EXPECT_EQ(expected[warp_id], translate_warp_id_of_sm_to_subcore(warp_id, 4))
        << "warp_id " << warp_id;
  }
  // Three subcores is also not a divisor of a power-of-two warp count.
  EXPECT_EQ(2u, translate_warp_id_of_sm_to_subcore(7, 3));
  EXPECT_EQ(2u, translate_warp_id_of_sm_to_subcore(8, 3));
  EXPECT_EQ(3u, translate_warp_id_of_sm_to_subcore(9, 3));
}

// --- check_is_reserved_regs_remodeling ---------------------------------------

TEST(CheckIsReservedRegsRemodeling, DetectsTheDiscardRegisterOfEachFile) {
  EXPECT_TRUE(check_is_reserved_regs_remodeling(255, OperandType::REG, true));
  EXPECT_TRUE(check_is_reserved_regs_remodeling(63, OperandType::UREG, true));
  EXPECT_TRUE(check_is_reserved_regs_remodeling(7, OperandType::PRED, true));
  EXPECT_TRUE(check_is_reserved_regs_remodeling(7, OperandType::UPRED, true));
}

TEST(CheckIsReservedRegsRemodeling, AcceptsOrdinaryRegistersOfEachFile) {
  EXPECT_FALSE(check_is_reserved_regs_remodeling(0, OperandType::REG, true));
  EXPECT_FALSE(check_is_reserved_regs_remodeling(254, OperandType::REG, true));
  EXPECT_FALSE(check_is_reserved_regs_remodeling(0, OperandType::UREG, true));
  EXPECT_FALSE(check_is_reserved_regs_remodeling(62, OperandType::UREG, true));
  EXPECT_FALSE(check_is_reserved_regs_remodeling(0, OperandType::PRED, true));
  EXPECT_FALSE(check_is_reserved_regs_remodeling(6, OperandType::PRED, true));
  EXPECT_FALSE(check_is_reserved_regs_remodeling(0, OperandType::UPRED, true));
  EXPECT_FALSE(check_is_reserved_regs_remodeling(6, OperandType::UPRED, true));
}

TEST(CheckIsReservedRegsRemodeling, ReservedNumbersAreScopedToTheirFile) {
  EXPECT_FALSE(check_is_reserved_regs_remodeling(63, OperandType::REG, true));
  EXPECT_FALSE(check_is_reserved_regs_remodeling(7, OperandType::REG, true));
  EXPECT_FALSE(check_is_reserved_regs_remodeling(255, OperandType::UREG, true));
  EXPECT_FALSE(check_is_reserved_regs_remodeling(7, OperandType::UREG, true));
  EXPECT_FALSE(check_is_reserved_regs_remodeling(255, OperandType::PRED, true));
  EXPECT_FALSE(check_is_reserved_regs_remodeling(63, OperandType::PRED, true));
  EXPECT_FALSE(check_is_reserved_regs_remodeling(255, OperandType::UPRED, true));
  EXPECT_FALSE(check_is_reserved_regs_remodeling(63, OperandType::UPRED, true));
}

TEST(CheckIsReservedRegsRemodeling, OperandTypesWithoutARegisterFileAreNever) {
  const OperandType non_register_types[] = {
      OperandType::IMM_UINT64, OperandType::IMM_DOUBLE, OperandType::CBANK,
      OperandType::MREF,       OperandType::GENERIC,    OperandType::BREG,
      OperandType::SR,         OperandType::SB,         OperandType::DESC,
      OperandType::CALL_TARGET, OperandType::NONE};
  const int reserved_numbers[] = {255, 63, 7};
  for (OperandType type : non_register_types) {
    for (int reg : reserved_numbers) {
      EXPECT_FALSE(check_is_reserved_regs_remodeling(reg, type, true))
          << "type " << static_cast<int>(type) << " reg " << reg;
    }
  }
}

TEST(CheckIsReservedRegsRemodeling, OutsideTraceModeNothingIsReserved) {
  EXPECT_FALSE(check_is_reserved_regs_remodeling(255, OperandType::REG, false));
  EXPECT_FALSE(check_is_reserved_regs_remodeling(63, OperandType::UREG, false));
  EXPECT_FALSE(check_is_reserved_regs_remodeling(7, OperandType::PRED, false));
  EXPECT_FALSE(check_is_reserved_regs_remodeling(7, OperandType::UPRED, false));
  // The register number is irrelevant once trace mode is off.
  for (int reg = 0; reg < 256; reg++) {
    EXPECT_FALSE(check_is_reserved_regs_remodeling(reg, OperandType::REG, false))
        << "reg " << reg;
  }
}

// --- translate_reg_to_global_id ----------------------------------------------

TEST(TranslateRegToGlobalId, RegularRegistersKeepTheirNumber) {
  EXPECT_EQ(0u, translate_reg_to_global_id(0, OperandType::REG));
  EXPECT_EQ(1u, translate_reg_to_global_id(1, OperandType::REG));
  EXPECT_EQ(254u, translate_reg_to_global_id(254, OperandType::REG));
  EXPECT_EQ(255u, translate_reg_to_global_id(RZ, OperandType::REG));
}

TEST(TranslateRegToGlobalId, UniformRegistersStartAtTheUniformBase) {
  EXPECT_EQ(256u, translate_reg_to_global_id(0, OperandType::UREG));
  EXPECT_EQ(260u, translate_reg_to_global_id(4, OperandType::UREG));
  EXPECT_EQ(319u, translate_reg_to_global_id(URZ, OperandType::UREG));
}

TEST(TranslateRegToGlobalId, PredicatesStartAtThePredicateBase) {
  EXPECT_EQ(512u, translate_reg_to_global_id(0, OperandType::PRED));
  EXPECT_EQ(515u, translate_reg_to_global_id(3, OperandType::PRED));
  EXPECT_EQ(519u, translate_reg_to_global_id(PT, OperandType::PRED));
}

TEST(TranslateRegToGlobalId, UniformPredicatesStartAtTheUniformPredicateBase) {
  EXPECT_EQ(520u, translate_reg_to_global_id(0, OperandType::UPRED));
  EXPECT_EQ(523u, translate_reg_to_global_id(3, OperandType::UPRED));
  EXPECT_EQ(527u, translate_reg_to_global_id(UPT, OperandType::UPRED));
}

// PR / UPR are the trace encoding for "the whole predicate register file", not
// for one predicate. The helper drops the offset for them, so their id collides
// with predicate 0 of the same file. Pinned as observed, not corrected here.
TEST(TranslateRegToGlobalId, WholePredicateRegisterCollidesWithPredicateZero) {
  EXPECT_EQ(512u, translate_reg_to_global_id(PR, OperandType::PRED));
  EXPECT_EQ(translate_reg_to_global_id(0, OperandType::PRED),
            translate_reg_to_global_id(PR, OperandType::PRED));
  EXPECT_EQ(520u, translate_reg_to_global_id(UPR, OperandType::UPRED));
  EXPECT_EQ(translate_reg_to_global_id(0, OperandType::UPRED),
            translate_reg_to_global_id(UPR, OperandType::UPRED));
}

// Every operand type that owns no register file falls through to the initial
// zero, which is the same id as regular register 0. Pinned as observed.
TEST(TranslateRegToGlobalId, OperandTypesWithoutARegisterFileMapToZero) {
  const OperandType non_register_types[] = {
      OperandType::IMM_UINT64, OperandType::IMM_DOUBLE, OperandType::CBANK,
      OperandType::MREF,       OperandType::GENERIC,    OperandType::BREG,
      OperandType::SR,         OperandType::SB,         OperandType::DESC,
      OperandType::CALL_TARGET, OperandType::NONE};
  for (OperandType type : non_register_types) {
    EXPECT_EQ(0u, translate_reg_to_global_id(4, type))
        << "type " << static_cast<int>(type);
  }
}

TEST(TranslateRegToGlobalId, TheFourFileRangesDoNotOverlap) {
  EXPECT_LT(translate_reg_to_global_id(RZ, OperandType::REG),
            translate_reg_to_global_id(0, OperandType::UREG));
  EXPECT_LT(translate_reg_to_global_id(URZ, OperandType::UREG),
            translate_reg_to_global_id(0, OperandType::PRED));
  EXPECT_LT(translate_reg_to_global_id(PT, OperandType::PRED),
            translate_reg_to_global_id(0, OperandType::UPRED));
}

// --- get_reg_type_eval -------------------------------------------------------

TEST(GetRegTypeEval, PassesPlainRegisterOperandsThrough) {
  EXPECT_EQ(OperandType::REG, eval("R5"));
  EXPECT_EQ(OperandType::REG, eval("RZ"));
  EXPECT_EQ(OperandType::UREG, eval("UR4"));
  EXPECT_EQ(OperandType::UREG, eval("URZ"));
  EXPECT_EQ(OperandType::PRED, eval("P0"));
  EXPECT_EQ(OperandType::PRED, eval("PT"));
  EXPECT_EQ(OperandType::UPRED, eval("UP1"));
}

TEST(GetRegTypeEval, PassesOperandsWithoutARegisterFileThrough) {
  EXPECT_EQ(OperandType::IMM_UINT64, eval("0x1234"));
  EXPECT_EQ(OperandType::SR, eval("SR_TID.X"));
  EXPECT_EQ(OperandType::BREG, eval("B0"));
}

TEST(GetRegTypeEval, ResolvesMemoryReferencesToTheirAddressRegisterFile) {
  EXPECT_EQ(OperandType::REG, eval("[R2]"));
  EXPECT_EQ(OperandType::REG, eval("[R2+0x10]"));
  EXPECT_EQ(OperandType::UREG, eval("[UR4]"));
}

TEST(GetRegTypeEval, ResolvesConstantBankOperandsToTheirIndexRegisterFile) {
  EXPECT_EQ(OperandType::REG, eval("c[0x0][R4]"));
  EXPECT_EQ(OperandType::UREG, eval("c[0x0][UR4]"));
}

TEST(GetRegTypeEval, LeavesRegisterFreeConstantBankOperandsUnchanged) {
  EXPECT_EQ(OperandType::CBANK, eval("c[0x0][0x160]"));
}

// The resolution scans for "UR" before "R", so an operand that names both files
// resolves to the uniform one. `desc[UR4][R2.64]` is the common descriptor form
// of a global access whose address register is R2, and it still reports UREG.
// Pinned as observed, not corrected here.
TEST(GetRegTypeEval, UniformFileWinsWhenAnOperandNamesBothFiles) {
  EXPECT_EQ(OperandType::UREG, eval("[R2+UR4]"));
  EXPECT_EQ(OperandType::UREG, eval("desc[UR4][R2.64]"));
}

}  // namespace
