# arch.md True-Path Rewrite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `arch.md` with a source-accurate, single-path architecture document for remodeling under the agreed assumptions (control-bit dependency path).

**Architecture:** Rebuild the document from runtime execution order outward: `SM::cycle()` -> `Subcore::cycle()` -> dependency control bits -> instruction supply -> shared memory unit lifecycle. Remove all optional-branch narrative and keep only the effective path under the approved assumptions.

**Tech Stack:** Markdown, ripgrep (`rg`), source cross-check against C++ files in `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling`.

---

### Task 1: Build a Failing Consistency Gate for Current `arch.md`

**Files:**
- Modify: `arch.md`
- Test: `arch.md` (regex gate via shell commands)

**Step 1: Write the failing test**

```bash
! rg -n "m_dispatch_latches|4 种寄存器文件|下面开始头脑风暴" /home/duanzhenwei/modern-gpu-simulator-micro-2025/arch.md
```

**Step 2: Run test to verify it fails**

Run:
```bash
! rg -n "m_dispatch_latches|4 种寄存器文件|下面开始头脑风暴" /home/duanzhenwei/modern-gpu-simulator-micro-2025/arch.md
```
Expected: FAIL (old inconsistent/polluted content exists before rewrite)

**Step 3: Write minimal implementation**

Rewrite `arch.md` skeleton to include:
```markdown
1. Scope & Assumptions
2. SM Top-Level Execution Order
3. Subcore Pipeline
4. Dependency Model (control-bit)
5. Instruction Supply Path
6. Execution Resources
7. Shared Memory Pipeline
8. Effective Configuration Matrix
9. Source Anchors
```

**Step 4: Run test to verify it passes**

Run:
```bash
! rg -n "m_dispatch_latches|4 种寄存器文件|下面开始头脑风暴" /home/duanzhenwei/modern-gpu-simulator-micro-2025/arch.md
```
Expected: PASS

**Step 5: Commit**

```bash
git add /home/duanzhenwei/modern-gpu-simulator-micro-2025/arch.md
git commit -m "docs(arch): rewrite remodeling true-path architecture"
```

### Task 2: Encode Control-Bit Dependency Path as Single Truth

**Files:**
- Modify: `arch.md`
- Test: `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/{sm.cc,subcore.cc,functional_unit.cc}`

**Step 1: Write the failing test**

```bash
rg -n "scoreboard|control-bit|wait barrier|yield|stall" /home/duanzhenwei/modern-gpu-simulator-micro-2025/arch.md
```

Define failure as: missing explicit statement of assumptions
- `is_remodeling_scoreboarding_enabled=false`
- `is_trace_mode=true`
- `is_captured_from_binary=true`

**Step 2: Run test to verify it fails**

Run:
```bash
rg -n "is_remodeling_scoreboarding_enabled=false|is_trace_mode=true|is_captured_from_binary=true" /home/duanzhenwei/modern-gpu-simulator-micro-2025/arch.md
```
Expected: FAIL if any assumption line is missing

**Step 3: Write minimal implementation**

Add dependency section with source-backed lifecycle:
```markdown
- Issue readiness gates: stall/yield/wait barrier
- Control stage: add pending read/write wait-barrier increments
- FU read-barrier release: pending read wait-barrier decrement
- Retirement: pending write wait-barrier decrement + ldgsts counter update
```

**Step 4: Run test to verify it passes**

Run:
```bash
rg -n "is_remodeling_scoreboarding_enabled=false|is_trace_mode=true|is_captured_from_binary=true" /home/duanzhenwei/modern-gpu-simulator-micro-2025/arch.md
```
Expected: PASS

**Step 5: Commit**

```bash
git add /home/duanzhenwei/modern-gpu-simulator-micro-2025/arch.md
git commit -m "docs(arch): document control-bit dependency true path"
```

### Task 3: Add Source Anchors and Run Final Verification

**Files:**
- Modify: `arch.md`
- Test: `arch.md` + remodeling source files

**Step 1: Write the failing test**

```bash
rg -n "Source Anchors|源码锚点" /home/duanzhenwei/modern-gpu-simulator-micro-2025/arch.md
```

Failure condition: no anchor section or no function-level anchors.

**Step 2: Run test to verify it fails**

Run:
```bash
rg -n "SM::cycle|Subcore::cycle|ldst_unit_sm::cycle|PendingRequestTable|IBuffer_Remodeled" /home/duanzhenwei/modern-gpu-simulator-micro-2025/arch.md
```
Expected: FAIL if any required anchor token missing

**Step 3: Write minimal implementation**

Add compact anchor map per section:
```markdown
- sm.cc: SM::cycle, SM::instruction_retirement
- subcore.cc: Subcore::cycle, Subcore::issue, Subcore::control_stage
- functional_unit.cc: functional_unit::release_read_barrier
- ldst_unit_sm.cc: ldst_unit_sm::cycle, ldst_unit_sm::issue
```

**Step 4: Run test to verify it passes**

Run:
```bash
rg -n "SM::cycle|Subcore::cycle|ldst_unit_sm::cycle|PendingRequestTable|IBuffer_Remodeled" /home/duanzhenwei/modern-gpu-simulator-micro-2025/arch.md
```
Expected: PASS

Then run @superpowers:verification-before-completion style checks:
```bash
rg -n "可选|否则|关闭时|if disabled" /home/duanzhenwei/modern-gpu-simulator-micro-2025/arch.md
rg -n "下面开始头脑风暴" /home/duanzhenwei/modern-gpu-simulator-micro-2025/arch.md
```
Expected:
- no branch-style narrative matches
- no pollution line

**Step 5: Commit**

```bash
git add /home/duanzhenwei/modern-gpu-simulator-micro-2025/arch.md
git commit -m "docs(arch): add source anchors and final consistency checks"
```
