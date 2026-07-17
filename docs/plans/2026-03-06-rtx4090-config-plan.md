# RTX4090 Config Regeneration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Regenerate the RTX 4090 GPGPU-Sim config from the stable RTXA6000 baseline with only RTX 4090 hardware identity overrides.

**Architecture:** Reuse the `SM86_RTXA6000` config as the stable modeling skeleton. Rewrite `SM89_RTX4090/gpgpusim.config` so that it keeps A6000's proven remodeling/cache settings and changes only compute capability, SM count, clocks, and L2 sizing for RTX 4090.

**Tech Stack:** GPGPU-Sim config files, shell diff/grep validation

---

### Task 1: Document the regeneration rule

**Files:**
- Create: `docs/plans/2026-03-06-rtx4090-config-design.md`
- Create: `docs/plans/2026-03-06-rtx4090-config-plan.md`

**Step 1: Write the design note**

Document that `SM86_RTXA6000` is the baseline and enumerate the allowed RTX 4090 overrides.

**Step 2: Verify the files exist**

Run: `ls docs/plans`
Expected: both new plan files are listed

### Task 2: Rewrite the RTX4090 config

**Files:**
- Modify: `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM89_RTX4090/gpgpusim.config`
- Reference: `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTXA6000/gpgpusim.config`

**Step 1: Rewrite from the RTXA6000 baseline**

Copy the stable A6000 parameter layout and keep only these RTX 4090-specific overrides:

- `-gpgpu_ptx_force_max_capability 89`
- `-gpgpu_compute_capability_minor 9`
- `-gpgpu_n_clusters 128`
- `-gpgpu_clock_domains 2520:2520:2520:10500`
- `-gpgpu_occupancy_sm_number 89`
- `-gpgpu_coalesce_arch 89`
- RTX 4090 L2 cache sizing line

**Step 2: Preserve non-config siblings**

Leave these files untouched unless separate evidence requires changes:

- `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM89_RTX4090/config_ampere_islip.icnt`
- `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM89_RTX4090/accelwattch_sass_sim.xml`

### Task 3: Structural verification

**Files:**
- Verify: `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM89_RTX4090/gpgpusim.config`
- Reference: `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTXA6000/gpgpusim.config`

**Step 1: Check the expected diff**

Run:

```bash
diff -u gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTXA6000/gpgpusim.config \
  gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM89_RTX4090/gpgpusim.config
```

Expected: differences are limited to RTX 4090 identity parameters and explanatory comments.

**Step 2: Check critical fields directly**

Run:

```bash
rg -n "gpgpu_ptx_force_max_capability|gpgpu_compute_capability_minor|gpgpu_n_clusters|gpgpu_clock_domains|gpgpu_occupancy_sm_number|gpgpu_coalesce_arch|gpgpu_cache:dl2" \
  gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM89_RTX4090/gpgpusim.config
```

Expected: the file reports the intended RTX 4090 values.

**Step 3: Confirm sibling files were not changed**

Run:

```bash
diff -u gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTXA6000/config_ampere_islip.icnt \
  gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM89_RTX4090/config_ampere_islip.icnt
diff -u gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTXA6000/accelwattch_sass_sim.xml \
  gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM89_RTX4090/accelwattch_sass_sim.xml
```

Expected: no output.
