# RTX4090 Config Regeneration Design

**Goal:** Rebuild the `SM89_RTX4090` GPGPU-Sim configuration from the stable `SM86_RTXA6000` baseline while preserving only the hardware identity differences required for RTX 4090 / Ada.

**Context**

The current repository already treats RTX 4090 as a supported standard config, but the safest regeneration strategy is to inherit the proven `SM86_RTXA6000` remodeling/cache pipeline settings and override only the card-specific hardware knobs.

**Chosen Approach**

Use `SM86_RTXA6000/gpgpusim.config` as the canonical baseline for all stable modeling parameters:

- frontend/remodeling parameters
- L0I/L1/shared-memory/cache pipeline settings
- memory scheduler and queue topology
- interconnect and scoreboard settings

Override only these RTX 4090 hardware identity parameters:

- `-gpgpu_ptx_force_max_capability 89`
- `-gpgpu_compute_capability_major/minor 8.9`
- `-gpgpu_n_clusters 128`
- `-gpgpu_occupancy_sm_number 89`
- `-gpgpu_coalesce_arch 89`
- `-gpgpu_clock_domains 2520:2520:2520:10500`
- RTX 4090 L2 sizing line

**Why This Approach**

- It minimizes risk by not retuning the stable A6000-derived microarchitectural parameters.
- It keeps the 4090 config aligned with the repository's existing Ada support path.
- It makes future diffs against `SM86_RTXA6000` easy to reason about.

**Validation**

Validation is diff-based rather than unit-test-based because this is a configuration regeneration task:

- compare regenerated `SM89_RTX4090/gpgpusim.config` against `SM86_RTXA6000/gpgpusim.config`
- confirm the diff is limited to the intended RTX 4090 hardware identity fields
- confirm `config_ampere_islip.icnt` and `accelwattch_sass_sim.xml` remain unchanged
