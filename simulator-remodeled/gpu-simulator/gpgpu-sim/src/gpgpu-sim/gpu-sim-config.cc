

#include "../../libcuda/gpgpu_context.h"
#include "../abstract_hardware_model.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/cuda_device_runtime.h"
#include "gpu-sim.h"

void power_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-accelwattch_xml_file", OPT_CSTR,
                         &g_power_config_name, "AccelWattch XML file",
                         "accelwattch_sass_sim.xml");

  option_parser_register(opp, "-power_simulation_enabled", OPT_BOOL,
                         &g_power_simulation_enabled,
                         "Turn on power simulator (1=On, 0=Off)", "0");

  option_parser_register(opp, "-power_per_cycle_dump", OPT_BOOL,
                         &g_power_per_cycle_dump,
                         "Dump detailed power output each cycle", "0");




  option_parser_register(opp, "-hw_perf_file_name", OPT_CSTR,
                         &g_hw_perf_file_name, "Hardware Performance Statistics file",
                         "hw_perf.csv");

  option_parser_register(opp, "-hw_perf_bench_name", OPT_CSTR,
                         &g_hw_perf_bench_name, "Kernel Name in Hardware Performance Statistics file",
                         "");

  option_parser_register(opp, "-power_simulation_mode", OPT_INT32,
                         &g_power_simulation_mode,
                         "Switch performance counter input for power simulation (0=Sim, 1=HW, 2=HW-Sim Hybrid)", "0");

  option_parser_register(opp, "-dvfs_enabled", OPT_BOOL,
                         &g_dvfs_enabled,
                         "Turn on DVFS for power model", "0");
  option_parser_register(opp, "-aggregate_power_stats", OPT_BOOL,
                         &g_aggregate_power_stats,
                         "Accumulate power across all kernels", "0");

  //Accelwattch Hyrbid Configuration

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_L1_RH", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_L1_RH],
                         "Get L1 Read Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(opp, "-accelwattch_hybrid_perfsim_L1_RM", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_L1_RM],
                         "Get L1 Read Misses for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(opp, "-accelwattch_hybrid_perfsim_L1_WH", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_L1_WH],
                         "Get L1 Write Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(opp, "-accelwattch_hybrid_perfsim_L1_WM", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_L1_WM],
                         "Get L1 Write Misses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_L2_RH", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_L2_RH],
                         "Get L2 Read Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(opp, "-accelwattch_hybrid_perfsim_L2_RM", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_L2_RM],
                         "Get L2 Read Misses for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(opp, "-accelwattch_hybrid_perfsim_L2_WH", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_L2_WH],
                         "Get L2 Write Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(opp, "-accelwattch_hybrid_perfsim_L2_WM", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_L2_WM],
                         "Get L2 Write Misses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_CC_ACC", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_CC_ACC],
                         "Get Constant Cache Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_SHARED_ACC", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_SHRD_ACC],
                         "Get Shared Memory Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_DRAM_RD", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_DRAM_RD],
                         "Get DRAM Reads for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(opp, "-accelwattch_hybrid_perfsim_DRAM_WR", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_DRAM_WR],
                         "Get DRAM Writes for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_NOC", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_NOC],
                         "Get Interconnect Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_PIPE_DUTY", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_PIPE_DUTY],
                         "Get Pipeline Duty Cycle Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_NUM_SM_IDLE", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_NUM_SM_IDLE],
                         "Get Number of Idle SMs for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_CYCLES", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_CYCLES],
                         "Get Executed Cycles for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_VOLTAGE", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_VOLTAGE],
                         "Get Chip Voltage for Accelwattch-Hybrid from Accel-Sim", "0");


  // Output Data Formats
  option_parser_register(
      opp, "-power_trace_enabled", OPT_BOOL, &g_power_trace_enabled,
      "produce a file for the power trace (1=On, 0=Off)", "0");

  option_parser_register(
      opp, "-power_trace_zlevel", OPT_INT32, &g_power_trace_zlevel,
      "Compression level of the power trace output log (0=no comp, 9=highest)",
      "6");

  option_parser_register(
      opp, "-steady_power_levels_enabled", OPT_BOOL,
      &g_steady_power_levels_enabled,
      "produce a file for the steady power levels (1=On, 0=Off)", "0");

  option_parser_register(opp, "-steady_state_definition", OPT_CSTR,
                         &gpu_steady_state_definition,
                         "allowed deviation:number of samples", "8:4");
}

void memory_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_perf_sim_memcpy", OPT_BOOL,
                         &m_perf_sim_memcpy, "Fill the L2 cache on memcpy",
                         "1");
  option_parser_register(opp, "-gpgpu_simple_dram_model", OPT_BOOL,
                         &simple_dram_model,
                         "simple_dram_model with fixed latency and BW", "0");
  option_parser_register(opp, "-gpgpu_dram_scheduler", OPT_INT32,
                         &scheduler_type, "0 = fifo, 1 = FR-FCFS (defaul)",
                         "1");
  option_parser_register(opp, "-gpgpu_dram_partition_queues", OPT_CSTR,
                         &gpgpu_L2_queue_config, "i2$:$2d:d2$:$2i", "8:8:8:8");

  option_parser_register(opp, "-l2_ideal", OPT_BOOL, &l2_ideal,
                         "Use a ideal L2 cache that always hit", "0");
  option_parser_register(opp, "-gpgpu_cache:dl2", OPT_CSTR,
                         &m_L2_config.m_config_string,
                         "unified banked L2 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>}",
                         "64:128:8,L:B:m:N,A:16:4,4");
  option_parser_register(opp, "-gpgpu_cache:dl2_texture_only", OPT_BOOL,
                         &m_L2_texure_only, "L2 cache used for texture only",
                         "1");
  option_parser_register(
      opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
      "number of memory modules (e.g. memory controllers) in gpu", "8");
  option_parser_register(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32,
                         &m_n_sub_partition_per_memory_channel,
                         "number of memory subpartition in each memory module",
                         "1");
  option_parser_register(opp, "-gpgpu_n_mem_per_ctrlr", OPT_UINT32,
                         &gpu_n_mem_per_ctrlr,
                         "number of memory chips per memory controller", "1");
  option_parser_register(opp, "-gpgpu_memlatency_stat", OPT_INT32,
                         &gpgpu_memlatency_stat,
                         "track and display latency statistics 0x2 enables MC, "
                         "0x4 enables queue logs",
                         "0");
  option_parser_register(opp, "-gpgpu_frfcfs_dram_sched_queue_size", OPT_INT32,
                         &gpgpu_frfcfs_dram_sched_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  option_parser_register(opp, "-gpgpu_dram_return_queue_size", OPT_INT32,
                         &gpgpu_dram_return_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  option_parser_register(opp, "-gpgpu_dram_buswidth", OPT_UINT32, &busW,
                         "default = 4 bytes (8 bytes per cycle at DDR)", "4");
  option_parser_register(
      opp, "-gpgpu_dram_burst_length", OPT_UINT32, &BL,
      "Burst length of each DRAM request (default = 4 data bus cycle)", "4");
  option_parser_register(opp, "-dram_data_command_freq_ratio", OPT_UINT32,
                         &data_command_freq_ratio,
                         "Frequency ratio between DRAM data bus and command "
                         "bus (default = 2 times, i.e. DDR)",
                         "2");
  option_parser_register(
      opp, "-gpgpu_dram_timing_opt", OPT_CSTR, &gpgpu_dram_timing_opt,
      "DRAM timing parameters = "
      "{nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}",
      "4:2:8:12:21:13:34:9:4:5:13:1:0:0");
  option_parser_register(opp, "-gpgpu_l2_rop_latency", OPT_UINT32, &rop_latency,
                         "ROP queue latency (default 85)", "85");
  option_parser_register(opp, "-dram_latency", OPT_UINT32, &dram_latency,
                         "DRAM latency (default 30)", "30");
  option_parser_register(opp, "-dram_dual_bus_interface", OPT_UINT32,
                         &dual_bus_interface,
                         "dual_bus_interface (default = 0) ", "0");
  option_parser_register(opp, "-dram_bnk_indexing_policy", OPT_UINT32,
                         &dram_bnk_indexing_policy,
                         "dram_bnk_indexing_policy (0 = normal indexing, 1 = "
                         "Xoring with the higher bits) (Default = 0)",
                         "0");
  option_parser_register(opp, "-dram_bnkgrp_indexing_policy", OPT_UINT32,
                         &dram_bnkgrp_indexing_policy,
                         "dram_bnkgrp_indexing_policy (0 = take higher bits, 1 "
                         "= take lower bits) (Default = 0)",
                         "0");
  option_parser_register(opp, "-dram_seperate_write_queue_enable", OPT_BOOL,
                         &seperate_write_queue_enabled,
                         "Seperate_Write_Queue_Enable", "0");
  option_parser_register(opp, "-dram_write_queue_size", OPT_CSTR,
                         &write_queue_size_opt, "Write_Queue_Size", "32:28:16");
  option_parser_register(
      opp, "-dram_elimnate_rw_turnaround", OPT_BOOL, &elimnate_rw_turnaround,
      "elimnate_rw_turnaround i.e set tWTR and tRTW = 0", "0");
  option_parser_register(opp, "-icnt_flit_size", OPT_UINT32, &icnt_flit_size,
                         "icnt_flit_size", "32");
  m_address_mapping.addrdec_setoption(opp);
}

void shader_core_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_simd_model", OPT_INT32, &model,
                         "1 = post-dominator", "1");
  option_parser_register(
      opp, "-gpgpu_shader_core_pipeline", OPT_CSTR,
      &gpgpu_shader_core_pipeline_opt,
      "shader core pipeline config, i.e., {<nthread>:<warpsize>}", "1024:32");
  option_parser_register(opp, "-gpgpu_tex_cache:l1", OPT_CSTR,
                         &m_L1T_config.m_config_string,
                         "per-shader L1 texture cache  (READ-ONLY) config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}",
                         "8:128:5,L:R:m:N,F:128:4,128:2");
  option_parser_register(
      opp, "-gpgpu_const_cache:l1", OPT_CSTR, &m_L1C_config.m_config_string,
      "per-shader L1 constant memory cache  (READ-ONLY) config "
      " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<"
      "merge>,<mq>} ",
      "64:64:2,L:R:f:N,A:2:32,4");
  option_parser_register(opp, "-gpgpu_cache:il1", OPT_CSTR,
                         &m_L1I_L1_half_C_cache_config.m_config_string,
                         "shader L1 instruction cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>} ",
                         "4:256:4,L:R:f:N,A:2:32,4");
  // MOD. Begin. Added L0I
  option_parser_register(opp, "-gpgpu_cache:il0", OPT_CSTR,
                         &m_L0I_config.m_config_string,
                         "shader L0 instruction cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>} ",
                         "N:64:128:16,L:R:f:N:L,S:2:48,4");
  // MOD. Begin. Added L0I
  option_parser_register(opp, "-gpgpu_cache:dl1", OPT_CSTR,
                         &m_L1D_config.m_config_string,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_l1_cache_write_ratio", OPT_UINT32,
                         &m_L1D_config.m_wr_percent, "L1D write ratio", "0");
  option_parser_register(opp, "-gpgpu_l1_banks", OPT_UINT32,
                         &m_L1D_config.l1_banks, "The number of L1 cache banks",
                         "1");
  option_parser_register(opp, "-gpgpu_l1_banks_byte_interleaving", OPT_UINT32,
                         &m_L1D_config.l1_banks_byte_interleaving,
                         "l1 banks byte interleaving granularity", "32");
  option_parser_register(opp, "-gpgpu_l1_banks_hashing_function", OPT_UINT32,
                         &m_L1D_config.l1_banks_hashing_function,
                         "l1 banks hashing function", "0");
  option_parser_register(opp, "-gpgpu_l1_latency", OPT_UINT32,
                         &m_L1D_config.l1_latency, "L1 Hit Latency", "1");
  option_parser_register(opp, "-gpgpu_smem_latency", OPT_UINT32, &smem_latency,
                         "smem Latency", "3");
  option_parser_register(opp, "-gpgpu_cache:dl1PrefL1", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefL1,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_cache:dl1PrefShared", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefShared,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_gmem_skip_L1D", OPT_BOOL, &gmem_skip_L1D,
                         "global memory access skip L1D cache (implements "
                         "-Xptxas -dlcm=cg, default=no skip)",
                         "0");

  option_parser_register(opp, "-gpgpu_perfect_mem", OPT_BOOL,
                         &gpgpu_perfect_mem,
                         "enable perfect memory mode (no cache miss)", "0");
  option_parser_register(
      opp, "-n_regfile_gating_group", OPT_UINT32, &n_regfile_gating_group,
      "group of lanes that should be read/written together)", "4");
  option_parser_register(
      opp, "-gpgpu_clock_gated_reg_file", OPT_BOOL, &gpgpu_clock_gated_reg_file,
      "enable clock gated reg file for power calculations", "0");
  option_parser_register(
      opp, "-gpgpu_clock_gated_lanes", OPT_BOOL, &gpgpu_clock_gated_lanes,
      "enable clock gated lanes for power calculations", "0");
  option_parser_register(opp, "-gpgpu_shader_registers", OPT_UINT32,
                         &gpgpu_shader_registers,
                         "Number of registers per shader core. Limits number "
                         "of concurrent CTAs. (default 8192)",
                         "8192");
  option_parser_register(
      opp, "-gpgpu_registers_per_block", OPT_UINT32, &gpgpu_registers_per_block,
      "Maximum number of registers per CTA. (default 8192)", "8192");
  option_parser_register(opp, "-gpgpu_ignore_resources_limitation", OPT_BOOL,
                         &gpgpu_ignore_resources_limitation,
                         "gpgpu_ignore_resources_limitation (default 0)", "0");
  option_parser_register(
      opp, "-gpgpu_shader_cta", OPT_UINT32, &max_cta_per_core,
      "Maximum number of concurrent CTAs in shader (default 32)", "32");
  option_parser_register(
      opp, "-gpgpu_num_cta_barriers", OPT_UINT32, &max_barriers_per_cta,
      "Maximum number of named barriers per CTA (default 16)", "16");
  option_parser_register(opp, "-gpgpu_n_clusters", OPT_UINT32, &n_simt_clusters,
                         "number of processing clusters", "10");
  option_parser_register(opp, "-gpgpu_n_cores_per_cluster", OPT_UINT32,
                         &n_simt_cores_per_cluster,
                         "number of simd cores per cluster", "3");
  option_parser_register(opp, "-gpgpu_n_cluster_ejection_buffer_size",
                         OPT_UINT32, &n_simt_ejection_buffer_size,
                         "number of packets in ejection buffer", "8");
  option_parser_register(
      opp, "-gpgpu_n_ldst_response_buffer_size", OPT_UINT32,
      &ldst_unit_response_queue_size,
      "number of response packets in ld/st unit ejection buffer", "2");
  option_parser_register(
      opp, "-gpgpu_shmem_per_block", OPT_UINT32, &gpgpu_shmem_per_block,
      "Size of shared memory per thread block or CTA (default 48kB)", "49152");
  option_parser_register(
      opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_size,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(opp, "-gpgpu_shmem_option", OPT_CSTR,
                         &gpgpu_shmem_option,
                         "Option list of shared memory sizes", "0");
  option_parser_register(
      opp, "-gpgpu_unified_l1d_size", OPT_UINT32,
      &m_L1D_config.m_unified_cache_size,
      "Size of unified data cache(L1D + shared memory) in KB", "0");
  option_parser_register(opp, "-gpgpu_adaptive_cache_config", OPT_BOOL,
                         &adaptive_cache_config, "adaptive_cache_config", "0");
  option_parser_register(
      opp, "-gpgpu_shmem_sizeDefault", OPT_UINT32, &gpgpu_shmem_sizeDefault,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_size_PrefL1", OPT_UINT32, &gpgpu_shmem_sizePrefL1,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(opp, "-gpgpu_shmem_size_PrefShared", OPT_UINT32,
                         &gpgpu_shmem_sizePrefShared,
                         "Size of shared memory per shader core (default 16kB)",
                         "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_num_banks", OPT_UINT32, &num_shmem_bank,
      "Number of banks in the shared memory in each shader core (default 16)",
      "16");
  option_parser_register(
      opp, "-gpgpu_shmem_limited_broadcast", OPT_BOOL, &shmem_limited_broadcast,
      "Limit shared memory to do one broadcast per cycle (default on)", "1");
  option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                         &mem_warp_parts,
                         "Number of portions a warp is divided into for shared "
                         "memory bank conflict check ",
                         "2");
  option_parser_register(
      opp, "-gpgpu_mem_unit_ports", OPT_INT32, &mem_unit_ports,
      "The number of memory transactions allowed per core cycle", "1");
  option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                         &mem_warp_parts,
                         "Number of portions a warp is divided into for shared "
                         "memory bank conflict check ",
                         "2");
  option_parser_register(
      opp, "-gpgpu_warpdistro_shader", OPT_INT32, &gpgpu_warpdistro_shader,
      "Specify which shader core to collect the warp size distribution from",
      "-1");
  option_parser_register(
      opp, "-gpgpu_warp_issue_shader", OPT_INT32, &gpgpu_warp_issue_shader,
      "Specify which shader core to collect the warp issue distribution from",
      "0");
  option_parser_register(opp, "-gpgpu_local_mem_map", OPT_BOOL,
                         &gpgpu_local_mem_map,
                         "Mapping from local memory space address to simulated "
                         "GPU physical address space (default = enabled)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_reg_banks", OPT_INT32,
                         &gpgpu_num_reg_banks,
                         "Number of register banks (default = 8)", "8");
  option_parser_register(
      opp, "-gpgpu_reg_bank_use_warp_id", OPT_BOOL, &gpgpu_reg_bank_use_warp_id,
      "Use warp ID in mapping registers to banks (default = off)", "0");
  option_parser_register(opp, "-gpgpu_sub_core_model", OPT_BOOL,
                         &sub_core_model,
                         "Sub Core Volta/Pascal model (default = off)", "0");
  option_parser_register(opp, "-gpgpu_enable_specialized_operand_collector",
                         OPT_BOOL, &enable_specialized_operand_collector,
                         "enable_specialized_operand_collector", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sp,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_units_dp,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sfu,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_int",
                         OPT_INT32, &gpgpu_operand_collector_num_units_int,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_tensor_core",
                         OPT_INT32,
                         &gpgpu_operand_collector_num_units_tensor_core,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_units_mem,
                         "number of collector units (default = 2)", "2");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_gen",
                         OPT_UINT32, &gpgpu_operand_collector_num_units_gen,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_dp,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_int",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_int,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_in_ports_tensor_core", OPT_INT32,
      &gpgpu_operand_collector_num_in_ports_tensor_core,
      "number of collector unit in ports (default = 1)", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_dp,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_int",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_int,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_out_ports_tensor_core", OPT_INT32,
      &gpgpu_operand_collector_num_out_ports_tensor_core,
      "number of collector unit in ports (default = 1)", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_coalesce_arch", OPT_INT32,
                         &gpgpu_coalesce_arch,
                         "Coalescing arch (GT200 = 13, Fermi = 20)", "13");
  option_parser_register(opp, "-gpgpu_num_sched_per_core", OPT_INT32,
                         &gpgpu_num_sched_per_core,
                         "Number of warp schedulers per core", "1");
  option_parser_register(opp, "-gpgpu_max_insn_issue_per_warp", OPT_INT32,
                         &gpgpu_max_insn_issue_per_warp,
                         "Max number of instructions that can be issued per "
                         "warp in one cycle by scheduler (either 1 or 2)",
                         "2");
  option_parser_register(opp, "-gpgpu_dual_issue_diff_exec_units", OPT_BOOL,
                         &gpgpu_dual_issue_diff_exec_units,
                         "should dual issue use two different execution unit "
                         "resources (Default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_simt_core_sim_order", OPT_INT32,
                         &simt_core_sim_order,
                         "Select the simulation order of cores in a cluster "
                         "(0=Fix, 1=Round-Robin)",
                         "1");
  option_parser_register(
      opp, "-gpgpu_pipeline_widths", OPT_CSTR, &pipeline_widths_string,
      "Pipeline widths "
      "ID_OC_SP,ID_OC_DP,ID_OC_INT,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_DP,OC_EX_"
      "INT,OC_EX_SFU,OC_EX_MEM,EX_WB,ID_OC_TENSOR_CORE,OC_EX_TENSOR_CORE",
      "1,1,1,1,1,1,1,1,1,1,1,1,1");
  option_parser_register(opp, "-gpgpu_tensor_core_avail", OPT_UINT32,
                         &gpgpu_tensor_core_avail,
                         "Tensor Core Available (default=0)", "0");
  option_parser_register(opp, "-gpgpu_num_sp_units", OPT_UINT32,
                         &gpgpu_num_sp_units, "Number of SP units (default=1)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_dp_units", OPT_UINT32,
                         &gpgpu_num_dp_units, "Number of DP units (default=0)",
                         "0");
  option_parser_register(opp, "-gpgpu_num_int_units", OPT_UINT32,
                         &gpgpu_num_int_units,
                         "Number of INT units (default=0)", "0");
  option_parser_register(opp, "-gpgpu_num_sfu_units", OPT_UINT32,
                         &gpgpu_num_sfu_units, "Number of SF units (default=1)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_tensor_core_units", OPT_UINT32,
                         &gpgpu_num_tensor_core_units,
                         "Number of tensor_core units (default=1)", "0");
  option_parser_register(
      opp, "-gpgpu_num_mem_units", OPT_UINT32, &gpgpu_num_mem_units,
      "Number if ldst units (default=1) WARNING: not hooked up to anything",
      "1");
  option_parser_register(
      opp, "-gpgpu_scheduler", OPT_CSTR, &gpgpu_scheduler_string,
      "Scheduler configuration: < lrr | gto | two_level_active > "
      "If "
      "two_level_active:<num_active_warps>:<inner_prioritization>:<outer_"
      "prioritization>"
      "For complete list of prioritization values see shader.h enum "
      "scheduler_prioritization_type"
      "Default: gto",
      "gto");

  option_parser_register(
      opp, "-gpgpu_concurrent_kernel_sm", OPT_BOOL, &gpgpu_concurrent_kernel_sm,
      "Support concurrent kernels on a SM (default = disabled)", "0");
  option_parser_register(opp, "-gpgpu_perfect_inst_const_cache", OPT_BOOL,
                         &perfect_inst_const_cache,
                         "perfect inst and const cache mode, so all inst and "
                         "const hits in the cache(default = disabled)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_inst_fetch_throughput", OPT_INT32, &inst_fetch_throughput,
      "the number of fetched intruction per warp each cycle", "1");
  option_parser_register(opp, "-gpgpu_reg_file_port_throughput", OPT_INT32,
                         &reg_file_port_throughput,
                         "the number ports of the register file", "1");

  for (unsigned j = 0; j < SPECIALIZED_UNIT_NUM; ++j) {
    std::stringstream ss;
    ss << "-specialized_unit_" << j + 1;
    option_parser_register(opp, ss.str().c_str(), OPT_CSTR,
                           &specialized_unit_string[j],
                           "specialized unit config"
                           " {<enabled>,<num_units>:<latency>:<initiation>,<ID_"
                           "OC_SPEC>:<OC_EX_SPEC>,<NAME>}",
                           "0,4,4,4,4,BRA");
  }

  // MOD. Predication
  option_parser_register(opp, "-is_trace_predication_enabled", OPT_BOOL,
                         &is_trace_predication_enabled,
                         "If enabled, the predication information of the traces is read and treated as regular registers beggining with the identifier 257 (264 in execution for PT which is P7 in traces)."
                         "(default = disabled)",
                         "0");

  // MOD. Fix WAR at baseline. Enable or disable scoreboard war in order to solve war hazards at baseline with different modes
  option_parser_register(
      opp, "-scoreboard_war_mode", OPT_CSTR, &scoreboard_war_mode,
      "Scoreboard war mode: < disabled | wb | opc > "
      "It enables a scoreboard_reads to solve potential WAR issues due to operand collector OoO issue."
      "wb: releases source register when the instruction is at WB stage."
      "opc: releases source register when the instruction leaves the operand collector stage"
      "Default: opc",
      "opc");
  // MOD.  Fix WAR at baseline. Configuration of the Scoreboard_reads
  option_parser_register(opp, "-scoreboard_war_max_uses_per_reg", OPT_UINT32,
                         &scoreboard_war_max_uses_per_reg, "Number of maximum uses per register allowed"
                         " in the scoreboard_reads in order to prevent WAR hazards in the baseline. (default=9999)",
                         "9999");
  
  option_parser_register(opp, "-scoreboard_war_static_power", OPT_DOUBLE,
                         &scoreboard_war_static_power, "Static power consumption of each scoreboard war that has more than one bit."
                         "Configure to any positive number (default=0)",
                         "0");

  option_parser_register(opp, "-scoreboard_war_dynamic_power", OPT_DOUBLE,
                         &scoreboard_war_dynamic_power, "Dynamic power consumption of each scoreboard war that has more than one bitr."
                         "Configure to any positive number (default=0)",
                         "0");

  // MOD. Begin. Fix loads after store
  option_parser_register(opp, "-is_fix_memory_reordering_enabled_baseline", OPT_BOOL,
                         &is_fix_memory_reordering_enabled_baseline,
                         "If enabled, the baseline fixes loads after issuing a store. The warp won't be allowed to issue loads until the store reaches WB."
                         "Fix loads after store (default = disabled)",
                         "0");
  // MOD. End

  // MOD. Begin. Improving fetch and decode
  option_parser_register(opp, "-is_fetch_and_decode_improved", OPT_BOOL,
                         &is_fetch_and_decode_improved,
                         "If enabled, the fetch and decode stages are separeted for each sub-core and gpgpu_inst_fetch_throughput is disabled."
                         "Fix the stages fetch and decode (default = disabled)",
                         "0");
  // MOD. End

  // MOD. Begin. Added L0I
  option_parser_register(opp, "-is_L0I_enabled", OPT_BOOL,
                         &is_L0I_enabled,
                         "If enabled, each sub-core has its own L0I. Only works if is_fetch_and_decode_improved is enabled too."
                         "Adds a L0I to each sub-core (default = disabled)",
                         "0");
  option_parser_register(opp, "-max_request_allowed_to_L1I", OPT_INT32,
                         &max_request_allowed_to_L1I, "Number of maximum request allowed to L1I in a given cycle. Also known as maximum number of ports for requests"
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-max_reply_allowed_from_L1I", OPT_INT32,
                         &max_reply_allowed_from_L1I, "Number of maximum reply allowed from L1I in a given cycle. Also known as maximum number of ports for replies"
                         "Configure to any positive number (default=1)",
                         "1"); 
  option_parser_register(opp, "-latency_L0_to_L1", OPT_INT32,
                         &latency_L0_to_L1, "Latency of requests from L0 and L1. L1 if it is instruction cahce, L1.5 constant cache."
                         "Configure to any positive number (default=40)",
                         "40"); 
  option_parser_register(opp, "-latency_L1_to_L0", OPT_INT32,
                         &latency_L1_to_L0, "Latency of replies from L1 to L0. L1 if it is instruction cahce, L1.5 constant cache.."
                         "Configure to any positive number (default=40)",
                         "40"); 
  // MOD. End

  // MOD. Begin. Fix misaligned fetched instructions
  option_parser_register(opp, "-is_fix_instruction_fetch_misalignment", OPT_BOOL,
                         &is_fix_instruction_fetch_misalignment,
                         "If enabled, instructions that are stored in different blocks of the Instruction cache won't be decoded and placed in the Instruction Buffer in the same fetch request. Only works if is_fetch_and_decode_improved is enabled too."
                         "Fix misaligment when fetching instructions (default = disabled)",
                         "0");
  // MOD. End. Fix misaligned fetched instructions

  // MOD. Begin. Instruction addresses of different kernels have a different address request in memory
  option_parser_register(opp, "-is_fix_different_kernels_pc_addresses", OPT_BOOL,
                         &is_fix_different_kernels_pc_addresses,
                         "If enabled, instruction addresses of different kernels to have a different address request in memory. It is mandatory to enable -is_fetch_and_decode_improved too."
                         "Fix instruction pc addresses requests (default = disabled)",
                         "0");
  // MOD. End

  // MOD. Begin. Instruction decodign when the following PC is far from the previous PC
  option_parser_register(opp, "-is_fix_not_decoding_not_contiguos_instructions", OPT_BOOL,
                         &is_fix_not_decoding_not_contiguos_instructions,
                         "If enabled, instructions with a separated PC are prevented from being decoding because it is impossible that have been fetch in the same request. It is mandatory to enable -is_fetch_and_decode_improved too."
                         "Fix instruction decodign when the following PC is far from the previous PC (default = disabled)",
                         "0");
  // MOD. End

  // MOD. Begin. Fixed LDST_Unit model
  option_parser_register(opp, "-is_improved_ldst_unit_enabled", OPT_BOOL,
                         &is_improved_ldst_unit_enabled,
                         "If enabled, the modeling of LDST_Unit is improved. Supporting to dispatch instructions and do different actions from different sub-cores."
                         "Improves the modeling of the LDST_Unit (default = disabled)",
                         "0");
  // MOD. End

  // MOD. Begin. Improved Result bus to take into account conflicts with RF banks
  option_parser_register(opp, "-is_improved_result_bus", OPT_BOOL,
                         &is_improved_result_bus,
                         "If enabled, the modeling of result bus is improved. It takes into account the number of ports in each RF bank to do not schedule instructions that are going to finish in the same cycle for more than the ports that are in that RF bank."
                         "Improves the modeling of the result bus (default = disabled)",
                         "0");
  // MOD. End

  // MOD. Begin. Improving OPC
  option_parser_register(opp, "-is_opc_improved", OPT_BOOL,
                         &is_opc_improved,
                         "If enabled, the OPC stage does not used the gpgpu_reg_file_port_throughput to make a loop of the OPC and recreates OPC stage more faithfully."
                         "gpgpu_reg_file_port_throughput will be used as the number of ports for each register file bank."
                         "Fix the OPC stage (default = disabled)",
                         "0");
  option_parser_register(opp, "-cu_num_ports", OPT_INT32,
                         &cu_num_ports, "Number of ports for each Collector Unit when they read operands from the Register File. Only enabled if is_opc_improved is enabled."
                         "Configure to any positive number (default=2)",
                         "2");

  // MOD. End. Improving OPC

  // MOD. Begin. Skip RF limitation
  option_parser_register(opp, "-is_skip_rf_limit_enabled", OPT_BOOL,
                         &is_skip_rf_limit_enabled,
                         "If enabled, the limitation of registers for a RF is not done."
                         "Skips checking the limitation of registers for a CTA (default = disabled)",
                         "0");
  // MOD. End

  // MOD. Begin. Relax barriers in baseline
  option_parser_register(opp, "-is_relax_barriers_baseline", OPT_BOOL,
                         &is_relax_barriers_baseline,
                         "If enabled, after a barrier only memory operations will stall a warp when this is waiting in a barrier (default = disabled)",
                         "0");
  // MOD. End

  // MOD. Begin. Extended IBuffer
  option_parser_register(opp, "-is_extended_ibuffer_enabled", OPT_BOOL,
                         &is_extended_ibuffer_enabled,
                         "If enabled, the extended buffer is used. Also if LOOG is enabled."
                         "(default = disabled)",
                         "0");
                         
  option_parser_register(opp, "-extended_ibuffer_size", OPT_INT32,
                         &extended_ibuffer_size, "Size of the extended Instruction Buffer. If LOOG is enabled, loog_frontend_size is used instead of this variable."
                         "Configure to any positive number (default=8)",
                         "8");

  option_parser_register(opp, "-fetch_decode_width", OPT_INT32,
                         &fetch_decode_width, "Size of the fetch and decode width. Cannot be bigger than extended_ibuffer_size or ibuffer_remodeled_size. If LOOG is enabled, loog_frontend_size is used instead of this variable."
                         "Configure to any positive number (default=2)",
                         "1");

  option_parser_register(opp, "-extended_ibuffer_static_power", OPT_DOUBLE,
                         &extended_ibuffer_static_power, "Static power consumption of each Extended Ibuffer."
                         "Configure to any positive number (default=0)",
                         "0");

  option_parser_register(opp, "-extended_ibuffer_dynamic_power", OPT_DOUBLE,
                         &extended_ibuffer_dynamic_power, "Dynamic power consumption of each Extended Ibuffer."
                         "Configure to any positive number (default=0)",
                         "0");
  // MOD. End. Extended IBuffer


  // MOD. Begin VPREG
  option_parser_register(
      opp, "-vpreg_mode", OPT_CSTR, &vpreg_mode_string,
      "Virtual Physical Registers mode: < disabled | reissue | reissue_informed > "
      "It enables a VPREG improvement in order to use renaming and reduce the size of register file. It also needs that -ibuffer_ooo_mode is set to wb."
      "reissue: In case of not having free physical registers at WB, the bits is_issued and is_reissued of the instruction in IBuffer_ooo are set to 1 and the instruction will be reissued."
      "reissue_informed: It works equally to reissue mode. However, the SOCGPU module knows and prevents to select instructions that are not the oldest when there aren't free physical registers."
      "disabled: The improvement is disabled."
      "Default: disabled",
      "disabled");

  option_parser_register(opp, "-vpreg_num_virtual_regs_per_sm", OPT_INT32,
                         &vpreg_num_virtual_regs_per_sm, "Number of virtual register inside each SM. Later on this number will be divided in 4 to assign equally to each sub-core. Recommended to be a power of 2"
                         "It is only used if vpreg is enabled. It should be bigger than the number of physical + maximum of destination registers allowed in an instruction."
                         "Configure to any number (default=2052)",
                         "2052");

  option_parser_register(opp, "-vpreg_num_physical_regs_per_sm", OPT_INT32,
                         &vpreg_num_physical_regs_per_sm, "Number of physical register inside each SM. Later on this number will be divided in 4 to assign equally to each sub-core. Recommended to be a power of 2"
                         "It is only used if vpreg is enabled."
                         "Configure to any positive number (default=2048)",
                         "2048");

  option_parser_register(opp, "-vpreg_balanced_banks_mode", OPT_BOOL,
                         &is_vpreg_balanced_banks_mode_enabled,
                         "If enabled, the allocation of physical registers will be balanced across the different banks."
                         "Virtual physical registers balanced banks mode (default = disabled)",
                         "0");

  option_parser_register(opp, "-vpreg_reissue_informed_socgpu_threshold", OPT_INT32,
                         &vpreg_reissue_informed_socgpu_threshold, "Number of physical registers that when it reaches this number or less than this number, only warps with the oldest instruction ready will be consider to be issued ."
                         "It is only used if vpreg is set to reissue_informed. It should be greater or equal than 0. This number will be for each sub-core free pool."
                         "Configure to any number greater or equal than 0 (default=0)",
                         "0");

  option_parser_register(opp, "-vpreg_max_rollback_entries_done_in_a_cycle", OPT_INT32,
                         &vpreg_max_rollback_entries_done_in_a_cycle, "Number of maximum entries that can do a rollback from the Instruction Buffer to the decode VMT in a cycle. Until the rollback is finished, the decode for that warp is stalled."
                         "If is set to 0 is treated like an ideal scenario where all the entries of the Instruction Bufffer do the rollback the same cycle as the flush is detected. Therefore, there isn't any stall. If it is configured equal to the size of the Instruction Buffer, it will have only one cycle stall."
                         "Configure to any number greater or equal than 0 and smaller or equal than the size of the Instruction Buffer(default=0)",
                         "0");

  option_parser_register(opp, "-is_vpreg_predicated_war_waw_dependencies_ignored", OPT_BOOL,
                         &is_vpreg_predicated_war_waw_dependencies_ignored,
                         "If enabled, the war and waw hazards because there is a predication register involded are ignored (default = disabled)",
                         "0");
  
  option_parser_register(opp, "-is_vpreg_predicated_dest_reg_dependencies_ignored", OPT_BOOL,
                         &is_vpreg_predicated_dest_reg_dependencies_ignored,
                         "If enabled, the dependencies when two instructions have the same logical destination register and different predication is ignored (default = disabled)",
                         "0");
  
  option_parser_register(opp, "-vpreg_merge_module_static_power", OPT_DOUBLE,
                         &vpreg_merge_module_static_power, "Static power consumption of each VPREG merge module."
                         "Configure to any positive number (default=0)",
                         "0");

  option_parser_register(opp, "-vpreg_merge_module_dynamic_power", OPT_DOUBLE,
                         &vpreg_merge_module_dynamic_power, "Dynamic power consumption of each VPREG merge module."
                         "Configure to any positive number (default=0)",
                         "0");

  option_parser_register(opp, "-vpreg_collector_unit_extra_static_power", OPT_DOUBLE,
                         &vpreg_collector_unit_extra_static_power, "Static power consumption of each collector unit module."
                         "Configure to any positive number (default=0)",
                         "0");

  option_parser_register(opp, "-vpreg_collector_unit_extra_dynamic_power", OPT_DOUBLE,
                         &vpreg_collector_unit_extra_dynamic_power, "Dynamic power consumption of each collector unit module."
                         "Configure to any positive number (default=0)",
                         "0");
  // MOD. End VPREG

  // MOD. Begin. Remodeling
  option_parser_register(
      opp, "-is_SM_remodeling_enabled", OPT_BOOL, &is_SM_remodeling_enabled,
      "If enabled, the simulator will use a more accurate model for the SMs "
      "based on NVIDIA Volta/Turing/Ampere."
      "is_SM_remodeling_enabled (default = disabled)",
      "0");
  option_parser_register(opp, "-num_subcores_in_SM", OPT_INT32,
                         &num_subcores_in_SM,
                         "Configures the number of subcores in the SM. Usually "
                         "4 in the latests NVIDIA architectures since Volta."
                         "num_subcores_in_SM (default = 4)",
                         "4");
  option_parser_register(
      opp, "-is_remodeling_scoreboarding_enabled", OPT_BOOL,
      &is_remodeling_scoreboarding_enabled,
      "If enabled, the simulator will use scoreboards in the accurate model "
      "for the SMs based on NVIDIA Volta/Turing/Ampere. Otherwise, it will use "
      "the model based on hints in the control bits written by the compiler. "
      "If it uses PTX mode, Scoreboards will be also enabled due to the "
      "imposibility of using the control bits. "
      "is_remodeling_scoreboarding_enabled (default = disabled)",
      "0");
  option_parser_register(opp, "-is_ibuffer_remodeled_enabled", OPT_BOOL,
                         &is_ibuffer_remodeled_enabled,
                         "If enabled, the extended buffer is used. Also if LOOG is enabled."
                         "(default = disabled)",
                         "0");
                         
  option_parser_register(opp, "-ibuffer_remodeled_size", OPT_INT32,
                         &ibuffer_remodeled_size, "Size of the extended Instruction Buffer. If LOOG is enabled, loog_frontend_size is used instead of this variable."
                         "Configure to any positive number (default=3)",
                         "3");
  option_parser_register(opp, "-num_wait_barriers_per_warp", OPT_UINT32,
                         &num_wait_barriers_per_warp, "Number of wait barriers that each warp has. Current architectures like Volta/Turing/Ampere have 6 barriers."
                         "Configure to any positive number (default=6)",
                         "6");
  option_parser_register(opp, "-sfu_latency", OPT_INT32,
                         &sfu_latency, "Latency of the SFU instructions."
                         "Configure to any positive number (default=21)",
                         "21");
  option_parser_register(opp, "-tensor_latency", OPT_INT32,
                         &tensor_latency, "Maximum latency of the Tensor instructions."
                         "Configure to any positive number (default=32)",
                         "32");
  option_parser_register(opp, "-tensor_extra_latency_16816_fp32_1688_fp32", OPT_INT32,
                         &tensor_extra_latency_16816_fp32_1688_fp32, "Maximum latency of the Tensor instructions."
                         "Configure to any positive number (default=0)",
                         "0");
  option_parser_register(opp, "-tensor_rate_per_cycle", OPT_INT32,
                         &tensor_rate_per_cycle, "Rate of processing of the tensor cores per cycle."
                         "Configure to any positive number (default=2048)",
                         "2048");
  option_parser_register(opp, "-branch_latency", OPT_INT32,
                         &branch_latency, "Latency of the branch instructions."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-half_latency", OPT_INT32,
                         &half_latency, "Latency of the half instructions."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-uniform_latency", OPT_INT32,
                         &uniform_latency, "Latency of the uniform instructions."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-predicate_latency", OPT_UINT32,
                         &predicate_latency, "Latency of the predicate instructions."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-miscellaneous_queue_latency", OPT_INT32,
                         &miscellaneous_queue_latency, "Latency of the miscellaneous queue instructions."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-miscellaneous_no_queue_latency", OPT_INT32,
                         &miscellaneous_no_queue_latency, "Latency of the miscellaneous no queue instructions."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-sfu_initiation", OPT_INT32,
                         &sfu_initiation, "Initiation interval of the SFU instructions in the dispatch latch. The number of cycles-1 that an instruction can not be sent to the same execution pipeline."
                         "Configure to any positive number (default=8)",
                         "8");
  option_parser_register(opp, "-tensor_initiation", OPT_INT32,
                         &tensor_initiation, "Initiation interval of the Tensor instructions in the dispatch latch. The number of cycles-1 that an instruction can not be sent to the same execution pipeline."
                         "Configure to any positive number (default=16)",
                         "16");
  option_parser_register(opp, "-branch_initiation", OPT_INT32,
                         &branch_initiation, "Initiation interval of the branch instructions in the dispatch latch. The number of cycles-1 that an instruction can not be sent to the same execution pipeline."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-half_initiation", OPT_INT32,
                         &half_initiation, "Initiation interval of the half instructions in the dispatch latch. The number of cycles-1 that an instruction can not be sent to the same execution pipeline."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-uniform_initiation", OPT_INT32,
                         &uniform_initiation, "Initiation interval of the uniform instructions in the dispatch latch. The number of cycles-1 that an instruction can not be sent to the same execution pipeline."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-predicate_initiation", OPT_INT32,
                         &predicate_initiation, "Initiation interval of the predicate instructions in the dispatch latch. The number of cycles-1 that an instruction can not be sent to the same execution pipeline."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-miscellaneous_queue_initiation", OPT_INT32,
                         &miscellaneous_queue_initiation, "Initiation interval of the miscellaneous queue instructions in the dispatch latch. The number of cycles-1 that an instruction can not be sent to the same execution pipeline."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-miscellaneous_no_queue_initiation", OPT_INT32,
                         &miscellaneous_no_queue_initiation, "Initiation interval of the miscellaneous no queue instructions in the dispatch latch. The number of cycles-1 that an instruction can not be sent to the same execution pipeline."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-miscellaneous_queue_size", OPT_UINT32,
                         &miscellaneous_queue_size, "Size of the queue inside each subcore for the miscellaneous queue instructions."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-memory_subcore_queue_size", OPT_UINT32,
                         &memory_subcore_queue_size, "Size of the queue inside each subcore for the LD/ST/TEXT instructions before being sent to the shared unit of the SM."
                         "Configure to any positive number (default=4)",
                         "4");
  option_parser_register(opp, "-memory_intermidiate_stages_subcore_unit", OPT_UINT32,
                         &memory_intermidiate_stages_subcore_unit, "Number of intermediate stages in the memory pipeline of the subcore unit."
                         "Configure to any positive number (default=3)",
                         "3");
  option_parser_register(opp, "-memory_sm_prt_size", OPT_UINT32,
                         &memory_sm_prt_size, "Size of the PRT inside the shared memory unit in each SM. It allows to have several instructions solving their accesses to the different types of memories."
                         "Configure to any positive number (default=64)",
                         "64");
  option_parser_register(opp, "-num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline_when_is_mem_inst", OPT_UINT32,
                         &num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline_when_is_mem_inst, "Number of cycles that the shared pipelines by sub-cores in the SM needs to wait until it is allowed to do the next issue when dispatched instruction is memory. Current cycle is included in the counter. For example, 1 allows to issue the next cycle."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline_when_is_dp_inst", OPT_UINT32,
                         &num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline_when_is_dp_inst, "Number of cycles that the shared pipelines by sub-cores in the SM needs to wait until it is allowed to do the next issue when dispatched instruction is DP. Only effective if -dp_sm_shared_queue_size is 1. Current cycle is included in the counter. For example, 1 allows to issue the next cycle."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-memory_subcore_extra_latency_load_shared_mem", OPT_UINT32,
                         &memory_subcore_extra_latency_load_shared_mem, "Offset latency of the shared memory instruction when is being processed (calculating address and other stuff) inside each subcore."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-memory_shared_memory_minimum_latency", OPT_UINT32,
                         &memory_shared_memory_minimum_latency, "Minimum latency of the shared memory instruction when is accessing to the shared memory of the SM in the memory structures."
                         "Configure to any positive number (default=16)",
                         "16");
  option_parser_register(opp, "-memory_shared_memory_extra_latency_ldsm_multiple_matrix", OPT_UINT32,
                         &memory_shared_memory_extra_latency_ldsm_multiple_matrix, "Extra latency required at SM shared memory structures when LDSM loads multiple matrices."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-memmory_max_concurrent_requests_shmem_per_sm", OPT_UINT32,
                         &memmory_max_concurrent_requests_shmem_per_sm, "Maximum number of shared memory instructions that can be concurrent in the memory SM structure."
                         "Configure to any positive number (default=4)",
                         "4");
  option_parser_register(opp, "-memmory_max_concurrent_requests_standard_per_sm", OPT_UINT32,
                         &memmory_max_concurrent_requests_standard_per_sm, "Maximum number of standard memory instructions (no shared) that can be concurrent in the memory SM structure."
                         "Configure to any positive number (default=8)",
                         "8");
  option_parser_register(opp, "-sm_memory_unit_l1c_access_queue_size", OPT_UINT32,
                         &sm_memory_unit_l1c_access_queue_size, "Size of the access queue to the L1 constant cache at the memory SM structure."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-sm_memory_unit_l1t_access_queue_size", OPT_UINT32,
                          &sm_memory_unit_l1t_access_queue_size, "Size of the access queue to the L1 texture cache at the memory SM structure."
                          "Configure to any positive number (default=1)",
                          "1");
  option_parser_register(opp, "-sm_memory_unit_l1d_access_queue_size", OPT_UINT32,
                          &sm_memory_unit_l1d_access_queue_size, "Size of the access queue to the L1 constant cache at the memory SM structure. There is one queue per bank."
                          "Configure to any positive number (default=1)",
                          "1");
  option_parser_register(opp, "-sm_memory_unit_shmem_access_queue_size", OPT_UINT32,
                          &sm_memory_unit_shmem_access_queue_size, "Size of the access queue to the Shared memory (SHMEM) at the memory SM structure."
                          "Configure to any positive number (default=1)",
                          "1");
  option_parser_register(opp, "-sm_memory_unit_bypass_l1d_directly_go_to_l2_access_queue_size", OPT_UINT32,
                          &sm_memory_unit_bypass_l1d_directly_go_to_l2_access_queue_size, "Size of the access queue to directly to L2 because is bypassin l1d at the memory SM structure. There is one queue per bank."
                          "Configure to any positive number (default=1)",
                          "1");
  option_parser_register(opp, "-sm_memory_unit_miscellaneous_access_queue_size", OPT_UINT32,
                          &sm_memory_unit_miscellaneous_access_queue_size, "Size of the access queue to the miscellaneous at the memory SM structure."
                          "Configure to any positive number (default=1)",
                          "1");
  option_parser_register(opp, "-constant_cache_latency_at_sm_structure", OPT_UINT32,
                         &constant_cache_latency_at_sm_structure, "Minimum latency of the constant  memory instruction when is accessing to the constant memory (l1 constant cache) of the SM in the memory structures."
                         "Configure to any positive number (default=20)",
                         "20");
  option_parser_register(opp, "-memory_l1d_minimum_latency", OPT_UINT32,
                         &memory_l1d_minimum_latency, "Minimum latency of the global memory instruction when is accessing to the L1D cache of the SM in the memory structures."
                         "Configure to any positive number (default=11)",
                         "11");
  option_parser_register(opp, "-memory_global_shared_latency_for_ldgsts", OPT_UINT32,
                         &memory_global_shared_latency_for_ldgsts, "Latency of the ICNT network between global and shared memory for the the LDGSTS."
                         "Configure to any positive number (default=11)",
                         "11");
  option_parser_register(opp, "-memory_l1d_max_lookups_per_cycle_per_bank", OPT_UINT32,
                         &memory_l1d_max_lookups_per_cycle_per_bank, "Maximum number of look ups allowed per cycle in each bank of the L1D. If there are 4 banks and this parameter is set to 4, it will perform a maximum of 16 look ups in total.."
                         "Configure to any positive number (default=4)",
                         "4");
  option_parser_register(opp, "-memory_maximum_coalescing_cycles", OPT_UINT32,
                         &memory_maximum_coalescing_cycles, "Minimum latency of the memory instruction when it needs to perform a coalescing operation for a memory instruction."
                         "Configure to any positive number (default=1)",
                         "1");                                                    
  option_parser_register(opp, "-offset_latency_firts_stage_memory_subcore", OPT_INT32,
                         &offset_latency_firts_stage_memory_subcore, "Number of cycles difference at the first stage of memory subcore unit respect to the baseline architecture Ampere."
                         "Configure to any positive number (default=0)",
                         "0");   
  option_parser_register(opp, "-memory_num_scalar_units_per_subcore", OPT_UINT32,
                         &memory_num_scalar_units_per_subcore, "Number of scalar units inside each subcore for the memory instructions. Used for calculating addresses"
                         "Configure to any positive number (default=4)",
                         "8");
  option_parser_register(opp, "-memory_subcore_link_to_sm_byte_size", OPT_UINT32,
                         &memory_subcore_link_to_sm_byte_size, "Byte size between the link of a subcore and the SM for transfering data of memory instructions to SM memory structures."
                         "Configure to any positive number (default=4)",
                         "4");
  option_parser_register(opp, "-is_load_half_bandwidth_in_the_subcore_link_to_sm_enabled", OPT_BOOL,
                         &is_load_half_bandwidth_in_the_subcore_link_to_sm_enabled, "If enabled, laods have half of the bandwidth specified at -memory_subcore_link_to_sm_byte_size."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-is_store_half_bandwidth_in_the_subcore_link_to_sm_enabled", OPT_BOOL,
                         &is_store_half_bandwidth_in_the_subcore_link_to_sm_enabled, "If enabled, stores have half of the bandwidth specified at -memory_subcore_link_to_sm_byte_size."
                         "Configure to any positive number (default=0)",
                         "0");
  option_parser_register(opp, "-dp_subcore_queue_size", OPT_UINT32,
                         &dp_subcore_queue_size, "Size of the queue inside each subcore for the DP instructions before being sent to the shared unit of the SM. Only used if there is not dedicated DP unit in each sub-core"
                         "Configure to any positive number (default=4)",
                         "4");
  option_parser_register(opp, "-dp_subcore_max_latency", OPT_UINT32,
                         &dp_subcore_max_latency, "Number of maximum cycles that the DP instructions takes for transferring the data from the subcore to the shared unit of the SM. Only used if there is not dedicated DP unit in each sub-core. While is active, it is not possible to transfer other DP instruction."
                         "Configure to any positive number (default=8)",
                         "8");
  option_parser_register(opp, "-is_dp_pipeline_shared_for_subcores", OPT_BOOL,
                         &is_dp_pipeline_shared_for_subcores,
                         "If enabled, Subcores have a shared unit for executing the DP instructions. The -dp_subcore_queue_size is taken into account ."
                         "(default = enabled)",
                         "1");
  option_parser_register(opp, "-dp_sm_shared_queue_size", OPT_UINT32,
                         &dp_sm_shared_queue_size, "Size of the queue inside each SM shared execution unit for the DP instructions. Only used if there is not dedicated DP unit in each sub-core"
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-dp_shared_intermidiate_stages", OPT_UINT32,
                         &dp_shared_intermidiate_stages, "Number of intermediate stages in the shared DP pipeline."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-is_fp32ops_allowed_in_int_pipeline", OPT_BOOL,
                         &is_fp32ops_allowed_in_int_pipeline,
                         "If enabled, FFMA instructions can be dispatched also to the INT execution pipeline."
                         "(default = disabled)",
                         "0");
  option_parser_register(opp, "-is_fp32_and_int_unified_pipeline", OPT_BOOL,
                         &is_fp32_and_int_unified_pipeline,
                         "If enabled, FP32 and INT instructions are dispatched to the same execution pipeline."
                         "(default = disabled)",
                         "0");
  option_parser_register(opp, "-invalidate_instruction_caches_at_kernel_end", OPT_BOOL,
                         &invalidate_instruction_caches_at_kernel_end,
                         "Invalidate instructions cache at the end of each kernel call", "0");
  option_parser_register(opp, "-ibuffer_coalescing", OPT_BOOL,
                         &ibuffer_coalescing,
                         "If enabled, IBuffers will snoop for instructions even though the served request is not for them."
                         "(default = disabled)",
                         "0");
  option_parser_register(opp, "-perfect_constant_cache", OPT_BOOL,
                         &perfect_constant_cache,
                         "If enabled, constant cache."
                         "(default = disabled)",
                         "0");
  option_parser_register(opp, "-perfect_instruction_cache", OPT_BOOL,
                         &perfect_instruction_cache,
                         "If enabled, Perfect_instruction cache."
                         "(default = disabled)",
                         "0");
  option_parser_register(opp, "-is_instruction_prefetching_enabled", OPT_BOOL,
                         &is_instruction_prefetching_enabled,
                         "If enabled, Instruction cache has prefetching enabled. The prefetcher just tries to prefetc subsequent blocks of the current one. That number of blocks can be configured with -prefetch_per_stream_buffer_size."
                         "(default = disabled)",
                         "0");
  option_parser_register(opp, "-prefetch_per_stream_buffer_size", OPT_UINT32,
                         &prefetch_per_stream_buffer_size, "Number of blocks that the instruction cache prefetcher (stream buffer)can hold. Only effective if -is_instruction_prefetching_enabled is enabled."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-prefetch_num_stream_buffers", OPT_UINT32,
                         &prefetch_num_stream_buffers, "Number of stream buffers in the first level instruction cache. Only effective if -is_instruction_prefetching_enabled is enabled."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-num_instruction_prefetches_per_cycle", OPT_UINT32,
                         &num_instruction_prefetches_per_cycle, "Number of blocks that the instruction cache prefetcher tries to prefetch every cycle. Only effective if -is_instruction_prefetching_enabled is enabled."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-is_rf_cache_enabled", OPT_BOOL,
                         &is_rf_cache_enabled,
                         "If enabled, Regular register file has the register file feature enabled."
                         "(default = enabled)",
                         "1");
  option_parser_register(opp, "-max_operands_regular_register_file", OPT_INT32,
                         &max_operands_regular_register_file, "Number of operands with regular register allowed per instruction. Only used when -is_rf_cache_enabled is enabled."
                         "Configure to any positive number (default=4)",
                         "4");
  option_parser_register(opp, "-max_latency_regular_register_file_latency", OPT_INT32,
                         &max_latency_regular_register_file_latency, "Maximum supported latency (cycles) for reading operands in the regular register file."
                         "Configure to any positive number (default=4)",
                         "4");
  option_parser_register(opp, "-num_regular_register_file_read_ports_per_bank", OPT_INT32,
                         &num_regular_register_file_read_ports_per_bank, "Number of read ports per bank in the regular register file available per cycle."
                         "Configure to any positive number (default=6)",
                         "6");
  option_parser_register(opp, "-num_regular_register_file_write_ports_per_bank", OPT_INT32,
                         &num_regular_register_file_write_ports_per_bank, "Number of write ports per bank in the regular register file available per cycle."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-max_size_register_file_write_queue_for_fixed_latency_instructions", OPT_INT32,
                         &max_size_register_file_write_queue_for_fixed_latency_instructions, "Maximum size of the register file write_queue for fixed latency instructions."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-max_pops_per_cycle_register_file_write_queue_for_fixed_latency_instructions", OPT_INT32,
                         &max_pops_per_cycle_register_file_write_queue_for_fixed_latency_instructions, "Maximum number of pops per cycle of the register file write_queue for fixed latency instructions."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-num_threads_granularity_read_regular_register_file_dp_inst", OPT_INT32,
                         &num_threads_granularity_read_regular_register_file_dp_inst, "Granularity of theads inside a warp for reading RF for DP instrucions. For example, 8 means that it needs 4 cycles because the width is 256 bits and takes 4 cycles to read the 32 threads (1024 width)."
                         "Configure to any positive number (default=8)",
                         "8");
  option_parser_register(opp, "-num_threads_granularity_read_regular_register_file_mem_inst", OPT_INT32,
                         &num_threads_granularity_read_regular_register_file_mem_inst, "Granularity of theads inside a warp for reading RF for memory instrucions. For example, 8 means that it needs 4 cycles because the width is 256 bits and takes 4 cycles to read the 32 threads (1024 width)."
                         "Configure to any positive number (default=8)",
                         "8");
  option_parser_register(opp, "-num_threads_granularity_read_regular_register_file_sfu_inst", OPT_INT32,
                         &num_threads_granularity_read_regular_register_file_sfu_inst, "Granularity of theads inside a warp for reading RF for special function instrucions. For example, 8 means that it needs 4 cycles because the width is 256 bits and takes 4 cycles to read the 32 threads (1024 width)."
                         "Configure to any positive number (default=8)",
                         "8");
  option_parser_register(opp, "-num_threads_granularity_read_regular_register_file_other_inst", OPT_INT32,
                         &num_threads_granularity_read_regular_register_file_other_inst, "Granularity of theads inside a warp for reading RF for DP instrucions. For example, 8 means that it needs 4 cycles because the width is 256 bits and takes 4 cycles to read the 32 threads (1024 width)."
                         "Configure to any positive number (default=8)",
                         "8");
  option_parser_register(opp, "-num_cycles_needed_to_write_a_reg_from_sm_struct_to_subcore", OPT_INT32,
                         &num_cycles_needed_to_write_a_reg_from_sm_struct_to_subcore, "Number of cycles needed to write a register from the SM structure (memory or shared dp if enabled) to the subcore."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-is_const_cache_accessed_blocks_tracking_enabled", OPT_BOOL,
                         &is_const_cache_accessed_blocks_tracking_enabled,
                         "If enabled, a set will track all the address accessed by the constant cache and then it will report the different number of accesses blocks at the end of the execution."
                         "(default = disabled)",
                         "0");
  option_parser_register(opp, "-is_global_memory_accesses_blocks_tracking_enabled", OPT_BOOL,
                         &is_global_memory_accesses_blocks_tracking_enabled,
                         "If enabled, a set will track all the address accessed by the global memory and then it will report the different number of accessed blocks at the end of the execution."
                         "(default = disabled)",
                         "0");
  option_parser_register(opp, "-num_const_cache_cycle_misses_before_switch_to_other_warp", OPT_UINT32,
                         &num_const_cache_cycle_misses_before_switch_to_other_warp, "Number of cycles the issue stage can not swap to a different warp due to constant cache keeps trying a miss."
                         "Configure to any positive number (default=4)",
                         "4");
  option_parser_register(opp, "-num_cycles_issue_port_busy_after_imadwide", OPT_UINT32,
                         &num_cycles_issue_port_busy_after_imadwide, "Number of cycles the issue stage can not issue any instruction because the previous one issued (an IMAD.WIDE) is keeping the issue port busy."
                         "Configure to any positive number (default=4)",
                         "4");
  option_parser_register(opp, "-num_stall_cycles_wait_after_bits_stall_0_and_yield", OPT_UINT32,
                         &num_stall_cycles_wait_after_bits_stall_0_and_yield, "Number of cycles that the stall counter is set when the instruction has the combination of Stall bits set to 0 and yield set on."
                         "Configure to any positive number (default=0)",
                         "0");
  option_parser_register(opp, "-num_cycles_to_stall_SM_at_gpu_memory_barrier", OPT_UINT32,
                         &num_cycles_to_stall_SM_at_gpu_memory_barrier, "Number of cycles that the SM is stalled after all the warps of the SM have reached the MEMBAR.GPU."
                         "Configure to any positive number (default=0)",
                         "186");
  option_parser_register(opp, "-num_cycles_to_stall_SM_at_system_memory_barrier", OPT_UINT32,
                         &num_cycles_to_stall_SM_at_system_memory_barrier, "Number of cycles that the SM is stalled after all the warps of the SM have reached the MEMBAR.SYS."
                         "Configure to any positive number (default=0)",
                         "2900");
  option_parser_register(opp, "-num_cycles_to_stall_SM_at_cta_memory_barrier", OPT_UINT32,
                         &num_cycles_to_stall_SM_at_cta_memory_barrier, "Number of cycles that the SM is stalled after all the warps of the SM have reached the MEMBAR.CTA."
                         "Configure to any positive number (default=0)",
                         "53");
  option_parser_register(
      opp, "-gpgpu_subcore_const_cache:l0", OPT_CSTR, &m_L0C_config.m_config_string,
      "per-subcore L0 constant memory cache  (READ-ONLY) config "
      " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<"
      "merge>,<mq>} ",
      "2:128:2,L:R:f:N,A:2:32,4");
  // MOD. End. Remodeling

  // MOD. Begin. Parallelism
  option_parser_register(opp, "-is_custom_omp_scheduler_enabled", OPT_BOOL,
                         &is_custom_omp_scheduler_enabled,
                         "If enabled, a set will track all the address accessed by the constant cache and then it will report the different number of accesses at the end of the execution."
                         "(default = enabled)",
                         "1");
  option_parser_register(opp, "-custom_omp_scheduler_ratio_to_dynamic", OPT_FLOAT,
                         &custom_omp_scheduler_ratio_to_dynamic,
                         "Ratio which below that number the OMP for-loop scheduler is set to dynamic. Range from 0 to 1."
                         "(default = 0.3)",
                         "0.3");
  // MOD. End. Parallelism

  option_parser_register(opp, "-filter_first_kernel_id", OPT_UINT32,
                          &filter_first_kernel_id, "If enabled, the simulation will start from the first kernel id configured in this parameter. If it has a value of 1 or 0 it is disabled."
                          "Configure to any positive number (default=0)",
                          "0");
  option_parser_register(opp, "-filter_last_kernel_id", OPT_UINT32,
                          &filter_last_kernel_id, "If enabled, the simulation will end at the last kernel id configured in this parameter. If it has a value of 1 or 0 it is disabled."
                          "Configure to any positive number (default=0)",
                          "0");
  // MOD. Begin. InterWarp coalescing
  option_parser_register(opp, "-measure_coalescing_potential_stats", OPT_BOOL,
    &measure_coalescing_potential_stats,
    "If enabled, We track the distance and how much intra and inter warp coalescing exists It consumes more RAM."
    "(default = 0 (disabled))",
    "0");
  option_parser_register(opp, "-is_interwarp_coalescing_enabled", OPT_BOOL,
    &is_interwarp_coalescing_enabled,
    "If enabled, We enable the Interwarp coalescing techniques."
    "(default = 0 (disabled))",
    "0");
  option_parser_register(opp, "-num_interwarp_coalescing_tables", OPT_UINT32,
    &num_interwarp_coalescing_tables,
    "Number of interwarp coalescing tables."
    "(default = 1)",
    "1");
  option_parser_register(opp, "-max_size_interwarp_coalescing_per_table", OPT_UINT32,
    &max_size_interwarp_coalescing_per_table,
    "Size of each interwarp coalescing tables."
    "(default = 64)",
    "64");
  option_parser_register(opp, "-interwarp_coalescing_quanta", OPT_UINT32,
    &interwarp_coalescing_quanta,
    "Number of cycles used for the quanta in interwarp coalescing in case of being need."
    "(default = 100000)",
    "100000");
  option_parser_register(opp, "-interwarp_coalescing_quanta_warppool_policy_miss_ratio_threshold", OPT_DOUBLE,
    &interwarp_coalescing_quanta_warppool_policy_miss_ratio_threshold,
    "Threshold where beyond that miss ratio in that quanta, GTL_WARPID policy is applied if we use WARPPOOL_HYBRID. Otherwise, OLDEST policy is applied."
    "(default = 0.99)",
    "0.99");
  option_parser_register(opp, "-number_of_coalescers", OPT_UINT32, &number_of_coalescers,
    "Number of coalescers applaying intra-warp coalescing to different instruction in each of the coalescers."
    "(default = 1)",
    "1");
  option_parser_register(opp, "-number_of_clusters_for_prt_selection", OPT_UINT32, &number_of_clusters_for_prt_selection,
    "Number of cluster that are going to be used for PRT selection when WARPID_N_CLUSTERS_WITH_OLDEST is chosen ."
    "(default = 16)", "16");
  option_parser_register(
    opp, "-interwarp_coalescing_selection_policy_string", OPT_CSTR, &interwarp_coalescing_selection_policy_string,
    "interwarp_coalescing_selection_policy_string mode: < OLDEST | GTL_WARPID | SAME_LAST_LEADER_INST_PC_THEN_OLDEST | WARPPOOL_HYBRID | DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_GENERIC | DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_CHECKING_WARP_ID | DEP_COUNT_WAIT_DETECTED_AT_DECODE_GENERIC | DEP_COUNT_WAIT_DETECTED_AT_DECODE_CHECKING_WARP_ID > "
    "Default: OLDEST",
    "OLDEST");
  option_parser_register(
    opp, "-prt_selection_policy_string", OPT_CSTR, &prt_selection_policy_string,
    "prt_selection_policy_string mode: < OLDEST | SAME_LAST_WARP_ID_THEN_OLDEST | SAME_LAST_INST_PC_THEN_OLDEST | WARPID_N_CLUSTERS_WITH_OLDEST | DEP_COUNT_WAIT_GENERIC_THEN_OLDEST | DEP_COUNT_WAIT_CHECKING_WARP_ID_THEN_OLDEST > "
    "Default: OLDEST",
    "OLDEST");
  
  // MOD. End. InterWarp coalescing

  // Virtual memory parameters
  option_parser_register(opp, "-is_num_virtual_pages_tracking_enabled", OPT_BOOL,
                         &is_num_virtual_pages_tracking_enabled,
                         "If enabled, a set will track all the different virtual pages. Then it will report the different number of virtual pages accessed at the end of the execution."
                         "(default = disabled)",
                         "0");
  option_parser_register(opp, "-virtual_page_size_in_bytes", OPT_UINT32,
                         &virtual_page_size_in_bytes, "Size of a virtual page in bytes. Default 2MiB == 2097152."
                         "Configure to any positive number (default=2097152)",
                         "2097152");
}

void gpgpu_sim_config::reg_options(option_parser_t opp) {
  gpgpu_functional_sim_config::reg_options(opp);
  m_shader_config.reg_options(opp);
  m_memory_config.reg_options(opp);
  power_config::reg_options(opp);
  option_parser_register(opp, "-gpgpu_max_cycle", OPT_INT64, &gpu_max_cycle_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_insn", OPT_INT64, &gpu_max_insn_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_cta", OPT_INT32, &gpu_max_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_completed_cta", OPT_INT32,
                         &gpu_max_completed_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(
      opp, "-gpgpu_runtime_stat", OPT_CSTR, &gpgpu_runtime_stat,
      "display runtime statistics such as dram utilization {<freq>:<flag>}",
      "10000:0");
  option_parser_register(opp, "-liveness_message_freq", OPT_INT64,
                         &liveness_message_freq,
                         "Minimum number of seconds between simulation "
                         "liveness messages (0 = always print)",
                         "1");
  option_parser_register(opp, "-gpgpu_compute_capability_major", OPT_UINT32,
                         &gpgpu_compute_capability_major,
                         "Major compute capability version number", "7");
  option_parser_register(opp, "-gpgpu_compute_capability_minor", OPT_UINT32,
                         &gpgpu_compute_capability_minor,
                         "Minor compute capability version number", "0");
  option_parser_register(opp, "-gpgpu_flush_l1_cache", OPT_BOOL,
                         &gpgpu_flush_l1_cache,
                         "Flush L1 cache at the end of each kernel call", "0");
  option_parser_register(opp, "-gpgpu_flush_l2_cache", OPT_BOOL,
                         &gpgpu_flush_l2_cache,
                         "Flush L2 cache at the end of each kernel call", "0");
  option_parser_register(
      opp, "-gpgpu_deadlock_detect", OPT_BOOL, &gpu_deadlock_detect,
      "Stop the simulation at deadlock (1=on (default), 0=off)", "1");
  option_parser_register(
      opp, "-gpgpu_ptx_instruction_classification", OPT_INT32,
      &(gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification),
      "if enabled will classify ptx instruction types per kernel (Max 255 "
      "kernels now)",
      "0");
  option_parser_register(
      opp, "-gpgpu_ptx_sim_mode", OPT_INT32,
      &(gpgpu_ctx->func_sim->g_ptx_sim_mode),
      "Select between Performance (default) or Functional simulation (1)", "0");
  option_parser_register(opp, "-gpgpu_clock_domains", OPT_CSTR,
                         &gpgpu_clock_domains,
                         "Clock Domain Frequencies in MhZ {<Core Clock>:<ICNT "
                         "Clock>:<L2 Clock>:<DRAM Clock>}",
                         "500.0:2000.0:2000.0:2000.0");
  option_parser_register(
      opp, "-gpgpu_max_concurrent_kernel", OPT_INT32, &max_concurrent_kernel,
      "maximum kernels that can run concurrently on GPU, set this value "
      "according to max resident grids for your compute capability", "32");
  option_parser_register(
      opp, "-gpgpu_cflog_interval", OPT_INT32, &gpgpu_cflog_interval,
      "Interval between each snapshot in control flow logger", "0");
  option_parser_register(opp, "-visualizer_enabled", OPT_BOOL,
                         &g_visualizer_enabled,
                         "Turn on visualizer output (1=On, 0=Off)", "1");
  option_parser_register(opp, "-visualizer_outputfile", OPT_CSTR,
                         &g_visualizer_filename,
                         "Specifies the output log file for visualizer", NULL);
  option_parser_register(
      opp, "-visualizer_zlevel", OPT_INT32, &g_visualizer_zlevel,
      "Compression level of the visualizer output log (0=no comp, 9=highest)",
      "6");
  option_parser_register(opp, "-gpgpu_stack_size_limit", OPT_INT32,
                         &stack_size_limit, "GPU thread stack size", "1024");
  option_parser_register(opp, "-gpgpu_heap_size_limit", OPT_INT32,
                         &heap_size_limit, "GPU malloc heap size ", "8388608");
  option_parser_register(opp, "-gpgpu_runtime_sync_depth_limit", OPT_INT32,
                         &runtime_sync_depth_limit,
                         "GPU device runtime synchronize depth", "2");
  option_parser_register(opp, "-gpgpu_runtime_pending_launch_count_limit",
                         OPT_INT32, &runtime_pending_launch_count_limit,
                         "GPU device runtime pending launch count", "2048");
  option_parser_register(opp, "-trace_enabled", OPT_BOOL, &Trace::enabled,
                         "Turn on traces", "0");
  option_parser_register(opp, "-trace_components", OPT_CSTR, &Trace::config_str,
                         "comma seperated list of traces to enable. "
                         "Complete list found in trace_streams.tup. "
                         "Default none",
                         "none");
  option_parser_register(
      opp, "-trace_sampling_core", OPT_INT32, &Trace::sampling_core,
      "The core which is printed using CORE_DPRINTF. Default 0", "0");
  option_parser_register(opp, "-trace_sampling_memory_partition", OPT_INT32,
                         &Trace::sampling_memory_partition,
                         "The memory partition which is printed using "
                         "MEMPART_DPRINTF. Default -1 (i.e. all)",
                         "-1");
  gpgpu_ctx->stats->ptx_file_line_stats_options(opp);

  // Jin: kernel launch latency
  option_parser_register(opp, "-gpgpu_kernel_launch_latency", OPT_INT32,
                         &(gpgpu_ctx->device_runtime->g_kernel_launch_latency),
                         "Kernel launch latency in cycles. Default: 0", "0");
  option_parser_register(opp, "-gpgpu_cdp_enabled", OPT_BOOL,
                         &(gpgpu_ctx->device_runtime->g_cdp_enabled),
                         "Turn on CDP", "0");

  option_parser_register(opp, "-gpgpu_TB_launch_latency", OPT_INT32,
                         &(gpgpu_ctx->device_runtime->g_TB_launch_latency),
                         "thread block launch latency in cycles. Default: 0",
                         "0");
}
