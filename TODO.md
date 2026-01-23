# TODO


## 跑通4090 + cuda12.6对齐流程, 走trace driven

  相比于modeled版本，simulator-remodeled/util/tuner/NVIDIA_GeForce_RTX_4090 少了额外加的部分内容，应该怎么生成？


gpu-sim.cc 目前承担了配置注册、kernel 调度、时钟域驱动、统计/功耗、调试工具等多重职责，修改任何一块都容易牵一发动全身。

  - 抽出统计/功耗聚合：将 create_gpu_per_sm_stats、gather/reset、gpu_print_stat、print_stats 迁到 gpu-sim-stats.cc，集中管理统计输出与清理逻辑，避免和调度/周期逻辑互相依赖。
  - 拆出 kernel 调度器：把 launch/select_kernel/can_start_kernel/set_kernel_done 相关逻辑放进一个 kernel_scheduler 类，gpgpu_sim 只持有它并在 cycle() 中调用。
  - 拆出时钟域与周期驱动：把 next_clock_domain 与 cycle 中对 CORE/ICNT/DRAM/L2 的驱动分成独立的子系统 tick 方法（例如 core_tick、icnt_tick 等），gpgpu_sim 只做编排。
  - 把 grid barrier/ICNT 处理抽成小模块，避免 cycle() 过长且逻辑交织。
  - 明确模块接口：尽量用前向声明和轻量上下文结构体（如 CycleContext）传递必要字段，减少头文件耦合。
