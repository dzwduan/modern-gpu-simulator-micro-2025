// These are the configration parameters that can be found publicly
// Sources:
// https://www.nvidia.com/content/dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf
// https://en.wikipedia.org/wiki/GeForce_30_series
// https://en.wikipedia.org/wiki/CUDA

#ifndef AMPERE_RTX4090_DEF_H
#define AMPERE_RTX4090_DEF_H

#include "./common/common.h"
#include "./common/deviceQuery.h"

// L1 Cache: Ada Lovelace 每个 SM 为 128KB (统一 L1 Data + Shared Memory)
#define L1_SIZE (128 * 1024) // bytes per SM

// GPU 核心 Boost 频率 (MHz): RTX 4090 约为 2520 MHz
#define CLK_FREQUENCY 2520 // frequency in MHz

// Issue Model: Ada 架构每个 SM 分为 4 个分区 (Partitions)，每个分区 1 个 Warp Scheduler
#define ISSUE_MODEL issue_model::single // quad-issue (4 warp schedulers per SM)

// Core Model: Lovelace 继承 Volta/Ampere 的 Subcore 模式
#define CORE_MODEL core_model::subcore  // microarchitecture model

// 内存类型：RTX 4090 使用 GDDR6X
#define DRAM_MODEL dram_model::GDDR6    // GDDR6X memory model

// 每个 SM 的 warp scheduler 数量
#define WARP_SCHEDS_PER_SM 4            // Correct, 4 schedulers per SM

// CUDA/HMMA 单元: Ada 架构 Tensor Core 吞吐量巨大，此处参数根据模拟器具体实现调整
// 4th Gen Tensor Cores 性能强劲，通常设为 2 或 4 (Depends on sim instruction width implementation)
#define SASS_hmma_per_PTX_wmma 2         

// L2 Cache 划分
// 【修正】RTX 4090 的 L2 缓存为 72 MB (满血 AD102 才是 96 MB)
// 显存位宽 384-bit -> 12 个 Memory Channels (32-bit each)
// 每个 Channel 对应 1 个 L2 Bank (Slice)，即总共 12 个 L2 Banks
#define L2_BANKS_PER_MEM_CHANNEL 1      // 修正为 1 (Total 12 banks for 12 channels)
#define L2_BANK_WIDTH_in_BYTE 64        // Cache Line Size 通常为 64 或 128 Bytes



#endif