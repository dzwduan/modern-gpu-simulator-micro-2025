#ifndef ICNT_HANDLER_H
#define ICNT_HANDLER_H

#include <memory>
#include "gpu-sim.h"

class gpgpu_sim;
class mem_fetch;
struct grid_barrier_notify_info;

class icnt_handler {
 public:
  explicit icnt_handler(gpgpu_sim &gpu);

  void icnt_tick(unsigned &partiton_replys_in_parallel_per_cycle);
  void icnt_transfer_tick();
  std::unique_ptr<grid_barrier_notify_info> register_grid_barrier_arrivement(
      mem_fetch *mf);

 private:
  gpgpu_sim &m_gpu;
};

#endif
