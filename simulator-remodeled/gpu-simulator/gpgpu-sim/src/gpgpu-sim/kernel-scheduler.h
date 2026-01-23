#ifndef KERNEL_SCHEDULER_H
#define KERNEL_SCHEDULER_H

#include "gpu-sim.h"

class gpgpu_sim;
class kernel_info_t;

class kernel_scheduler {
 public:
  explicit kernel_scheduler(gpgpu_sim &gpu);

  void launch(kernel_info_t *kinfo);
  bool can_start_kernel() const;
  kernel_info_t *select_kernel();
  void set_kernel_done(kernel_info_t *kernel);

 private:
  gpgpu_sim &m_gpu;
};

#endif
