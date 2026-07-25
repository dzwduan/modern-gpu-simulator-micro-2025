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

#include "access_queue.h"

#include <cassert>

#include "../../abstract_hardware_model.h"

namespace remodel {

AccessQueue::AccessQueue(unsigned int max_size) : m_max_size(max_size) {}

AccessQueue::~AccessQueue() {
  while(!m_accesses.empty()) {
    mem_access_t* inst = m_accesses.front();
    m_accesses.pop();
    delete inst;
  }
}

void AccessQueue::push(mem_access_t* access) {
  assert(m_accesses.size() < m_max_size);
  m_accesses.push(access);
}

void AccessQueue::pop() {
  assert(!m_accesses.empty());
  m_accesses.pop();
}

mem_access_t* AccessQueue::front() {
  assert(!m_accesses.empty());
  return m_accesses.front();
}

bool AccessQueue::full() {
  return m_accesses.size() == m_max_size;
}

bool AccessQueue::empty() {
  return m_accesses.empty();
}

unsigned int AccessQueue::size() {
  return m_accesses.size();
}

} // namespace remodel
