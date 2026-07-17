// Copyright (c) 2007-2012, Trustees of The Leland Stanford Junior University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE.

// Module and Stats statistics utilities derived from Booksim, retained after
// the intersim2 interconnect retirement because the DRAM, core, and functional
// statistics paths depend on them through intersim2_stats.h.

#include "intersim2_stats.h"

#include <cmath>
#include <limits>

Module::Module(Module *parent, const string &name) {
  _name = name;

  if (parent) {
    parent->_AddChild(this);
    _fullname = parent->_fullname + "/" + name;
  } else {
    _fullname = name;
  }
}

void Module::_AddChild(Module *child) { _children.push_back(child); }

void Module::DisplayHierarchy(int level, ostream &os) const {
  vector<Module *>::const_iterator mod_iter;

  for (int l = 0; l < level; l++) {
    os << "  ";
  }

  os << _name << endl;

  for (mod_iter = _children.begin(); mod_iter != _children.end(); mod_iter++) {
    (*mod_iter)->DisplayHierarchy(level + 1);
  }
}

void Module::Error(const string &msg) const {
  cout << "Error in " << _fullname << " : " << msg << endl;
  exit(-1);
}

void Module::Debug(const string &msg) const {
  cout << "Debug (" << _fullname << ") : " << msg << endl;
}

void Module::Display(ostream &os) const {
  os << "Display method not implemented for " << _fullname << endl;
}

Stats::Stats(Module *parent, const string &name, double bin_size, int num_bins)
    : Module(parent, name), _num_bins(num_bins), _bin_size(bin_size) {
  Clear();
}

void Stats::Clear() {
  _num_samples = 0;
  _sample_sum = 0.0;
  _sample_squared_sum = 0.0;

  _hist.assign(_num_bins, 0);

  _min = numeric_limits<double>::quiet_NaN();
  _max = -numeric_limits<double>::quiet_NaN();
}

double Stats::Average() const { return _sample_sum / (double)_num_samples; }

double Stats::Variance() const {
  return (_sample_squared_sum * (double)_num_samples -
          _sample_sum * _sample_sum) /
         ((double)_num_samples * (double)_num_samples);
}

double Stats::Min() const { return _min; }

double Stats::Max() const { return _max; }

double Stats::Sum() const { return _sample_sum; }

double Stats::SquaredSum() const { return _sample_squared_sum; }

int Stats::NumSamples() const { return _num_samples; }

void Stats::AddSample(double val) {
  ++_num_samples;
  _sample_sum += val;

  // NOTE: the negation ensures that NaN values are handled correctly!
  _max = !(val <= _max) ? val : _max;
  _min = !(val >= _min) ? val : _min;

  // double clamp between 0 and num_bins-1
  int b = (int)fmax(floor(val / _bin_size), 0.0);
  b = (b >= _num_bins) ? (_num_bins - 1) : b;

  _hist[b]++;
}

void Stats::Display(ostream &os) const { os << *this << endl; }

ostream &operator<<(ostream &os, const Stats &s) {
  vector<int> const &v = s._hist;
  os << "[ ";
  for (size_t i = 0; i < v.size(); ++i) {
    os << v[i] << " ";
  }
  os << "]";
  return os;
}
