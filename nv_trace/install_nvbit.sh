#!/bin/bash
# Install NVBit binary instrumentation framework
# NVBit v1.7.6 supports Volta, Turing, Ampere, Ada Lovelace (RTX 4090), Hopper

export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"

rm -rf $BASH_ROOT/nvbit_release
echo "Downloading NVBit v1.7.6..."
wget https://github.com/NVlabs/NVBit/releases/download/v1.7.6/nvbit-Linux-x86_64-1.7.6.tar.bz2
tar -xf nvbit-Linux-x86_64-1.7.6.tar.bz2 -C $BASH_ROOT
rm nvbit-Linux-x86_64-1.7.6.tar.bz2
mv $BASH_ROOT/nvbit_release_x86_64 $BASH_ROOT/nvbit_release

echo "NVBit installed successfully at $BASH_ROOT/nvbit_release"
echo "Supported architectures: Volta (sm_70), Turing (sm_75), Ampere (sm_80/86), Ada Lovelace (sm_89), Hopper (sm_90)"
