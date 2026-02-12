#!/bin/bash

export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"

NVBIT_VERSION=${NVBIT_VERSION:-"1.7.7.1"}

case "$NVBIT_VERSION" in
    1.7.6)
        SUPPORTED_ARCH="Volta (sm_70), Turing (sm_75), Ampere (sm_80/86), Ada Lovelace (sm_89), Hopper (sm_90), Blackwell (sm_100/110)"
        ;;
    1.7.7|1.7.7.1)
        SUPPORTED_ARCH="Volta (sm_70), Turing (sm_75), Ampere (sm_80/86), Ada Lovelace (sm_89), Hopper (sm_90), Blackwell (sm_100/110/120)"
        ;;
    *)
        echo "Warning: untested NVBit version $NVBIT_VERSION, proceeding anyway..."
        SUPPORTED_ARCH="unknown"
        ;;
esac

TARBALL="nvbit-Linux-x86_64-${NVBIT_VERSION}.tar.bz2"
URL="https://github.com/NVlabs/NVBit/releases/download/v${NVBIT_VERSION}/${TARBALL}"

rm -rf $BASH_ROOT/nvbit_release
echo "Downloading NVBit v${NVBIT_VERSION}..."
wget --no-check-certificate --no-proxy "$URL" -O "$TARBALL"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download NVBit v${NVBIT_VERSION} from $URL"
    exit 1
fi

tar -xf "$TARBALL" -C $BASH_ROOT
rm "$TARBALL"

if [ -d "$BASH_ROOT/nvbit_release_x86_64" ]; then
    mv $BASH_ROOT/nvbit_release_x86_64 $BASH_ROOT/nvbit_release
elif [ -d "$BASH_ROOT/nvbit_release" ]; then
    echo "nvbit_release directory already exists"
else
    echo "ERROR: Unexpected extraction directory structure"
    exit 1
fi

echo "NVBit v${NVBIT_VERSION} installed successfully at $BASH_ROOT/nvbit_release"
echo "Supported architectures: $SUPPORTED_ARCH"
