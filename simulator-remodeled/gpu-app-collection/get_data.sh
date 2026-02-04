#!/bin/bash
set -x
export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
DATA_ROOT="${BASH_ROOT}/data_dirs"
DOWNLOAD_DIR="${DATA_ROOT}/.downloads"

extract_to_data_root() {
    local archive="$1"
    if tar tzf "$archive" | head -n 1 | grep -q "^data_dirs/"; then
        tar xzvf "$archive" -C "$DATA_ROOT" --strip-components=1
    else
        tar xzvf "$archive" -C "$DATA_ROOT"
    fi
}

if [ ! -d "$DATA_ROOT" ]; then
    mkdir -p "$DOWNLOAD_DIR" "$DATA_ROOT"
    cd "$DOWNLOAD_DIR"
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/other-apps-part1.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/other-apps-part2.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/other-apps-part3.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/other-apps-part4.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/other-apps-part5.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/other-apps-part6.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/other-apps-part7.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/other-apps-part8.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/other-apps-part9.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/other-apps-part10.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/tango-data-part1.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/tango-data-part2.tar.gz
    extract_to_data_root other-apps-part1.tar.gz
    extract_to_data_root other-apps-part2.tar.gz
    extract_to_data_root other-apps-part3.tar.gz
    extract_to_data_root other-apps-part4.tar.gz
    extract_to_data_root other-apps-part5.tar.gz
    extract_to_data_root other-apps-part6.tar.gz
    extract_to_data_root other-apps-part7.tar.gz
    extract_to_data_root other-apps-part8.tar.gz
    extract_to_data_root other-apps-part9.tar.gz
    extract_to_data_root other-apps-part10.tar.gz
    extract_to_data_root tango-data-part1.tar.gz
    extract_to_data_root tango-data-part2.tar.gz
    rm other-apps-part1.tar.gz
    rm other-apps-part2.tar.gz
    rm other-apps-part3.tar.gz
    rm other-apps-part4.tar.gz
    rm other-apps-part5.tar.gz
    rm other-apps-part6.tar.gz
    rm other-apps-part7.tar.gz
    rm other-apps-part8.tar.gz
    rm other-apps-part9.tar.gz
    rm other-apps-part10.tar.gz
    rm tango-data-part1.tar.gz
    rm tango-data-part2.tar.gz
    mv "$DATA_ROOT"/tango/CifarNet "$DATA_ROOT"/tango/Tango-CN
    mv "$DATA_ROOT"/tango/ResNet "$DATA_ROOT"/tango/Tango-RN
    mv "$DATA_ROOT"/tango/LSTM "$DATA_ROOT"/tango/Tango-LSTM
    mv "$DATA_ROOT"/tango/GRU "$DATA_ROOT"/tango/Tango-GRU
    mv "$DATA_ROOT"/tango/AlexNet "$DATA_ROOT"/tango/Tango-AN
    mv "$DATA_ROOT"/tango/SqueezeNet "$DATA_ROOT"/tango/Tango-SN
    mkdir -p "$DATA_ROOT"/cuda/lonestargpu-2.0/lonestar-bfs-wla
    ln -s "$DATA_ROOT"/cuda/lonestargpu-2.0/inputs "$DATA_ROOT"/cuda/lonestargpu-2.0/lonestar-bfs-wla/data
fi
