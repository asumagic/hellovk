#!/usr/bin/env bash

set -ex
cd "$(dirname "$0")"
target_dir="../../workdir/shaders"
glslc shader.frag -o $target_dir/frag.spv
glslc shader.vert -o $target_dir/vert.spv