#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
CLEAN_BUILD=false
TEST_TARGET="testbed"
VERBOSE=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Compile and run test(s) for ComputeEngine.

Options:
  --clean       Clean build before running
  --test NAME   Specify test target (default: testbed)
  --verbose     Verbose build output
  --help        Show this help message
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)     CLEAN_BUILD=true; shift ;;
        --test)      TEST_TARGET="$2"; shift 2 ;;
        --verbose)   VERBOSE=true; shift ;;
        --help)      usage ;;
        *)           echo "Unknown option: $1"; usage ;;
    esac
done

if [[ "$CLEAN_BUILD" == true && -d "$BUILD_DIR" ]]; then
    echo "=== Cleaning build directory ==="
    rm -rf "$BUILD_DIR"
fi

echo "=== Configuring CMake ==="
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
    -DCE_BUILD_CUDA=ON \
    -DCE_BUILD_OPENCL=ON \
    -DCE_BUILD_VULKAN=ON \
    -DCE_BUILD_TESTBED=ON \
    -DCE_BUILD_SHARED=OFF

echo "=== Building target: ${TEST_TARGET} ==="
if [[ "$VERBOSE" == true ]]; then
    cmake --build "$BUILD_DIR" --target "$TEST_TARGET" --verbose -j"$(nproc)"
else
    cmake --build "$BUILD_DIR" --target "$TEST_TARGET" -j"$(nproc)"
fi

echo "=== Running ${TEST_TARGET} ==="
"${BUILD_DIR}/testbed/${TEST_TARGET}"
echo "=== Done (exit code: $?) ==="
