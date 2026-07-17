#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ref_file="$repo_root/runtime/llama-cpp.ref"
source_dir="${PROBE_LLAMA_CPP_SOURCE:-$repo_root/.cache/llama.cpp}"
build_dir="${PROBE_LLAMA_CPP_BUILD:-$source_dir/build}"

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required. On macOS: brew install cmake" >&2
  exit 1
fi

if [[ ! -f "$ref_file" ]]; then
  echo "missing llama.cpp pin: $ref_file" >&2
  exit 1
fi
ref="$(tr -d '[:space:]' < "$ref_file")"
if [[ ! "$ref" =~ ^[0-9a-f]{40}$ ]]; then
  echo "invalid llama.cpp commit in $ref_file" >&2
  exit 1
fi

if [[ -d "$source_dir/.git" ]]; then
  origin="$(git -C "$source_dir" remote get-url origin)"
  if [[ "$origin" != "https://github.com/ggml-org/llama.cpp.git" && "$origin" != "git@github.com:ggml-org/llama.cpp.git" ]]; then
    echo "refusing to reuse $source_dir because its origin is $origin" >&2
    exit 1
  fi
  if [[ -n "$(git -C "$source_dir" status --porcelain)" ]]; then
    echo "refusing to change a dirty llama.cpp checkout: $source_dir" >&2
    exit 1
  fi
elif [[ -e "$source_dir" ]]; then
  echo "refusing to overwrite non-repository path: $source_dir" >&2
  exit 1
else
  git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git "$source_dir"
fi

git -C "$source_dir" fetch --depth=1 origin "$ref"
git -C "$source_dir" checkout --detach "$ref"

generator_args=()
if command -v ninja >/dev/null 2>&1; then
  generator_args=(-G Ninja)
fi

cmake -S "$source_dir" -B "$build_dir" \
  "${generator_args[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_ACCELERATE=ON \
  -DLLAMA_CURL=ON
cmake --build "$build_dir" --target llama-bench llama-server llama-cli --parallel

echo "Built pinned llama.cpp $ref"
echo "Binaries: $build_dir/bin"
