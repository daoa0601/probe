#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr const char * kTokenEmbedding = "token_embd.weight";
constexpr const char * kOutput = "output.weight";
constexpr size_t kCopyBufferBytes = 8u*1024u*1024u;
constexpr int64_t kQuantizeChunkRows = 1024;

std::string normalize_type_name(const char * value) {
    std::string normalized;
    for (const unsigned char ch : std::string(value)) {
        if (ch == '-' || ch == '_') {
            continue;
        }
        normalized.push_back(
            static_cast<char>(std::toupper(ch)));
    }
    return normalized;
}

ggml_type parse_type(const char * value) {
    const std::string requested = normalize_type_name(value);
    for (int type = 0; type < GGML_TYPE_COUNT; ++type) {
        const auto candidate = static_cast<ggml_type>(type);
        const char * name = ggml_type_name(candidate);
        if (name != nullptr && normalize_type_name(name) == requested) {
            return candidate;
        }
    }
    throw std::runtime_error(
        "unknown GGML type: " + std::string(value));
}

void write_zeros(std::ofstream & output, size_t bytes) {
    static const std::vector<char> zeros(4096, 0);
    while (bytes > 0) {
        const size_t count = std::min(bytes, zeros.size());
        output.write(zeros.data(), static_cast<std::streamsize>(count));
        bytes -= count;
    }
}

void copy_range(
        std::ifstream & input,
        std::ofstream & output,
        uint64_t offset,
        size_t bytes,
        std::vector<char> & buffer) {
    input.seekg(static_cast<std::streamoff>(offset));
    size_t remaining = bytes;
    while (remaining > 0) {
        const size_t count = std::min(remaining, buffer.size());
        input.read(buffer.data(), static_cast<std::streamsize>(count));
        output.write(buffer.data(), static_cast<std::streamsize>(count));
        remaining -= count;
    }
}

uint64_t padded_size(uint64_t bytes, uint64_t alignment) {
    return (bytes + alignment - 1)/alignment*alignment;
}

struct gguf_deleter {
    void operator()(gguf_context * context) const {
        gguf_free(context);
    }
};

struct ggml_deleter {
    void operator()(ggml_context * context) const {
        ggml_free(context);
    }
};

using gguf_ptr = std::unique_ptr<gguf_context, gguf_deleter>;
using ggml_ptr = std::unique_ptr<ggml_context, ggml_deleter>;

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        std::fprintf(
            stderr,
            "usage: %s INPUT.gguf OUTPUT.gguf OUTPUT_TYPE [--dry-run]\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    const std::filesystem::path input_path = argv[1];
    const std::filesystem::path output_path = argv[2];
    const bool dry_run = argc == 5 &&
        std::string(argv[4]) == "--dry-run";
    if (argc == 5 && !dry_run) {
        std::fprintf(stderr, "unknown option: %s\n", argv[4]);
        return EXIT_FAILURE;
    }

    std::filesystem::path partial_path =
        output_path.string() + ".partial";
    try {
        if (std::filesystem::absolute(input_path).lexically_normal() ==
            std::filesystem::absolute(output_path).lexically_normal()) {
            throw std::runtime_error(
                "input and output paths must differ");
        }
        if (!std::filesystem::is_regular_file(input_path)) {
            throw std::runtime_error("input is not a regular file");
        }
        if (!dry_run &&
            (std::filesystem::exists(output_path) ||
             std::filesystem::exists(partial_path))) {
            throw std::runtime_error(
                "output or partial output already exists");
        }

        const ggml_type output_type = parse_type(argv[3]);
        const ggml_type_traits * output_traits =
            ggml_get_type_traits(output_type);
        if (output_traits == nullptr ||
            !output_traits->is_quantized ||
            output_traits->from_float_ref == nullptr) {
            throw std::runtime_error(
                "output type is not a supported quantized type");
        }
        if (ggml_quantize_requires_imatrix(output_type)) {
            throw std::runtime_error(
                "output type requires an importance matrix");
        }

        ggml_context * metadata_raw = nullptr;
        gguf_init_params input_params = {
            /*.no_alloc =*/ true,
            /*.ctx      =*/ &metadata_raw,
        };
        gguf_ptr input_context(
            gguf_init_from_file(input_path.c_str(), input_params));
        ggml_ptr metadata_context(metadata_raw);
        if (!input_context || !metadata_context) {
            throw std::runtime_error("failed to read GGUF metadata");
        }
        if (gguf_get_alignment(input_context.get()) !=
            GGUF_DEFAULT_ALIGNMENT) {
            throw std::runtime_error(
                "non-default GGUF alignment is not supported");
        }
        if (gguf_find_tensor(input_context.get(), kOutput) >= 0) {
            throw std::runtime_error(
                "model already has an explicit output.weight");
        }
        const int64_t token_index =
            gguf_find_tensor(input_context.get(), kTokenEmbedding);
        if (token_index < 0) {
            throw std::runtime_error(
                "model has no token_embd.weight");
        }
        ggml_tensor * token = ggml_get_tensor(
            metadata_context.get(),
            kTokenEmbedding);
        if (token == nullptr) {
            throw std::runtime_error(
                "token embedding metadata is unavailable");
        }
        const int64_t columns = token->ne[0];
        const int64_t rows = ggml_nrows(token);
        if (columns%ggml_blck_size(output_type) != 0) {
            throw std::runtime_error(
                "token embedding row is not divisible by output block size");
        }
        const ggml_type_traits * input_traits =
            ggml_get_type_traits(token->type);
        if (input_traits == nullptr || input_traits->to_float == nullptr) {
            throw std::runtime_error(
                "token embedding type cannot be dequantized");
        }

        gguf_ptr output_context(gguf_init_empty());
        gguf_set_kv(output_context.get(), input_context.get());
        const int64_t input_tensor_count =
            gguf_get_n_tensors(input_context.get());
        for (int64_t index = 0; index < input_tensor_count; ++index) {
            ggml_tensor * tensor = ggml_get_tensor(
                metadata_context.get(),
                gguf_get_tensor_name(input_context.get(), index));
            if (tensor == nullptr) {
                throw std::runtime_error(
                    "input tensor metadata is unavailable");
            }
            gguf_add_tensor(output_context.get(), tensor);
        }

        ggml_init_params output_metadata_params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_ptr output_metadata(
            ggml_init(output_metadata_params));
        if (!output_metadata) {
            throw std::runtime_error(
                "failed to allocate output tensor metadata");
        }
        ggml_tensor * output_tensor = ggml_new_tensor(
            output_metadata.get(),
            output_type,
            ggml_n_dims(token),
            token->ne);
        ggml_set_name(output_tensor, kOutput);
        gguf_add_tensor(output_context.get(), output_tensor);

        const size_t input_row_bytes =
            ggml_row_size(token->type, columns);
        const size_t output_row_bytes =
            ggml_row_size(output_type, columns);
        const uint64_t output_tensor_bytes =
            static_cast<uint64_t>(output_row_bytes)*rows;
        const uint64_t metadata_bytes =
            gguf_get_meta_size(output_context.get());
        uint64_t expected_file_bytes = metadata_bytes;
        for (int64_t index = 0;
             index < gguf_get_n_tensors(output_context.get());
             ++index) {
            expected_file_bytes += padded_size(
                gguf_get_tensor_size(output_context.get(), index),
                GGUF_DEFAULT_ALIGNMENT);
        }

        std::printf(
            "{\"input_tensors\":%lld,\"output_tensors\":%lld,"
            "\"embedding_type\":\"%s\",\"output_type\":\"%s\","
            "\"columns\":%lld,\"rows\":%lld,"
            "\"output_tensor_bytes\":%llu,"
            "\"expected_file_bytes\":%llu,\"dry_run\":%s}\n",
            static_cast<long long>(input_tensor_count),
            static_cast<long long>(
                gguf_get_n_tensors(output_context.get())),
            ggml_type_name(token->type),
            ggml_type_name(output_type),
            static_cast<long long>(columns),
            static_cast<long long>(rows),
            static_cast<unsigned long long>(output_tensor_bytes),
            static_cast<unsigned long long>(expected_file_bytes),
            dry_run ? "true" : "false");
        if (dry_run) {
            return EXIT_SUCCESS;
        }

        const auto start = std::chrono::steady_clock::now();
        std::ifstream input(input_path, std::ios::binary);
        std::ofstream output(partial_path, std::ios::binary);
        input.exceptions(
            std::ifstream::badbit | std::ifstream::failbit);
        output.exceptions(
            std::ofstream::badbit | std::ofstream::failbit);

        std::vector<uint8_t> metadata(metadata_bytes);
        gguf_get_meta_data(output_context.get(), metadata.data());
        output.write(
            reinterpret_cast<const char *>(metadata.data()),
            static_cast<std::streamsize>(metadata.size()));

        std::vector<char> copy_buffer(kCopyBufferBytes);
        for (int64_t index = 0; index < input_tensor_count; ++index) {
            const size_t bytes =
                gguf_get_tensor_size(input_context.get(), index);
            const uint64_t offset =
                gguf_get_data_offset(input_context.get()) +
                gguf_get_tensor_offset(input_context.get(), index);
            copy_range(
                input,
                output,
                offset,
                bytes,
                copy_buffer);
            write_zeros(
                output,
                padded_size(bytes, GGUF_DEFAULT_ALIGNMENT) - bytes);
        }

        const uint64_t token_offset =
            gguf_get_data_offset(input_context.get()) +
            gguf_get_tensor_offset(input_context.get(), token_index);
        input.seekg(static_cast<std::streamoff>(token_offset));
        const int64_t chunk_capacity =
            std::min(kQuantizeChunkRows, rows);
        std::vector<uint8_t> source(
            static_cast<size_t>(chunk_capacity)*input_row_bytes);
        std::vector<float> dequantized(
            static_cast<size_t>(chunk_capacity)*columns);
        std::vector<uint8_t> quantized(
            static_cast<size_t>(chunk_capacity)*output_row_bytes);
        const unsigned hardware_threads =
            std::max(1u, std::thread::hardware_concurrency());

        for (int64_t row = 0; row < rows;
             row += chunk_capacity) {
            const int64_t chunk_rows =
                std::min(chunk_capacity, rows - row);
            const size_t source_bytes =
                static_cast<size_t>(chunk_rows)*input_row_bytes;
            input.read(
                reinterpret_cast<char *>(source.data()),
                static_cast<std::streamsize>(source_bytes));

            const unsigned thread_count = std::min<unsigned>(
                hardware_threads,
                static_cast<unsigned>(chunk_rows));
            std::vector<std::thread> workers;
            workers.reserve(thread_count);
            std::vector<size_t> written(thread_count, 0);
            for (unsigned worker = 0; worker < thread_count; ++worker) {
                const int64_t first =
                    chunk_rows*worker/thread_count;
                const int64_t last =
                    chunk_rows*(worker + 1)/thread_count;
                workers.emplace_back([&, worker, first, last]() {
                    for (int64_t local_row = first;
                         local_row < last;
                         ++local_row) {
                        input_traits->to_float(
                            source.data() +
                                static_cast<size_t>(local_row)*
                                input_row_bytes,
                            dequantized.data() +
                                static_cast<size_t>(local_row)*columns,
                            columns);
                    }
                    written[worker] = ggml_quantize_chunk(
                        output_type,
                        dequantized.data() +
                            static_cast<size_t>(first)*columns,
                        quantized.data() +
                            static_cast<size_t>(first)*
                            output_row_bytes,
                        0,
                        last - first,
                        columns,
                        nullptr);
                });
            }
            for (auto & worker : workers) {
                worker.join();
            }
            size_t chunk_output_bytes = 0;
            for (const size_t worker_bytes : written) {
                chunk_output_bytes += worker_bytes;
            }
            const size_t expected_chunk_bytes =
                static_cast<size_t>(chunk_rows)*output_row_bytes;
            if (chunk_output_bytes != expected_chunk_bytes ||
                !ggml_validate_row_data(
                    output_type,
                    quantized.data(),
                    expected_chunk_bytes)) {
                throw std::runtime_error(
                    "quantizer produced invalid output rows");
            }
            output.write(
                reinterpret_cast<const char *>(quantized.data()),
                static_cast<std::streamsize>(expected_chunk_bytes));
        }
        write_zeros(
            output,
            padded_size(
                output_tensor_bytes,
                GGUF_DEFAULT_ALIGNMENT) - output_tensor_bytes);
        output.close();
        input.close();

        const uint64_t actual_file_bytes =
            std::filesystem::file_size(partial_path);
        if (actual_file_bytes != expected_file_bytes) {
            throw std::runtime_error(
                "output size does not match GGUF metadata");
        }
        std::filesystem::rename(partial_path, output_path);
        const double elapsed_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
        std::printf(
            "{\"status\":\"ok\",\"output\":\"%s\","
            "\"bytes\":%llu,\"elapsed_seconds\":%.3f}\n",
            output_path.c_str(),
            static_cast<unsigned long long>(actual_file_bytes),
            elapsed_seconds);
        ggml_quantize_free();
        return EXIT_SUCCESS;
    } catch (const std::exception & error) {
        if (std::filesystem::exists(partial_path)) {
            std::error_code remove_error;
            std::filesystem::remove(partial_path, remove_error);
        }
        std::fprintf(stderr, "gguf-untie-output: %s\n", error.what());
        ggml_quantize_free();
        return EXIT_FAILURE;
    }
}
