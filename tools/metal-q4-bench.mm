#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr NSUInteger kFunctionConstantMulMv = 600;
constexpr int32_t kSmallQuantBlockElements = 32;
constexpr int32_t kQ6KBlockElements = 256;
constexpr int32_t kQ4BlockBytes = 18;
constexpr int32_t kQ8BlockBytes = 34;
constexpr int32_t kQ6KBlockBytes = 210;
constexpr int32_t kQ4RowsPerSimdgroup = 4;
constexpr int32_t kQ8RowsPerThreadgroup = 2;
constexpr int32_t kQ6KRowsPerSimdgroup = 2;
constexpr int32_t kWarmupSubmissions = 4;
constexpr int32_t kWarmupDispatches = 64;

struct __attribute__((packed)) Q8ActivationBlock {
    _Float16 d;
    int8_t qs[kSmallQuantBlockElements];
};

static_assert(sizeof(Q8ActivationBlock) == kQ8BlockBytes);

struct alignas(8) MulMvArgs {
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne10;
    int32_t ne11;
    int32_t ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    int32_t nr0;
    int16_t r2;
    int16_t r3;
};

static_assert(sizeof(MulMvArgs) == 112);

void print_usage(const char * program) {
    std::fprintf(
        stderr,
        "usage: %s METALLIB FUNCTION NSG NE00 NE01 [SAMPLES] "
        "[DISPATCHES_PER_SAMPLE [COMPARE_FUNCTION COMPARE_NSG]]\n",
        program);
}

bool parse_positive_int(const char * text, const char * name, int32_t * value) {
    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0 || parsed > INT32_MAX) {
        std::fprintf(stderr, "invalid %s value: %s\n", name, text);
        return false;
    }
    *value = static_cast<int32_t>(parsed);
    return true;
}

void print_error(NSString * context, NSError * error) {
    std::fprintf(
        stderr,
        "%s: %s\n",
        context.UTF8String,
        error.localizedDescription.UTF8String);
}

NSString * json_escape(NSString * value) {
    NSData * data = [NSJSONSerialization dataWithJSONObject:@[value] options:0 error:nil];
    NSString * array = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    return [array substringWithRange:NSMakeRange(1, array.length - 2)];
}

double mean(const std::vector<double> & values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double sample_stddev(const std::vector<double> & values, double average) {
    if (values.size() < 2) {
        return 0.0;
    }
    double squared_error = 0.0;
    for (double value : values) {
        const double error = value - average;
        squared_error += error * error;
    }
    return std::sqrt(squared_error / (values.size() - 1));
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() % 2 != 0) {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) / 2.0;
}

void initialize_quant_weights(
        id<MTLBuffer> buffer,
        size_t block_count,
        int32_t block_bytes,
        bool is_q6_k) {
    auto * bytes = static_cast<uint8_t *>(buffer.contents);
    for (size_t block = 0; block < block_count; ++block) {
        uint8_t * current = bytes + block * static_cast<size_t>(block_bytes);
        if (is_q6_k) {
            for (int byte = 0; byte < 3*kQ6KBlockElements/4; ++byte) {
                current[byte] =
                    static_cast<uint8_t>((block * 29 + byte * 17) & 0xff);
            }
            auto * scales = reinterpret_cast<int8_t *>(
                current + 3*kQ6KBlockElements/4);
            for (int scale = 0; scale < kQ6KBlockElements/16; ++scale) {
                scales[scale] = static_cast<int8_t>(
                    ((block * 11 + scale * 7) % 31) - 15);
            }
            current[kQ6KBlockBytes - 2] = 0x00;
            current[kQ6KBlockBytes - 1] = 0x3c;  // binary16 1.0
            continue;
        }
        current[0] = 0x00;
        current[1] = 0x3c;  // IEEE-754 binary16 1.0
        for (int byte = 2; byte < block_bytes; ++byte) {
            current[byte] = static_cast<uint8_t>((block * 29 + byte * 17) & 0xff);
        }
    }
}

float q6_k_value(const uint8_t * block, int32_t element) {
    const int32_t half = element / 128;
    const int32_t within_half = element % 128;
    const int32_t band = within_half / 32;
    const int32_t lane = within_half % 32;
    const uint8_t * ql = block + half*64;
    const uint8_t * qh =
        block + kQ6KBlockElements/2 + half*32;
    const auto * scales = reinterpret_cast<const int8_t *>(
        block + 3*kQ6KBlockElements/4 + half*8);
    const int32_t scale_group = lane/16;

    int32_t quant = 0;
    int32_t scale_index = 0;
    switch (band) {
        case 0:
            quant =
                (ql[lane] & 0x0f) |
                (((qh[lane] >> 0) & 0x03) << 4);
            scale_index = scale_group + 0;
            break;
        case 1:
            quant =
                (ql[lane + 32] & 0x0f) |
                (((qh[lane] >> 2) & 0x03) << 4);
            scale_index = scale_group + 2;
            break;
        case 2:
            quant =
                (ql[lane] >> 4) |
                (((qh[lane] >> 4) & 0x03) << 4);
            scale_index = scale_group + 4;
            break;
        case 3:
            quant =
                (ql[lane + 32] >> 4) |
                (((qh[lane] >> 6) & 0x03) << 4);
            scale_index = scale_group + 6;
            break;
    }
    return static_cast<float>(
        scales[scale_index] * (quant - 32));
}

void initialize_activations(id<MTLBuffer> buffer, int32_t element_count) {
    auto * values = static_cast<float *>(buffer.contents);
    for (int32_t index = 0; index < element_count; ++index) {
        values[index] = static_cast<float>((index % 31) - 15) / 16.0f;
    }
}

void quantize_activations_q8_0(
        id<MTLBuffer> f32_buffer,
        id<MTLBuffer> q8_buffer,
        int32_t element_count) {
    auto * values = static_cast<float *>(f32_buffer.contents);
    auto * blocks = static_cast<Q8ActivationBlock *>(q8_buffer.contents);
    const int32_t block_count = element_count/kSmallQuantBlockElements;
    for (int32_t block = 0; block < block_count; ++block) {
        float absolute_max = 0.0f;
        for (int32_t element = 0;
             element < kSmallQuantBlockElements;
             ++element) {
            absolute_max = std::max(
                absolute_max,
                std::abs(
                    values[
                        block*kSmallQuantBlockElements + element]));
        }
        const float scale =
            absolute_max == 0.0f ? 0.0f : absolute_max/127.0f;
        blocks[block].d = static_cast<_Float16>(scale);
        const float stored_scale = static_cast<float>(blocks[block].d);
        for (int32_t element = 0;
             element < kSmallQuantBlockElements;
             ++element) {
            const int32_t quantized =
                stored_scale == 0.0f
                    ? 0
                    : std::clamp(
                        static_cast<int32_t>(std::round(
                            values[
                                block*kSmallQuantBlockElements + element] /
                            stored_scale)),
                        -127,
                        127);
            blocks[block].qs[element] =
                static_cast<int8_t>(quantized);
            values[block*kSmallQuantBlockElements + element] =
                stored_scale*static_cast<float>(quantized);
        }
    }
}

}  // namespace

int main(int argc, const char * argv[]) {
    if (argc < 6 || argc > 10 || argc == 9) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    int32_t nsg = 0;
    int32_t ne00 = 0;
    int32_t ne01 = 0;
    int32_t sample_count = 7;
    int32_t dispatches_per_sample = 5;
    int32_t compare_nsg = 0;
    const bool compare = argc == 10;
    if (!parse_positive_int(argv[3], "nsg", &nsg) ||
        !parse_positive_int(argv[4], "ne00", &ne00) ||
        !parse_positive_int(argv[5], "ne01", &ne01) ||
        (argc >= 7 && !parse_positive_int(argv[6], "samples", &sample_count)) ||
        (argc >= 8 &&
         !parse_positive_int(argv[7], "dispatches_per_sample", &dispatches_per_sample)) ||
        (compare && !parse_positive_int(argv[9], "compare_nsg", &compare_nsg))) {
        return EXIT_FAILURE;
    }
    if (nsg > INT16_MAX ||
        compare_nsg > INT16_MAX ||
        ne00 % kSmallQuantBlockElements != 0) {
        std::fprintf(
            stderr,
            "nsg values must fit int16 and ne00 must be divisible by 32\n");
        return EXIT_FAILURE;
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            std::fprintf(stderr, "no Metal device is available\n");
            return EXIT_FAILURE;
        }

        NSError * error = nil;
        NSURL * library_url =
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id<MTLLibrary> library = [device newLibraryWithURL:library_url error:&error];
        if (library == nil) {
            print_error(@"load metallib", error);
            return EXIT_FAILURE;
        }

        NSString * function_name = [NSString stringWithUTF8String:argv[2]];
        NSString * compare_function_name =
            compare ? [NSString stringWithUTF8String:argv[8]] : nil;
        const bool is_q8 =
            [function_name rangeOfString:@"mul_mv_q8_0_f32"].location !=
                NSNotFound;
        const bool is_q6_k =
            [function_name rangeOfString:@"q6_K"].location != NSNotFound;
        const bool uses_q8_activation =
            [function_name rangeOfString:@"q4_0_q8_0"].location !=
                NSNotFound;
        const bool is_rows_kernel =
            [function_name rangeOfString:@"_rows_"].location != NSNotFound;
        const bool compare_is_rows_kernel =
            compare &&
            [compare_function_name rangeOfString:@"_rows_"].location !=
                NSNotFound;
        const bool compare_is_q8 =
            compare &&
            [compare_function_name
                rangeOfString:@"mul_mv_q8_0_f32"].location !=
                NSNotFound;
        const bool compare_is_q6_k =
            compare &&
            [compare_function_name rangeOfString:@"q6_K"].location !=
                NSNotFound;
        const bool compare_uses_q8_activation =
            compare &&
            [compare_function_name rangeOfString:@"q4_0_q8_0"].location !=
                NSNotFound;
        if (is_q8 && is_q6_k) {
            std::fprintf(stderr, "ambiguous primary quantization\n");
            return EXIT_FAILURE;
        }
        if (is_q6_k && ne00 % kQ6KBlockElements != 0) {
            std::fprintf(
                stderr,
                "Q6_K ne00 must be divisible by 256\n");
            return EXIT_FAILURE;
        }
        if (compare &&
            (compare_is_q8 != is_q8 ||
             compare_is_q6_k != is_q6_k)) {
            std::fprintf(
                stderr,
                "primary and comparison functions must use the same quantization\n");
            return EXIT_FAILURE;
        }

        const auto nr0_for_function =
            [is_q8, is_q6_k](NSString * name) -> int32_t {
            if ([name hasSuffix:@"_r1"]) {
                return 1;
            }
            if ([name hasSuffix:@"_r2"]) {
                return 2;
            }
            if ([name hasSuffix:@"_r4"]) {
                return 4;
            }
            if ([name hasSuffix:@"_r8"]) {
                return 8;
            }
            if ([name hasSuffix:@"_r16"]) {
                return 16;
            }
            return is_q8
                ? kQ8RowsPerThreadgroup
                : (is_q6_k
                    ? kQ6KRowsPerSimdgroup
                    : kQ4RowsPerSimdgroup);
        };
        const auto make_pipeline =
            [&](NSString * name, int32_t pipeline_nsg)
                -> id<MTLComputePipelineState> {
            const int16_t nsg_i16 = static_cast<int16_t>(pipeline_nsg);
            const int16_t one = 1;
            MTLFunctionConstantValues * constants =
                [[MTLFunctionConstantValues alloc] init];
            [constants setConstantValue:&nsg_i16
                                   type:MTLDataTypeShort
                                atIndex:kFunctionConstantMulMv + 0];
            [constants setConstantValue:&one
                                   type:MTLDataTypeShort
                                atIndex:kFunctionConstantMulMv + 2];
            [constants setConstantValue:&one
                                   type:MTLDataTypeShort
                                atIndex:kFunctionConstantMulMv + 3];
            [constants setConstantValue:&one
                                   type:MTLDataTypeShort
                                atIndex:kFunctionConstantMulMv + 4];

            id<MTLFunction> function =
                [library newFunctionWithName:name
                              constantValues:constants
                                       error:&error];
            if (function == nil) {
                print_error(@"specialize function", error);
                return nil;
            }

            MTLComputePipelineDescriptor * descriptor =
                [[MTLComputePipelineDescriptor alloc] init];
            descriptor.label = name;
            descriptor.computeFunction = function;
            descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
            id<MTLComputePipelineState> result =
                [device newComputePipelineStateWithDescriptor:descriptor
                                                      options:MTLPipelineOptionNone
                                                   reflection:nil
                                                        error:&error];
            if (result == nil) {
                print_error(@"compile pipeline", error);
            }
            return result;
        };

        const int32_t primary_nr0 = nr0_for_function(function_name);
        const int32_t compare_nr0 =
            compare ? nr0_for_function(compare_function_name) : 0;
        const int32_t quant_block_elements =
            is_q6_k
                ? kQ6KBlockElements
                : kSmallQuantBlockElements;
        const int32_t block_bytes =
            is_q8
                ? kQ8BlockBytes
                : (is_q6_k ? kQ6KBlockBytes : kQ4BlockBytes);
        const int32_t rows_per_threadgroup =
            is_q8
                ? primary_nr0*(is_rows_kernel ? nsg : 1)
                : primary_nr0*nsg;
        const int32_t compare_rows_per_threadgroup =
            is_q8
                ? compare_nr0*(compare_is_rows_kernel ? compare_nsg : 1)
                : compare_nr0*compare_nsg;
        const int32_t nr0 =
            primary_nr0;
        const NSUInteger threadgroup_memory_bytes =
            is_q8 && !is_rows_kernel
                ? 32*sizeof(float)*static_cast<NSUInteger>(primary_nr0)
                : 0;
        const NSUInteger compare_threadgroup_memory_bytes =
            is_q8 && !compare_is_rows_kernel
                ? 32*sizeof(float)*static_cast<NSUInteger>(compare_nr0)
                : 0;
        id<MTLComputePipelineState> pipeline =
            make_pipeline(function_name, nsg);
        id<MTLComputePipelineState> compare_pipeline =
            compare ? make_pipeline(compare_function_name, compare_nsg) : nil;
        if (pipeline == nil || (compare && compare_pipeline == nil)) {
            return EXIT_FAILURE;
        }

        const size_t blocks_per_row =
            static_cast<size_t>(ne00 / quant_block_elements);
        const size_t block_count = blocks_per_row * static_cast<size_t>(ne01);
        const size_t weight_bytes =
            block_count * static_cast<size_t>(block_bytes);
        const size_t activation_bytes = static_cast<size_t>(ne00) * sizeof(float);
        const size_t q8_activation_bytes =
            static_cast<size_t>(ne00/kSmallQuantBlockElements) *
            sizeof(Q8ActivationBlock);
        const size_t output_bytes = static_cast<size_t>(ne01) * sizeof(float);

        id<MTLBuffer> weights =
            [device newBufferWithLength:weight_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> activations =
            [device newBufferWithLength:activation_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> q8_activations =
            [device newBufferWithLength:q8_activation_bytes
                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> output =
            [device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
        if (weights == nil ||
            activations == nil ||
            q8_activations == nil ||
            output == nil) {
            std::fprintf(stderr, "unable to allocate benchmark buffers\n");
            return EXIT_FAILURE;
        }
        initialize_quant_weights(
            weights,
            block_count,
            block_bytes,
            is_q6_k);
        initialize_activations(activations, ne00);
        if (uses_q8_activation || compare_uses_q8_activation) {
            quantize_activations_q8_0(
                activations,
                q8_activations,
                ne00);
        }
        std::memset(output.contents, 0, output_bytes);

        const uint64_t weight_row_bytes =
            blocks_per_row * static_cast<uint64_t>(block_bytes);
        MulMvArgs args = {
            ne00,
            ne01,
            1,
            static_cast<uint64_t>(block_bytes),
            weight_row_bytes,
            weight_row_bytes * static_cast<uint64_t>(ne01),
            weight_row_bytes * static_cast<uint64_t>(ne01),
            ne00,
            1,
            1,
            sizeof(float),
            activation_bytes,
            activation_bytes,
            activation_bytes,
            ne01,
            1,
            nr0,
            1,
            1,
        };
        MulMvArgs compare_args = args;
        compare_args.nr0 = compare_nr0;
        if (uses_q8_activation) {
            args.nb10 = sizeof(Q8ActivationBlock);
            args.nb11 = q8_activation_bytes;
            args.nb12 = q8_activation_bytes;
            args.nb13 = q8_activation_bytes;
        }
        if (compare_uses_q8_activation) {
            compare_args.nb10 = sizeof(Q8ActivationBlock);
            compare_args.nb11 = q8_activation_bytes;
            compare_args.nb12 = q8_activation_bytes;
            compare_args.nb13 = q8_activation_bytes;
        }

        id<MTLBuffer> primary_activation_buffer =
            uses_q8_activation ? q8_activations : activations;
        id<MTLBuffer> compare_activation_buffer =
            compare_uses_q8_activation ? q8_activations : activations;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            std::fprintf(stderr, "unable to create Metal command queue\n");
            return EXIT_FAILURE;
        }

        const NSUInteger output_rows_per_threadgroup =
            static_cast<NSUInteger>(rows_per_threadgroup);
        const MTLSize threadgroups = MTLSizeMake(
            (static_cast<NSUInteger>(ne01) + output_rows_per_threadgroup - 1) /
                output_rows_per_threadgroup,
            1,
            1);
        const MTLSize threads = MTLSizeMake(32, static_cast<NSUInteger>(nsg), 1);
        const NSUInteger compare_output_rows_per_threadgroup =
            static_cast<NSUInteger>(compare_rows_per_threadgroup);
        const MTLSize compare_threadgroups = MTLSizeMake(
            compare
                ? (static_cast<NSUInteger>(ne01) +
                   compare_output_rows_per_threadgroup - 1) /
                    compare_output_rows_per_threadgroup
                : 1,
            1,
            1);
        const MTLSize compare_threads =
            MTLSizeMake(32, static_cast<NSUInteger>(compare ? compare_nsg : 1), 1);

        auto run_dispatches = [&](
                                  id<MTLComputePipelineState> selected_pipeline,
                                  const MulMvArgs & selected_args,
                                  id<MTLBuffer> selected_activations,
                                  MTLSize selected_threadgroups,
                                  MTLSize selected_threads,
                                  NSUInteger selected_threadgroup_memory_bytes,
                                  int32_t dispatch_count,
                                  double * elapsed_seconds) -> bool {
            id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState:selected_pipeline];
            [encoder setBytes:&selected_args length:sizeof(selected_args) atIndex:0];
            [encoder setBuffer:weights offset:0 atIndex:1];
            [encoder setBuffer:selected_activations offset:0 atIndex:2];
            [encoder setBuffer:output offset:0 atIndex:3];
            if (selected_threadgroup_memory_bytes > 0) {
                [encoder setThreadgroupMemoryLength:selected_threadgroup_memory_bytes
                                           atIndex:0];
            }
            for (int32_t dispatch = 0; dispatch < dispatch_count; ++dispatch) {
                [encoder dispatchThreadgroups:selected_threadgroups
                         threadsPerThreadgroup:selected_threads];
            }
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.status != MTLCommandBufferStatusCompleted) {
                print_error(@"execute benchmark", command_buffer.error);
                return false;
            }
            *elapsed_seconds = command_buffer.GPUEndTime - command_buffer.GPUStartTime;
            return *elapsed_seconds > 0.0;
        };

        for (int32_t warmup = 0; warmup < kWarmupSubmissions; ++warmup) {
            double warmup_seconds = 0.0;
            if (!run_dispatches(
                    pipeline,
                    args,
                    primary_activation_buffer,
                    threadgroups,
                    threads,
                    threadgroup_memory_bytes,
                    kWarmupDispatches,
                    &warmup_seconds)) {
                std::fprintf(stderr, "Metal GPU timestamps are unavailable\n");
                return EXIT_FAILURE;
            }
            if (compare &&
                !run_dispatches(
                    compare_pipeline,
                    compare_args,
                    compare_activation_buffer,
                    compare_threadgroups,
                    compare_threads,
                    compare_threadgroup_memory_bytes,
                    kWarmupDispatches,
                    &warmup_seconds)) {
                std::fprintf(stderr, "comparison GPU timestamps are unavailable\n");
                return EXIT_FAILURE;
            }
        }

        std::vector<double> milliseconds;
        std::vector<double> compare_milliseconds;
        milliseconds.reserve(sample_count);
        compare_milliseconds.reserve(sample_count);
        for (int32_t sample = 0; sample < sample_count; ++sample) {
            double elapsed_seconds = 0.0;
            double compare_elapsed_seconds = 0.0;
            const bool primary_first = sample % 2 == 0;
            for (int order = 0; order < (compare ? 2 : 1); ++order) {
                const bool run_primary =
                    !compare || (order == 0) == primary_first;
                if (run_primary) {
                    if (!run_dispatches(
                            pipeline,
                            args,
                            primary_activation_buffer,
                            threadgroups,
                            threads,
                            threadgroup_memory_bytes,
                            dispatches_per_sample,
                            &elapsed_seconds)) {
                        return EXIT_FAILURE;
                    }
                } else if (!run_dispatches(
                               compare_pipeline,
                               compare_args,
                               compare_activation_buffer,
                               compare_threadgroups,
                               compare_threads,
                               compare_threadgroup_memory_bytes,
                               dispatches_per_sample,
                               &compare_elapsed_seconds)) {
                    return EXIT_FAILURE;
                }
            }
            milliseconds.push_back(
                elapsed_seconds * 1000.0 / static_cast<double>(dispatches_per_sample));
            if (compare) {
                compare_milliseconds.push_back(
                    compare_elapsed_seconds * 1000.0 /
                    static_cast<double>(dispatches_per_sample));
            }
        }
        if (compare) {
            double correctness_seconds = 0.0;
            if (!run_dispatches(
                    pipeline,
                    args,
                    primary_activation_buffer,
                    threadgroups,
                    threads,
                    threadgroup_memory_bytes,
                    1,
                    &correctness_seconds)) {
                return EXIT_FAILURE;
            }
        }

        const double average_ms = mean(milliseconds);
        const double median_ms = median(milliseconds);
        const double stddev_ms = sample_stddev(milliseconds, average_ms);
        const double effective_gbps =
            static_cast<double>(weight_bytes) / (median_ms * 1.0e6);
        const double compare_average_ms =
            compare ? mean(compare_milliseconds) : 0.0;
        const double compare_median_ms =
            compare ? median(compare_milliseconds) : 0.0;
        const double compare_stddev_ms =
            compare
                ? sample_stddev(compare_milliseconds, compare_average_ms)
                : 0.0;
        const double compare_effective_gbps =
            compare
                ? static_cast<double>(weight_bytes) /
                    (compare_median_ms * 1.0e6)
                : 0.0;
        const auto * output_values = static_cast<const float *>(output.contents);
        const auto * weight_values = static_cast<const uint8_t *>(weights.contents);
        const auto * activation_values = static_cast<const float *>(activations.contents);
        double gpu_checksum = 0.0;
        double cpu_checksum = 0.0;
        double max_abs_error = 0.0;
        for (int32_t index = 0; index < ne01; index += 97) {
            double expected = 0.0;
            const uint8_t * row =
                weight_values + static_cast<size_t>(index) * weight_row_bytes;
            for (size_t block = 0; block < blocks_per_row; ++block) {
                const uint8_t * quant_block =
                    row + block * static_cast<size_t>(block_bytes);
                const uint8_t * quant = quant_block + 2;
                for (int32_t element = 0;
                     element < quant_block_elements;
                     ++element) {
                    if (is_q6_k) {
                        expected += static_cast<double>(
                            q6_k_value(quant_block, element)) *
                            activation_values[
                                block*quant_block_elements + element];
                        continue;
                    }
                    int32_t value = 0;
                    if (is_q8) {
                        value = static_cast<int8_t>(quant[element]);
                    } else {
                        const uint8_t packed =
                            quant[element % (quant_block_elements / 2)];
                        const int32_t nibble =
                            element < quant_block_elements / 2
                                ? packed & 0x0f
                                : packed >> 4;
                        value = nibble - 8;
                    }
                    expected += static_cast<double>(value) *
                        activation_values[
                            block * quant_block_elements + element];
                }
            }
            gpu_checksum += output_values[index];
            cpu_checksum += expected;
            max_abs_error =
                std::max(max_abs_error, std::abs(expected - output_values[index]));
        }

        std::printf(
            "{"
            "\"device\":%s,"
            "\"function\":%s,"
            "\"quantization\":%s,"
            "\"activation_type\":%s,"
            "\"nsg\":%d,"
            "\"ne00\":%d,"
            "\"ne01\":%d,"
            "\"weight_bytes\":%zu,"
            "\"samples\":%d,"
            "\"dispatches_per_sample\":%d,"
            "\"warmup_submissions\":%d,"
            "\"warmup_dispatches\":%d,"
            "\"mean_gpu_ms\":%.9f,"
            "\"median_gpu_ms\":%.9f,"
            "\"stddev_gpu_ms\":%.9f,"
            "\"effective_weight_gbps\":%.6f,"
            "\"gpu_checksum\":%.9f,"
            "\"cpu_checksum\":%.9f,"
            "\"max_abs_error\":%.9f,"
            "\"thread_execution_width\":%lu,"
            "\"max_threads_per_threadgroup\":%lu,"
            "\"static_threadgroup_memory_bytes\":%lu,"
            "\"sample_gpu_ms\":[",
            json_escape(device.name).UTF8String,
            json_escape(function_name).UTF8String,
            is_q8
                ? "\"q8_0\""
                : (is_q6_k ? "\"q6_K\"" : "\"q4_0\""),
            uses_q8_activation ? "\"q8_0\"" : "\"f32\"",
            nsg,
            ne00,
            ne01,
            weight_bytes,
            sample_count,
            dispatches_per_sample,
            kWarmupSubmissions,
            kWarmupDispatches,
            average_ms,
            median_ms,
            stddev_ms,
            effective_gbps,
            gpu_checksum,
            cpu_checksum,
            max_abs_error,
            static_cast<unsigned long>(pipeline.threadExecutionWidth),
            static_cast<unsigned long>(pipeline.maxTotalThreadsPerThreadgroup),
            static_cast<unsigned long>(pipeline.staticThreadgroupMemoryLength));
        for (size_t index = 0; index < milliseconds.size(); ++index) {
            std::printf("%s%.9f", index == 0 ? "" : ",", milliseconds[index]);
        }
        std::printf("],\"comparison\":");
        if (compare) {
            std::printf(
                "{"
                "\"function\":%s,"
                "\"activation_type\":%s,"
                "\"nsg\":%d,"
                "\"mean_gpu_ms\":%.9f,"
                "\"median_gpu_ms\":%.9f,"
                "\"stddev_gpu_ms\":%.9f,"
                "\"effective_weight_gbps\":%.6f,"
                "\"primary_vs_comparison_median_latency_pct\":%.6f,"
                "\"thread_execution_width\":%lu,"
                "\"max_threads_per_threadgroup\":%lu,"
                "\"static_threadgroup_memory_bytes\":%lu,"
                "\"sample_gpu_ms\":[",
                json_escape(compare_function_name).UTF8String,
                compare_uses_q8_activation ? "\"q8_0\"" : "\"f32\"",
                compare_nsg,
                compare_average_ms,
                compare_median_ms,
                compare_stddev_ms,
                compare_effective_gbps,
                (median_ms/compare_median_ms - 1.0)*100.0,
                static_cast<unsigned long>(
                    compare_pipeline.threadExecutionWidth),
                static_cast<unsigned long>(
                    compare_pipeline.maxTotalThreadsPerThreadgroup),
                static_cast<unsigned long>(
                    compare_pipeline.staticThreadgroupMemoryLength));
            for (size_t index = 0;
                 index < compare_milliseconds.size();
                 ++index) {
                std::printf(
                    "%s%.9f",
                    index == 0 ? "" : ",",
                    compare_milliseconds[index]);
            }
            std::printf("]}");
        } else {
            std::printf("null");
        }
        std::printf("}\n");
    }

    return EXIT_SUCCESS;
}
