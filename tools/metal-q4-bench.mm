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
constexpr int32_t kQ4BlockElements = 32;
constexpr int32_t kQ4BlockBytes = 18;
constexpr int32_t kRowsPerSimdgroup = 4;
constexpr int32_t kWarmupSubmissions = 4;
constexpr int32_t kWarmupDispatches = 64;

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
        "usage: %s METALLIB FUNCTION NSG NE00 NE01 [SAMPLES] [DISPATCHES_PER_SAMPLE]\n",
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

void initialize_q4_weights(id<MTLBuffer> buffer, size_t block_count) {
    auto * bytes = static_cast<uint8_t *>(buffer.contents);
    for (size_t block = 0; block < block_count; ++block) {
        uint8_t * current = bytes + block * kQ4BlockBytes;
        current[0] = 0x00;
        current[1] = 0x3c;  // IEEE-754 binary16 1.0
        for (int byte = 2; byte < kQ4BlockBytes; ++byte) {
            current[byte] = static_cast<uint8_t>((block * 29 + byte * 17) & 0xff);
        }
    }
}

void initialize_activations(id<MTLBuffer> buffer, int32_t element_count) {
    auto * values = static_cast<float *>(buffer.contents);
    for (int32_t index = 0; index < element_count; ++index) {
        values[index] = static_cast<float>((index % 31) - 15) / 16.0f;
    }
}

}  // namespace

int main(int argc, const char * argv[]) {
    if (argc < 6 || argc > 8) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    int32_t nsg = 0;
    int32_t ne00 = 0;
    int32_t ne01 = 0;
    int32_t sample_count = 7;
    int32_t dispatches_per_sample = 5;
    if (!parse_positive_int(argv[3], "nsg", &nsg) ||
        !parse_positive_int(argv[4], "ne00", &ne00) ||
        !parse_positive_int(argv[5], "ne01", &ne01) ||
        (argc >= 7 && !parse_positive_int(argv[6], "samples", &sample_count)) ||
        (argc >= 8 &&
         !parse_positive_int(argv[7], "dispatches_per_sample", &dispatches_per_sample))) {
        return EXIT_FAILURE;
    }
    if (nsg > INT16_MAX || ne00 % kQ4BlockElements != 0) {
        std::fprintf(stderr, "nsg must fit int16 and ne00 must be divisible by 32\n");
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

        const int16_t nsg_i16 = static_cast<int16_t>(nsg);
        const int16_t one = 1;
        MTLFunctionConstantValues * constants = [[MTLFunctionConstantValues alloc] init];
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

        NSString * function_name = [NSString stringWithUTF8String:argv[2]];
        id<MTLFunction> function = [library newFunctionWithName:function_name
                                                constantValues:constants
                                                         error:&error];
        if (function == nil) {
            print_error(@"specialize function", error);
            return EXIT_FAILURE;
        }

        MTLComputePipelineDescriptor * descriptor = [[MTLComputePipelineDescriptor alloc] init];
        descriptor.label = function_name;
        descriptor.computeFunction = function;
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;

        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithDescriptor:descriptor
                                                  options:MTLPipelineOptionNone
                                               reflection:nil
                                                    error:&error];
        if (pipeline == nil) {
            print_error(@"compile pipeline", error);
            return EXIT_FAILURE;
        }

        const size_t blocks_per_row = static_cast<size_t>(ne00 / kQ4BlockElements);
        const size_t block_count = blocks_per_row * static_cast<size_t>(ne01);
        const size_t weight_bytes = block_count * kQ4BlockBytes;
        const size_t activation_bytes = static_cast<size_t>(ne00) * sizeof(float);
        const size_t output_bytes = static_cast<size_t>(ne01) * sizeof(float);

        id<MTLBuffer> weights =
            [device newBufferWithLength:weight_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> activations =
            [device newBufferWithLength:activation_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> output =
            [device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
        if (weights == nil || activations == nil || output == nil) {
            std::fprintf(stderr, "unable to allocate benchmark buffers\n");
            return EXIT_FAILURE;
        }
        initialize_q4_weights(weights, block_count);
        initialize_activations(activations, ne00);
        std::memset(output.contents, 0, output_bytes);

        const uint64_t weight_row_bytes = blocks_per_row * kQ4BlockBytes;
        MulMvArgs args = {
            ne00,
            ne01,
            1,
            kQ4BlockBytes,
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
            kRowsPerSimdgroup,
            1,
            1,
        };

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            std::fprintf(stderr, "unable to create Metal command queue\n");
            return EXIT_FAILURE;
        }

        const NSUInteger output_rows_per_threadgroup =
            static_cast<NSUInteger>(kRowsPerSimdgroup * nsg);
        const MTLSize threadgroups = MTLSizeMake(
            (static_cast<NSUInteger>(ne01) + output_rows_per_threadgroup - 1) /
                output_rows_per_threadgroup,
            1,
            1);
        const MTLSize threads = MTLSizeMake(32, static_cast<NSUInteger>(nsg), 1);

        auto run_dispatches = [&](int32_t dispatch_count, double * elapsed_seconds) -> bool {
            id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBytes:&args length:sizeof(args) atIndex:0];
            [encoder setBuffer:weights offset:0 atIndex:1];
            [encoder setBuffer:activations offset:0 atIndex:2];
            [encoder setBuffer:output offset:0 atIndex:3];
            for (int32_t dispatch = 0; dispatch < dispatch_count; ++dispatch) {
                [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads];
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
            if (!run_dispatches(kWarmupDispatches, &warmup_seconds)) {
                std::fprintf(stderr, "Metal GPU timestamps are unavailable\n");
                return EXIT_FAILURE;
            }
        }

        std::vector<double> milliseconds;
        milliseconds.reserve(sample_count);
        for (int32_t sample = 0; sample < sample_count; ++sample) {
            double elapsed_seconds = 0.0;
            if (!run_dispatches(dispatches_per_sample, &elapsed_seconds)) {
                return EXIT_FAILURE;
            }
            milliseconds.push_back(
                elapsed_seconds * 1000.0 / static_cast<double>(dispatches_per_sample));
        }

        const double average_ms = mean(milliseconds);
        const double median_ms = median(milliseconds);
        const double stddev_ms = sample_stddev(milliseconds, average_ms);
        const double effective_gbps =
            static_cast<double>(weight_bytes) / (median_ms * 1.0e6);
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
                const uint8_t * quant = row + block * kQ4BlockBytes + 2;
                for (int32_t element = 0; element < kQ4BlockElements; ++element) {
                    const uint8_t packed =
                        quant[element % (kQ4BlockElements / 2)];
                    const int32_t nibble =
                        element < kQ4BlockElements / 2 ? packed & 0x0f : packed >> 4;
                    expected += static_cast<double>(nibble - 8) *
                        activation_values[block * kQ4BlockElements + element];
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
        std::printf("]}\n");
    }

    return EXIT_SUCCESS;
}
