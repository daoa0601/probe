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
#include <vector>

namespace {

constexpr NSUInteger kFunctionConstantRope = 800;
constexpr int32_t kWarmupSubmissions = 4;
constexpr int32_t kWarmupDispatches = 64;

struct alignas(8) NormArgs {
    int32_t ne00;
    int32_t ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float eps;
    int32_t nef1[3];
    int32_t nef2[3];
    int32_t nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
};

struct alignas(8) RopeArgs {
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    int32_t ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne0;
    int32_t ne1;
    int32_t ne2;
    int32_t ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t n_past;
    int32_t n_dims;
    int32_t n_ctx_orig;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    int32_t sect_0;
    int32_t sect_1;
    int32_t sect_2;
    int32_t sect_3;
    bool src2;
};

void print_usage(const char * program) {
    std::fprintf(
        stderr,
        "usage: %s METALLIB HEADS HEAD_ELEMENTS "
        "[SAMPLES [DISPATCHES_PER_SAMPLE]]\n",
        program);
}

bool parse_positive_int(const char * text, const char * name, int32_t * value) {
    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed <= 0 || parsed > INT32_MAX) {
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
    NSData * data =
        [NSJSONSerialization dataWithJSONObject:@[value] options:0 error:nil];
    NSString * array =
        [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    return [array substringWithRange:NSMakeRange(1, array.length - 2)];
}

double mean(const std::vector<double> & values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() % 2 != 0) {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) / 2.0;
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

void print_samples(const std::vector<double> & samples) {
    for (size_t index = 0; index < samples.size(); ++index) {
        if (index != 0) {
            std::printf(",");
        }
        std::printf("%.9f", samples[index]);
    }
}

}  // namespace

int main(int argc, const char * argv[]) {
    if (argc < 4 || argc > 6) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    int32_t head_count = 0;
    int32_t head_elements = 0;
    int32_t sample_count = 11;
    int32_t dispatches_per_sample = 64;
    if (!parse_positive_int(argv[2], "heads", &head_count) ||
        (argc >= 5 &&
         !parse_positive_int(argv[4], "samples", &sample_count)) ||
        !parse_positive_int(argv[3], "head_elements", &head_elements) ||
        (argc >= 6 &&
         !parse_positive_int(
             argv[5],
             "dispatches_per_sample",
             &dispatches_per_sample))) {
        return EXIT_FAILURE;
    }
    if ((head_elements != 256 && head_elements != 512) ||
        head_elements % 4 != 0) {
        std::fprintf(
            stderr,
            "head_elements must be 256 or 512 and divisible by 4\n");
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
        id<MTLLibrary> library =
            [device newLibraryWithURL:library_url error:&error];
        if (library == nil) {
            print_error(@"load metallib", error);
            return EXIT_FAILURE;
        }

        const auto make_pipeline =
            [&](NSString * name, bool rope_constants)
                -> id<MTLComputePipelineState> {
            id<MTLFunction> function = nil;
            if (rope_constants) {
                const bool false_value = false;
                MTLFunctionConstantValues * constants =
                    [[MTLFunctionConstantValues alloc] init];
                [constants setConstantValue:&false_value
                                       type:MTLDataTypeBool
                                    atIndex:kFunctionConstantRope + 0];
                [constants setConstantValue:&false_value
                                       type:MTLDataTypeBool
                                    atIndex:kFunctionConstantRope + 1];
                function =
                    [library newFunctionWithName:name
                                  constantValues:constants
                                           error:&error];
            } else {
                function = [library newFunctionWithName:name];
            }
            if (function == nil) {
                print_error(@"specialize function", error);
                return nil;
            }

            MTLComputePipelineDescriptor * descriptor =
                [[MTLComputePipelineDescriptor alloc] init];
            descriptor.label = name;
            descriptor.computeFunction = function;
            descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
            id<MTLComputePipelineState> pipeline =
                [device newComputePipelineStateWithDescriptor:descriptor
                                                      options:MTLPipelineOptionNone
                                                   reflection:nil
                                                        error:&error];
            if (pipeline == nil) {
                print_error(@"compile pipeline", error);
            }
            return pipeline;
        };

        id<MTLComputePipelineState> norm_pipeline =
            make_pipeline(@"kernel_rms_norm_mul_f32_4", false);
        id<MTLComputePipelineState> rope_pipeline =
            make_pipeline(@"kernel_rope_neox_f32", true);
        id<MTLComputePipelineState> fused_pipeline =
            make_pipeline(@"kernel_rms_norm_mul_rope_neox_f32", true);
        if (norm_pipeline == nil ||
            rope_pipeline == nil ||
            fused_pipeline == nil) {
            return EXIT_FAILURE;
        }

        const size_t head_bytes =
            static_cast<size_t>(head_elements)*sizeof(float);
        const size_t tensor_bytes =
            head_bytes*static_cast<size_t>(head_count);

        id<MTLBuffer> source =
            [device newBufferWithLength:tensor_bytes
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> scale =
            [device newBufferWithLength:head_bytes
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> positions =
            [device newBufferWithLength:sizeof(int32_t)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> intermediate =
            [device newBufferWithLength:tensor_bytes
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> baseline_output =
            [device newBufferWithLength:tensor_bytes
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> fused_output =
            [device newBufferWithLength:tensor_bytes
                               options:MTLResourceStorageModeShared];
        if (source == nil ||
            scale == nil ||
            positions == nil ||
            intermediate == nil ||
            baseline_output == nil ||
            fused_output == nil) {
            std::fprintf(stderr, "unable to allocate benchmark buffers\n");
            return EXIT_FAILURE;
        }

        auto * source_values = static_cast<float *>(source.contents);
        auto * scale_values = static_cast<float *>(scale.contents);
        for (int32_t head = 0; head < head_count; ++head) {
            for (int32_t index = 0; index < head_elements; ++index) {
                source_values[
                    static_cast<size_t>(head)*head_elements + index] =
                    static_cast<float>(
                        ((head*17 + index*13) % 257) - 128) /
                    64.0f;
            }
        }
        for (int32_t index = 0; index < head_elements; ++index) {
            scale_values[index] =
                0.75f + static_cast<float>(index % 31)/64.0f;
        }
        *static_cast<int32_t *>(positions.contents) = 173;
        std::memset(intermediate.contents, 0, tensor_bytes);
        std::memset(baseline_output.contents, 0, tensor_bytes);
        std::memset(fused_output.contents, 0, tensor_bytes);

        const uint64_t head_stride = head_bytes;
        const uint64_t token_stride =
            head_bytes*static_cast<uint64_t>(head_count);
        NormArgs norm_args = {
            head_elements,
            head_elements/4,
            head_stride,
            token_stride,
            token_stride,
            1.0e-6f,
            {head_count, 1, 1},
            {1, 1, 1},
            {1, 1, 1},
            {head_stride, head_stride, head_stride},
            {token_stride, head_stride, head_stride},
            {token_stride, head_stride, head_stride},
        };
        RopeArgs rope_args = {
            head_elements,
            head_count,
            1,
            1,
            sizeof(float),
            head_stride,
            token_stride,
            token_stride,
            head_elements,
            head_count,
            1,
            1,
            sizeof(float),
            head_stride,
            token_stride,
            token_stride,
            0,
            head_elements,
            8192,
            1000000.0f,
            1.0f,
            0.0f,
            1.0f,
            32.0f,
            1.0f,
            0,
            0,
            0,
            0,
            false,
        };

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            std::fprintf(stderr, "unable to create Metal command queue\n");
            return EXIT_FAILURE;
        }

        const MTLSize threadgroups =
            MTLSizeMake(static_cast<NSUInteger>(head_count), 1, 1);
        const MTLSize norm_threads =
            MTLSizeMake(static_cast<NSUInteger>(head_elements/4), 1, 1);
        const MTLSize rope_threads =
            MTLSizeMake(static_cast<NSUInteger>(head_elements), 1, 1);
        const MTLSize fused_threads = norm_threads;
        constexpr NSUInteger kSharedBytes = 32*sizeof(float);

        const auto run =
            [&](bool fused, int32_t dispatch_count, double * elapsed_seconds)
                -> bool {
            id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder =
                [command_buffer computeCommandEncoder];
            for (int32_t dispatch = 0;
                 dispatch < dispatch_count;
                 ++dispatch) {
                if (fused) {
                    [encoder setComputePipelineState:fused_pipeline];
                    [encoder setBytes:&norm_args
                               length:sizeof(norm_args)
                              atIndex:0];
                    [encoder setBytes:&rope_args
                               length:sizeof(rope_args)
                              atIndex:1];
                    [encoder setBuffer:source offset:0 atIndex:2];
                    [encoder setBuffer:scale offset:0 atIndex:3];
                    [encoder setBuffer:positions offset:0 atIndex:4];
                    [encoder setBuffer:source offset:0 atIndex:5];
                    [encoder setBuffer:fused_output offset:0 atIndex:6];
                    [encoder setThreadgroupMemoryLength:kSharedBytes
                                               atIndex:0];
                    [encoder dispatchThreadgroups:threadgroups
                             threadsPerThreadgroup:fused_threads];
                } else {
                    [encoder setComputePipelineState:norm_pipeline];
                    [encoder setBytes:&norm_args
                               length:sizeof(norm_args)
                              atIndex:0];
                    [encoder setBuffer:source offset:0 atIndex:1];
                    [encoder setBuffer:scale offset:0 atIndex:2];
                    [encoder setBuffer:source offset:0 atIndex:3];
                    [encoder setBuffer:intermediate offset:0 atIndex:4];
                    [encoder setThreadgroupMemoryLength:kSharedBytes
                                               atIndex:0];
                    [encoder dispatchThreadgroups:threadgroups
                             threadsPerThreadgroup:norm_threads];

                    [encoder setComputePipelineState:rope_pipeline];
                    [encoder setBytes:&rope_args
                               length:sizeof(rope_args)
                              atIndex:0];
                    [encoder setBuffer:intermediate offset:0 atIndex:1];
                    [encoder setBuffer:positions offset:0 atIndex:2];
                    [encoder setBuffer:source offset:0 atIndex:3];
                    [encoder setBuffer:baseline_output offset:0 atIndex:4];
                    [encoder dispatchThreadgroups:threadgroups
                             threadsPerThreadgroup:rope_threads];
                }
            }
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.status != MTLCommandBufferStatusCompleted) {
                print_error(@"execute benchmark", command_buffer.error);
                return false;
            }
            *elapsed_seconds =
                command_buffer.GPUEndTime - command_buffer.GPUStartTime;
            return *elapsed_seconds > 0.0;
        };

        for (int32_t warmup = 0;
             warmup < kWarmupSubmissions;
             ++warmup) {
            double elapsed_seconds = 0.0;
            if (!run(false, kWarmupDispatches, &elapsed_seconds) ||
                !run(true, kWarmupDispatches, &elapsed_seconds)) {
                std::fprintf(stderr, "Metal GPU timestamps are unavailable\n");
                return EXIT_FAILURE;
            }
        }

        std::vector<double> baseline_ms;
        std::vector<double> fused_ms;
        baseline_ms.reserve(sample_count);
        fused_ms.reserve(sample_count);
        for (int32_t sample = 0; sample < sample_count; ++sample) {
            double baseline_seconds = 0.0;
            double fused_seconds = 0.0;
            const bool baseline_first = sample % 2 == 0;
            for (int order = 0; order < 2; ++order) {
                const bool run_baseline =
                    (order == 0) == baseline_first;
                if (!run(
                        !run_baseline,
                        dispatches_per_sample,
                        run_baseline
                            ? &baseline_seconds
                            : &fused_seconds)) {
                    return EXIT_FAILURE;
                }
            }
            baseline_ms.push_back(
                baseline_seconds*1000.0/dispatches_per_sample);
            fused_ms.push_back(
                fused_seconds*1000.0/dispatches_per_sample);
        }

        double correctness_seconds = 0.0;
        if (!run(false, 1, &correctness_seconds) ||
            !run(true, 1, &correctness_seconds)) {
            return EXIT_FAILURE;
        }

        const auto * baseline_values =
            static_cast<const float *>(baseline_output.contents);
        const auto * fused_values =
            static_cast<const float *>(fused_output.contents);
        const size_t element_count =
            static_cast<size_t>(head_count)*head_elements;
        double max_abs_error = 0.0;
        size_t exact_elements = 0;
        for (size_t index = 0; index < element_count; ++index) {
            const double difference =
                std::abs(
                    static_cast<double>(baseline_values[index]) -
                    static_cast<double>(fused_values[index]));
            max_abs_error = std::max(max_abs_error, difference);
            if (std::memcmp(
                    baseline_values + index,
                    fused_values + index,
                    sizeof(float)) == 0) {
                ++exact_elements;
            }
        }

        const double baseline_mean = mean(baseline_ms);
        const double fused_mean = mean(fused_ms);
        const double baseline_median = median(baseline_ms);
        const double fused_median = median(fused_ms);
        const double baseline_stddev =
            sample_stddev(baseline_ms, baseline_mean);
        const double fused_stddev =
            sample_stddev(fused_ms, fused_mean);
        const double latency_change =
            (fused_median/baseline_median - 1.0)*100.0;

        std::printf(
            "{"
            "\"device\":%s,"
            "\"heads\":%d,"
            "\"head_elements\":%d,"
            "\"samples\":%d,"
            "\"dispatches_per_sample\":%d,"
            "\"warmup_submissions\":%d,"
            "\"warmup_dispatches\":%d,"
            "\"timing\":\"command_buffer_gpu_timestamp\","
            "\"baseline_dispatches_per_sequence\":2,"
            "\"fused_dispatches_per_sequence\":1,"
            "\"baseline_mean_gpu_ms\":%.9f,"
            "\"baseline_median_gpu_ms\":%.9f,"
            "\"baseline_stddev_gpu_ms\":%.9f,"
            "\"fused_mean_gpu_ms\":%.9f,"
            "\"fused_median_gpu_ms\":%.9f,"
            "\"fused_stddev_gpu_ms\":%.9f,"
            "\"fused_latency_change_pct\":%.6f,"
            "\"max_abs_error\":%.9g,"
            "\"exact_elements\":%zu,"
            "\"element_count\":%zu,"
            "\"baseline_samples_gpu_ms\":[",
            json_escape(device.name).UTF8String,
            head_count,
            head_elements,
            sample_count,
            dispatches_per_sample,
            kWarmupSubmissions,
            kWarmupDispatches,
            baseline_mean,
            baseline_median,
            baseline_stddev,
            fused_mean,
            fused_median,
            fused_stddev,
            latency_change,
            max_abs_error,
            exact_elements,
            element_count);
        print_samples(baseline_ms);
        std::printf("],\"fused_samples_gpu_ms\":[");
        print_samples(fused_ms);
        std::printf("]}\n");
    }

    return EXIT_SUCCESS;
}
