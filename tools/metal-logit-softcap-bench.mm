#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace {

constexpr NSUInteger kFunctionConstantUnary = 1200;
constexpr NSUInteger kFunctionConstantBin = 1300;
constexpr int16_t kUnaryScale = 10;
constexpr int16_t kUnaryTanh = 100;
constexpr int32_t kElements = 262144;
constexpr int32_t kThreads = 256;
constexpr int32_t kWarmupSubmissions = 4;
constexpr int32_t kWarmupSequences = 32;

struct alignas(8) UnaryArgs {
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
    float slope;
    float scale;
    float bias;
    float val;
    float min;
    float max;
};

struct alignas(8) BinArgs {
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    int32_t ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne10;
    int32_t ne11;
    int32_t ne12;
    int32_t ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    int32_t ne2;
    int32_t ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    uint64_t offs;
    uint64_t o1[8];
};

struct LogitSoftcapArgs {
    int32_t ne0;
    float scale0;
    float bias0;
    float scale1;
    float bias1;
};

static_assert(sizeof(UnaryArgs) == 120);
static_assert(sizeof(BinArgs) == 216);
static_assert(sizeof(LogitSoftcapArgs) == 20);

void print_usage(const char * program) {
    std::fprintf(
        stderr,
        "usage: %s METALLIB [SAMPLES] [SEQUENCES_PER_SAMPLE]\n",
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

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 != 0
        ? values[middle]
        : (values[middle - 1] + values[middle]) / 2.0;
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

id<MTLComputePipelineState> make_unary_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    int16_t op) {
    const bool contiguous = false;
    MTLFunctionConstantValues * constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&op
                           type:MTLDataTypeShort
                        atIndex:kFunctionConstantUnary + 0];
    [constants setConstantValue:&contiguous
                           type:MTLDataTypeBool
                        atIndex:kFunctionConstantUnary + 1];

    NSError * error = nil;
    id<MTLFunction> function =
        [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                      constantValues:constants
                               error:&error];
    if (function == nil) {
        print_error(@"specialize unary function", error);
        return nil;
    }
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
        print_error(@"compile unary pipeline", error);
    }
    return pipeline;
}

id<MTLComputePipelineState> make_add_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library) {
    const int16_t op = 0;
    const int16_t fuse_count = 1;
    const bool row_broadcast = false;
    const bool column_broadcast = false;
    MTLFunctionConstantValues * constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&op
                           type:MTLDataTypeShort
                        atIndex:kFunctionConstantBin + 0];
    [constants setConstantValue:&fuse_count
                           type:MTLDataTypeShort
                        atIndex:kFunctionConstantBin + 1];
    [constants setConstantValue:&row_broadcast
                           type:MTLDataTypeBool
                        atIndex:kFunctionConstantBin + 2];
    [constants setConstantValue:&column_broadcast
                           type:MTLDataTypeBool
                        atIndex:kFunctionConstantBin + 3];

    NSError * error = nil;
    id<MTLFunction> function =
        [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32_4"
                      constantValues:constants
                               error:&error];
    if (function == nil) {
        print_error(@"specialize add function", error);
        return nil;
    }
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
        print_error(@"compile add pipeline", error);
    }
    return pipeline;
}

id<MTLCounterSet> timestamp_counter_set(id<MTLDevice> device) {
    for (id<MTLCounterSet> counter_set in device.counterSets) {
        if ([counter_set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
            return counter_set;
        }
    }
    return nil;
}

}  // namespace

int main(int argc, const char * argv[]) {
    if (argc < 2 || argc > 4) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    int32_t sample_count = 31;
    int32_t sequences_per_sample = 64;
    if ((argc >= 3 && !parse_positive_int(argv[2], "samples", &sample_count)) ||
        (argc >= 4 &&
         !parse_positive_int(argv[3], "sequences_per_sample", &sequences_per_sample))) {
        return EXIT_FAILURE;
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            std::fprintf(stderr, "no Metal device is available\n");
            return EXIT_FAILURE;
        }
        if (![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
            std::fprintf(stderr, "stage-boundary counter sampling is unavailable\n");
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

        id<MTLComputePipelineState> scale =
            make_unary_pipeline(device, library, kUnaryScale);
        id<MTLComputePipelineState> tanh =
            make_unary_pipeline(device, library, kUnaryTanh);
        id<MTLComputePipelineState> add = make_add_pipeline(device, library);
        id<MTLFunction> fused_function =
            [library newFunctionWithName:@"kernel_logit_softcap_add_f32"];
        id<MTLComputePipelineState> fused =
            [device newComputePipelineStateWithFunction:fused_function error:&error];
        if (scale == nil || tanh == nil || add == nil || fused == nil) {
            if (fused == nil) {
                print_error(@"compile fused pipeline", error);
            }
            return EXIT_FAILURE;
        }

        const size_t bytes = static_cast<size_t>(kElements) * sizeof(float);
        id<MTLBuffer> source =
            [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> residual =
            [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> temporary0 =
            [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> temporary1 =
            [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> temporary2 =
            [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> baseline_output =
            [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> fused_output =
            [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (source == nil || residual == nil || temporary0 == nil || temporary1 == nil ||
            temporary2 == nil || baseline_output == nil || fused_output == nil) {
            std::fprintf(stderr, "unable to allocate benchmark buffers\n");
            return EXIT_FAILURE;
        }

        auto * source_values = static_cast<float *>(source.contents);
        auto * residual_values = static_cast<float *>(residual.contents);
        for (int32_t element = 0; element < kElements; ++element) {
            source_values[element] =
                static_cast<float>((element * 17) % 2047 - 1023) / 64.0f;
            residual_values[element] =
                static_cast<float>((element * 29) % 1021 - 510) / 128.0f;
        }

        const int32_t elements4 = kElements/4;
        const uint64_t row_bytes = bytes;
        UnaryArgs unary_args = {
            elements4, 1, 1, 1,
            sizeof(float), row_bytes, row_bytes, row_bytes,
            elements4, 1, 1, 1,
            sizeof(float), row_bytes, row_bytes, row_bytes,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        };
        BinArgs add_args = {
            elements4, 1, 1, 1,
            sizeof(float), row_bytes, row_bytes, row_bytes,
            elements4, 1, 1, 1,
            sizeof(float), row_bytes, row_bytes, row_bytes,
            elements4, 1, 1, 1,
            sizeof(float), row_bytes, row_bytes, row_bytes,
            0, {0},
        };
        LogitSoftcapArgs fused_args = {
            kElements,
            1.0f/30.0f,
            0.0f,
            30.0f,
            0.0f,
        };

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCounterSet> timestamp_set = timestamp_counter_set(device);
        MTLCounterSampleBufferDescriptor * counter_descriptor =
            [[MTLCounterSampleBufferDescriptor alloc] init];
        counter_descriptor.counterSet = timestamp_set;
        counter_descriptor.label = @"Probe logit-softcap timestamps";
        counter_descriptor.storageMode = MTLStorageModeShared;
        counter_descriptor.sampleCount = 2;
        id<MTLCounterSampleBuffer> counter_buffer =
            [device newCounterSampleBufferWithDescriptor:counter_descriptor error:&error];
        if (queue == nil || timestamp_set == nil || counter_buffer == nil) {
            print_error(@"create timestamp counter buffer", error);
            return EXIT_FAILURE;
        }

        auto encode_unary = ^(
            id<MTLComputeCommandEncoder> encoder,
            id<MTLComputePipelineState> pipeline,
            id<MTLBuffer> input,
            id<MTLBuffer> output,
            float scale_value,
            float bias_value) {
            UnaryArgs args = unary_args;
            args.scale = scale_value;
            args.bias = bias_value;
            [encoder setComputePipelineState:pipeline];
            [encoder setBytes:&args length:sizeof(args) atIndex:0];
            [encoder setBuffer:input offset:0 atIndex:1];
            [encoder setBuffer:output offset:0 atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(elements4/kThreads, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
        };

        auto run = [&](bool use_fused, int32_t sequence_count, double * milliseconds) -> bool {
            MTLComputePassDescriptor * pass = [MTLComputePassDescriptor computePassDescriptor];
            MTLComputePassSampleBufferAttachmentDescriptor * attachment =
                pass.sampleBufferAttachments[0];
            attachment.sampleBuffer = counter_buffer;
            attachment.startOfEncoderSampleIndex = 0;
            attachment.endOfEncoderSampleIndex = 1;

            id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder =
                [command_buffer computeCommandEncoderWithDescriptor:pass];
            for (int32_t sequence = 0; sequence < sequence_count; ++sequence) {
                if (use_fused) {
                    [encoder setComputePipelineState:fused];
                    [encoder setBytes:&fused_args length:sizeof(fused_args) atIndex:0];
                    [encoder setBuffer:source offset:0 atIndex:1];
                    [encoder setBuffer:residual offset:0 atIndex:2];
                    [encoder setBuffer:fused_output offset:0 atIndex:3];
                    [encoder dispatchThreadgroups:MTLSizeMake(elements4/kThreads, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
                } else {
                    encode_unary(
                        encoder, scale, source, temporary0, fused_args.scale0, fused_args.bias0);
                    encode_unary(encoder, tanh, temporary0, temporary1, 0.0f, 0.0f);
                    encode_unary(
                        encoder, scale, temporary1, temporary2, fused_args.scale1, fused_args.bias1);

                    [encoder setComputePipelineState:add];
                    [encoder setBytes:&add_args length:sizeof(add_args) atIndex:0];
                    [encoder setBuffer:temporary2 offset:0 atIndex:1];
                    [encoder setBuffer:residual offset:0 atIndex:2];
                    [encoder setBuffer:baseline_output offset:0 atIndex:3];
                    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
                }
            }
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.status != MTLCommandBufferStatusCompleted) {
                print_error(@"execute benchmark", command_buffer.error);
                return false;
            }

            NSData * data = [counter_buffer resolveCounterRange:NSMakeRange(0, 2)];
            if (data.length != 2 * sizeof(MTLCounterResultTimestamp)) {
                std::fprintf(stderr, "unable to resolve two timestamp samples\n");
                return false;
            }
            const auto * timestamps =
                static_cast<const MTLCounterResultTimestamp *>(data.bytes);
            if (timestamps[0].timestamp == 0 ||
                timestamps[1].timestamp == 0 ||
                timestamps[0].timestamp == MTLCounterErrorValue ||
                timestamps[1].timestamp == MTLCounterErrorValue ||
                timestamps[1].timestamp <= timestamps[0].timestamp) {
                std::fprintf(stderr, "invalid stage timestamp sample\n");
                return false;
            }
            *milliseconds =
                static_cast<double>(timestamps[1].timestamp - timestamps[0].timestamp) /
                (1.0e6 * static_cast<double>(sequence_count));
            return true;
        };

        for (int32_t warmup = 0; warmup < kWarmupSubmissions; ++warmup) {
            double ignored = 0.0;
            if (!run(false, kWarmupSequences, &ignored) ||
                !run(true, kWarmupSequences, &ignored)) {
                return EXIT_FAILURE;
            }
        }

        std::vector<double> baseline_ms;
        std::vector<double> fused_ms;
        baseline_ms.reserve(sample_count);
        fused_ms.reserve(sample_count);
        for (int32_t sample = 0; sample < sample_count; ++sample) {
            double baseline = 0.0;
            double candidate = 0.0;
            const bool fused_first = sample % 2 != 0;
            if ((fused_first &&
                 (!run(true, sequences_per_sample, &candidate) ||
                  !run(false, sequences_per_sample, &baseline))) ||
                (!fused_first &&
                 (!run(false, sequences_per_sample, &baseline) ||
                  !run(true, sequences_per_sample, &candidate)))) {
                return EXIT_FAILURE;
            }
            baseline_ms.push_back(baseline);
            fused_ms.push_back(candidate);
        }

        double max_baseline_abs_error = 0.0;
        double max_fused_abs_error = 0.0;
        double max_between_abs_error = 0.0;
        double baseline_checksum = 0.0;
        double fused_checksum = 0.0;
        const auto * baseline_values =
            static_cast<const float *>(baseline_output.contents);
        const auto * fused_values =
            static_cast<const float *>(fused_output.contents);
        for (int32_t element = 0; element < kElements; ++element) {
            const float expected =
                std::tanh(source_values[element] * fused_args.scale0 + fused_args.bias0) *
                    fused_args.scale1 +
                fused_args.bias1 +
                residual_values[element];
            max_baseline_abs_error = std::max(
                max_baseline_abs_error,
                std::abs(static_cast<double>(baseline_values[element] - expected)));
            max_fused_abs_error = std::max(
                max_fused_abs_error,
                std::abs(static_cast<double>(fused_values[element] - expected)));
            max_between_abs_error = std::max(
                max_between_abs_error,
                std::abs(static_cast<double>(
                    fused_values[element] - baseline_values[element])));
            baseline_checksum += baseline_values[element];
            fused_checksum += fused_values[element];
        }

        const double baseline_mean = mean(baseline_ms);
        const double fused_mean = mean(fused_ms);
        const double baseline_median = median(baseline_ms);
        const double fused_median = median(fused_ms);
        std::printf(
            "{"
            "\"device\":%s,"
            "\"elements\":%d,"
            "\"samples\":%d,"
            "\"sequences_per_sample\":%d,"
            "\"timing\":\"stage_boundary_gpu_timestamp\","
            "\"baseline_dispatches_per_sequence\":4,"
            "\"fused_dispatches_per_sequence\":1,"
            "\"baseline_mean_ms\":%.9f,"
            "\"baseline_median_ms\":%.9f,"
            "\"baseline_stddev_ms\":%.9f,"
            "\"fused_mean_ms\":%.9f,"
            "\"fused_median_ms\":%.9f,"
            "\"fused_stddev_ms\":%.9f,"
            "\"speedup\":%.6f,"
            "\"saved_intermediate_bytes\":%zu,"
            "\"baseline_checksum\":%.9f,"
            "\"fused_checksum\":%.9f,"
            "\"max_baseline_abs_error\":%.9f,"
            "\"max_fused_abs_error\":%.9f,"
            "\"max_between_abs_error\":%.9f,"
            "\"sample_baseline_ms\":[",
            json_escape(device.name).UTF8String,
            kElements,
            sample_count,
            sequences_per_sample,
            baseline_mean,
            baseline_median,
            sample_stddev(baseline_ms, baseline_mean),
            fused_mean,
            fused_median,
            sample_stddev(fused_ms, fused_mean),
            baseline_median / fused_median,
            3 * bytes,
            baseline_checksum,
            fused_checksum,
            max_baseline_abs_error,
            max_fused_abs_error,
            max_between_abs_error);
        for (size_t index = 0; index < baseline_ms.size(); ++index) {
            std::printf("%s%.9f", index == 0 ? "" : ",", baseline_ms[index]);
        }
        std::printf("],\"sample_fused_ms\":[");
        for (size_t index = 0; index < fused_ms.size(); ++index) {
            std::printf("%s%.9f", index == 0 ? "" : ",", fused_ms[index]);
        }
        std::printf("]}\n");
    }

    return EXIT_SUCCESS;
}
