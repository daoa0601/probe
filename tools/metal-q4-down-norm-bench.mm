#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

constexpr NSUInteger kFcMulMv = 600;
constexpr NSUInteger kFcBin = 1300;
constexpr int32_t kDefaultInput = 15360;
constexpr int32_t kDefaultOutput = 3840;
constexpr int32_t kNsg = 4;
constexpr int32_t kNr0 = 4;
constexpr int32_t kQk = 32;
constexpr int32_t kBlockBytes = 18;
constexpr float kEps = 1.0e-6f;

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

struct alignas(8) FusedArgs {
    MulMvArgs mv;
    int32_t n_groups;
    uint32_t epoch;
    float eps;
};

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

static_assert(sizeof(MulMvArgs) == 112);
static_assert(sizeof(FusedArgs) == 128);

double mean(const std::vector<double> & values) {
    return std::accumulate(values.begin(), values.end(), 0.0)/
        static_cast<double>(values.size());
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size()/2;
    return values.size()%2 == 0
        ? (values[middle - 1] + values[middle])/2.0
        : values[middle];
}

double sample_stddev(
        const std::vector<double> & values,
        double average) {
    if (values.size() < 2) {
        return 0.0;
    }
    double sum = 0.0;
    for (double value : values) {
        const double delta = value - average;
        sum += delta*delta;
    }
    return std::sqrt(sum/static_cast<double>(values.size() - 1));
}

void print_error(const char * context, NSError * error) {
    std::fprintf(
        stderr,
        "%s: %s\n",
        context,
        error.localizedDescription.UTF8String);
}

id<MTLComputePipelineState> make_pipeline(
        id<MTLDevice> device,
        id<MTLLibrary> library,
        NSString * name,
        MTLFunctionConstantValues * constants) {
    NSError * error = nil;
    id<MTLFunction> function = constants == nil
        ? [library newFunctionWithName:name]
        : [library newFunctionWithName:name
                        constantValues:constants
                                 error:&error];
    if (function == nil) {
        print_error("load function", error);
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
        print_error("create pipeline", error);
    }
    return pipeline;
}

}  // namespace

int main(int argc, const char * argv[]) {
    if (argc < 2 || argc > 10) {
        std::fprintf(
            stderr,
            "usage: %s METALLIB [SAMPLES] [ITERATIONS_PER_SAMPLE "
            "[FUSED_NR0 FUSED_NSG [ACCUMULATORS [INPUT_WIDTH "
            "[down|down_exact|down_exact_ab|attn|attn_ffn "
            "[OUTPUT_WIDTH]]]]]]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int samples = argc >= 3 ? std::atoi(argv[2]) : 9;
    const int iterations = argc >= 4 ? std::atoi(argv[3]) : 8;
    const int fused_nr0 = argc >= 5 ? std::atoi(argv[4]) : 4;
    const int fused_nsg = argc >= 6 ? std::atoi(argv[5]) : 4;
    const int accumulators = argc >= 7 ? std::atoi(argv[6]) : 1;
    const int32_t input =
        argc >= 8 ? std::atoi(argv[7]) : kDefaultInput;
    const int32_t output =
        argc >= 10 ? std::atoi(argv[9]) : kDefaultOutput;
    const bool attention_tail =
        argc >= 9 && std::strcmp(argv[8], "attn") == 0;
    const bool attention_ffn_tail =
        argc >= 9 && std::strcmp(argv[8], "attn_ffn") == 0;
    const bool down_exact_tail =
        argc >= 9 && std::strcmp(argv[8], "down_exact") == 0;
    const bool down_exact_ab =
        argc >= 9 && std::strcmp(argv[8], "down_exact_ab") == 0;
    const bool attention_family =
        attention_tail || attention_ffn_tail;
    const bool known_tail =
        argc < 9 ||
        std::strcmp(argv[8], "down") == 0 ||
        down_exact_tail ||
        down_exact_ab ||
        attention_family;
    const bool supported_shape =
        (fused_nr0 == 1 && fused_nsg == 8) ||
        (fused_nr0 == 2 &&
         fused_nsg >= 4 && fused_nsg <= 8) ||
        (fused_nr0 == 3 && fused_nsg == 8) ||
        (fused_nr0 == 4 &&
         (fused_nsg == 4 || fused_nsg == 8)) ||
        (fused_nr0 == 8 &&
         (fused_nsg == 2 || fused_nsg == 4));
    const bool supported_accumulators =
        accumulators == 1 ||
        (!attention_family &&
         fused_nr0 == 2 && fused_nsg == 8 &&
         (accumulators == 2 || accumulators == 4));
    if (samples < 3 ||
        iterations < 1 ||
        input <= 0 ||
        input%kQk != 0 ||
        output <= 0 ||
        output%4 != 0 ||
        !known_tail ||
        !supported_shape ||
        !supported_accumulators ||
        ((down_exact_tail || down_exact_ab) &&
         (fused_nr0 != 2 ||
          fused_nsg != 8 ||
          input != kDefaultInput ||
          output != kDefaultOutput))) {
        std::fprintf(
            stderr,
            "samples must be >= 3, iterations >= 1, and the fused "
            "(NR0, NSG) pair must be one of (1,8), (2,4..8), "
            "(3,8), (4,4), "
            "(4,8), (8,2), or (8,4); multiple accumulators are "
            "supported only for (2,8), with values 2 or 4; input "
            "width must be a positive multiple of 32, output width "
            "must be a positive multiple of 4; down_exact requires "
            "(2,8), input width 15360, and output width 3840\n");
        return EXIT_FAILURE;
    }
    const int fused_groups =
        (output + fused_nr0*fused_nsg - 1)/
        (fused_nr0*fused_nsg);
    const int baseline_groups =
        (output + kNr0*kNsg - 1)/(kNr0*kNsg);

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            std::fprintf(stderr, "no Metal device\n");
            return EXIT_FAILURE;
        }
        NSError * error = nil;
        NSURL * url = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:argv[1]]];
        id<MTLLibrary> library =
            [device newLibraryWithURL:url error:&error];
        if (library == nil) {
            print_error("load metallib", error);
            return EXIT_FAILURE;
        }

        const int16_t nsg = kNsg;
        const int16_t one = 1;
        MTLFunctionConstantValues * mv_constants =
            [[MTLFunctionConstantValues alloc] init];
        [mv_constants setConstantValue:&nsg
                                  type:MTLDataTypeShort
                               atIndex:kFcMulMv + 0];
        [mv_constants setConstantValue:&one
                                  type:MTLDataTypeShort
                               atIndex:kFcMulMv + 2];
        [mv_constants setConstantValue:&one
                                  type:MTLDataTypeShort
                               atIndex:kFcMulMv + 3];
        [mv_constants setConstantValue:&one
                                  type:MTLDataTypeShort
                               atIndex:kFcMulMv + 4];

        const int16_t mul_op = 2;
        const bool row_broadcast = true;
        const bool col_broadcast = true;
        MTLFunctionConstantValues * bin_constants =
            [[MTLFunctionConstantValues alloc] init];
        [bin_constants setConstantValue:&mul_op
                                   type:MTLDataTypeShort
                                atIndex:kFcBin + 0];
        [bin_constants setConstantValue:&one
                                   type:MTLDataTypeShort
                                atIndex:kFcBin + 1];
        [bin_constants setConstantValue:&row_broadcast
                                   type:MTLDataTypeBool
                                atIndex:kFcBin + 2];
        [bin_constants setConstantValue:&col_broadcast
                                   type:MTLDataTypeBool
                                atIndex:kFcBin + 3];

        id<MTLComputePipelineState> mv_pipeline = make_pipeline(
            device,
            library,
            @"kernel_mul_mv_q4_0_f32",
            mv_constants);
        id<MTLComputePipelineState> norm_pipeline = make_pipeline(
            device,
            library,
            @"kernel_rms_norm_mul_add_f32_4",
            nil);
        id<MTLComputePipelineState> ffn_norm_pipeline = make_pipeline(
            device,
            library,
            @"kernel_rms_norm_mul_f32_4",
            nil);
        id<MTLComputePipelineState> scale_pipeline = make_pipeline(
            device,
            library,
            @"kernel_bin_fuse_f32_f32_f32",
            bin_constants);
        NSString * fused_name =
            (down_exact_tail || down_exact_ab)
            ? @"kernel_mul_mv_q4_0_f32_down_norm_add_exact_r2_nsg8"
            : attention_ffn_tail
            ? [NSString stringWithFormat:
                @"kernel_mul_mv_q4_0_f32_attn_ffn_norm_r%d_nsg%d",
                fused_nr0,
                fused_nsg]
            : attention_tail
            ? [NSString stringWithFormat:
                @"kernel_mul_mv_q4_0_f32_attn_norm_add_r%d_nsg%d",
                fused_nr0,
                fused_nsg]
            : (accumulators == 1
                ? [NSString stringWithFormat:
                    @"kernel_mul_mv_q4_0_f32_down_norm_add_r%d_nsg%d",
                    fused_nr0,
                    fused_nsg]
                : [NSString stringWithFormat:
                    @"kernel_mul_mv_q4_0_f32_down_norm_add_r%d_nsg%d_a%d",
                    fused_nr0,
                    fused_nsg,
                    accumulators]);
        id<MTLComputePipelineState> fused_pipeline = make_pipeline(
            device,
            library,
            fused_name,
            nil);
        id<MTLComputePipelineState> generic_fused_pipeline =
            down_exact_ab
                ? make_pipeline(
                    device,
                    library,
                    @"kernel_mul_mv_q4_0_f32_down_norm_add_r2_nsg8",
                    nil)
                : nil;
        if (mv_pipeline == nil ||
            norm_pipeline == nil ||
            ffn_norm_pipeline == nil ||
            scale_pipeline == nil ||
            fused_pipeline == nil ||
            (down_exact_ab && generic_fused_pipeline == nil)) {
            return EXIT_FAILURE;
        }

        const size_t blocks_per_row = input/kQk;
        const size_t weight_row_bytes =
            blocks_per_row*kBlockBytes;
        const size_t weight_bytes =
            weight_row_bytes*output;
        const size_t input_bytes = input*sizeof(float);
        const size_t output_bytes = output*sizeof(float);
        const MTLResourceOptions shared = MTLResourceStorageModeShared;
        id<MTLBuffer> weights =
            [device newBufferWithLength:weight_bytes options:shared];
        id<MTLBuffer> activation =
            [device newBufferWithLength:input_bytes options:shared];
        id<MTLBuffer> raw =
            [device newBufferWithLength:output_bytes options:shared];
        id<MTLBuffer> state =
            [device newBufferWithLength:output_bytes options:shared];
        id<MTLBuffer> norm_weight =
            [device newBufferWithLength:output_bytes options:shared];
        id<MTLBuffer> residual =
            [device newBufferWithLength:output_bytes options:shared];
        id<MTLBuffer> ffn_norm_weight =
            [device newBufferWithLength:output_bytes options:shared];
        id<MTLBuffer> norm_output =
            [device newBufferWithLength:output_bytes options:shared];
        id<MTLBuffer> baseline_hidden =
            [device newBufferWithLength:output_bytes options:shared];
        id<MTLBuffer> fused_hidden =
            [device newBufferWithLength:output_bytes options:shared];
        id<MTLBuffer> baseline_output =
            [device newBufferWithLength:output_bytes options:shared];
        id<MTLBuffer> fused_output =
            [device newBufferWithLength:output_bytes options:shared];
        id<MTLBuffer> layer_scale =
            [device newBufferWithLength:sizeof(float) options:shared];
        if (weights == nil ||
            activation == nil ||
            raw == nil ||
            state == nil ||
            norm_weight == nil ||
            residual == nil ||
            ffn_norm_weight == nil ||
            norm_output == nil ||
            baseline_hidden == nil ||
            fused_hidden == nil ||
            baseline_output == nil ||
            fused_output == nil ||
            layer_scale == nil) {
            std::fprintf(stderr, "Metal buffer allocation failed\n");
            return EXIT_FAILURE;
        }

        auto * weight_bytes_ptr =
            static_cast<uint8_t *>(weights.contents);
        for (size_t block = 0;
             block < blocks_per_row*static_cast<size_t>(output);
             ++block) {
            uint8_t * current =
                weight_bytes_ptr + block*kBlockBytes;
            current[0] = 0x00;
            current[1] = 0x3c;
            for (int byte = 2; byte < kBlockBytes; ++byte) {
                current[byte] =
                    static_cast<uint8_t>(
                        (block*29 + byte*17)&0xff);
            }
        }
        auto * activation_values =
            static_cast<float *>(activation.contents);
        for (int32_t i = 0; i < input; ++i) {
            activation_values[i] =
                static_cast<float>((i%31) - 15)/16.0f;
        }
        auto * norm_values =
            static_cast<float *>(norm_weight.contents);
        auto * residual_values =
            static_cast<float *>(residual.contents);
        auto * ffn_norm_values =
            static_cast<float *>(ffn_norm_weight.contents);
        for (int32_t i = 0; i < output; ++i) {
            norm_values[i] = 0.75f +
                static_cast<float>(i%19)/64.0f;
            residual_values[i] =
                static_cast<float>((i%23) - 11)/32.0f;
            ffn_norm_values[i] = 0.8125f +
                static_cast<float>(i%17)/80.0f;
        }
        *static_cast<float *>(layer_scale.contents) = 0.875f;
        std::memset(raw.contents, 0, output_bytes);
        std::memset(state.contents, 0, output_bytes);
        std::memset(norm_output.contents, 0, output_bytes);
        std::memset(baseline_hidden.contents, 0, output_bytes);
        std::memset(fused_hidden.contents, 0, output_bytes);
        std::memset(baseline_output.contents, 0, output_bytes);
        std::memset(fused_output.contents, 0, output_bytes);

        MulMvArgs mv_args = {
            input,
            output,
            1,
            kBlockBytes,
            weight_row_bytes,
            weight_bytes,
            weight_bytes,
            input,
            1,
            1,
            sizeof(float),
            input_bytes,
            input_bytes,
            input_bytes,
            output,
            1,
            kNr0,
            1,
            1,
        };
        FusedArgs fused_args = {
            mv_args,
            fused_groups,
            1,
            kEps,
        };
        NormArgs norm_args = {
            output,
            output/4,
            output_bytes,
            output_bytes,
            output_bytes,
            kEps,
            { 1, 1, 1 },
            { 1, 1, 1 },
            { 1, 1, 1 },
            { output_bytes, output_bytes, output_bytes },
            { output_bytes, output_bytes, output_bytes },
            { output_bytes, output_bytes, output_bytes },
        };
        BinArgs bin_args = {
            output,
            1,
            1,
            1,
            sizeof(float),
            output_bytes,
            output_bytes,
            output_bytes,
            1,
            1,
            1,
            1,
            sizeof(float),
            sizeof(float),
            sizeof(float),
            sizeof(float),
            output,
            1,
            1,
            1,
            sizeof(float),
            output_bytes,
            output_bytes,
            output_bytes,
            0,
            { 0 },
        };

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            std::fprintf(stderr, "Metal command queue failed\n");
            return EXIT_FAILURE;
        }
        __block uint32_t next_epoch = 1;

        const auto encode_baseline = ^(
                id<MTLComputeCommandEncoder> encoder) {
            if (down_exact_ab) {
                FusedArgs current = fused_args;
                current.epoch = next_epoch++;
                [encoder setComputePipelineState:generic_fused_pipeline];
                [encoder setBytes:&current
                           length:sizeof(current)
                          atIndex:0];
                [encoder setBuffer:weights offset:0 atIndex:1];
                [encoder setBuffer:activation offset:0 atIndex:2];
                [encoder setBuffer:raw offset:0 atIndex:3];
                [encoder setBuffer:state offset:0 atIndex:4];
                [encoder setBuffer:norm_weight offset:0 atIndex:5];
                [encoder setBuffer:residual offset:0 atIndex:6];
                [encoder setBuffer:layer_scale offset:0 atIndex:7];
                [encoder setBuffer:baseline_output offset:0 atIndex:8];
                [encoder setThreadgroupMemoryLength:
                    fused_nsg*sizeof(float)
                                            atIndex:0];
                [encoder dispatchThreadgroups:
                    MTLSizeMake(fused_groups, 1, 1)
                         threadsPerThreadgroup:
                    MTLSizeMake(32, fused_nsg, 1)];
                return;
            }
            [encoder setComputePipelineState:mv_pipeline];
            [encoder setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
            [encoder setBuffer:weights offset:0 atIndex:1];
            [encoder setBuffer:activation offset:0 atIndex:2];
            [encoder setBuffer:raw offset:0 atIndex:3];
            [encoder dispatchThreadgroups:
                MTLSizeMake(baseline_groups, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(32, kNsg, 1)];

            [encoder setComputePipelineState:norm_pipeline];
            [encoder setBytes:&norm_args
                       length:sizeof(norm_args)
                      atIndex:0];
            [encoder setBuffer:raw offset:0 atIndex:1];
            [encoder setBuffer:norm_weight offset:0 atIndex:2];
            [encoder setBuffer:residual offset:0 atIndex:3];
            [encoder setBuffer:
                attention_tail
                    ? baseline_output
                    : (attention_ffn_tail
                        ? baseline_hidden
                        : norm_output)
                         offset:0
                        atIndex:4];
            [encoder setThreadgroupMemoryLength:32*sizeof(float)
                                        atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(output/4, 1, 1)];

            if (attention_ffn_tail) {
                [encoder setComputePipelineState:ffn_norm_pipeline];
                [encoder setBytes:&norm_args
                           length:sizeof(norm_args)
                          atIndex:0];
                [encoder setBuffer:baseline_hidden offset:0 atIndex:1];
                [encoder setBuffer:ffn_norm_weight offset:0 atIndex:2];
                [encoder setBuffer:residual offset:0 atIndex:3];
                [encoder setBuffer:baseline_output offset:0 atIndex:4];
                [encoder setThreadgroupMemoryLength:32*sizeof(float)
                                            atIndex:0];
                [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                         threadsPerThreadgroup:
                    MTLSizeMake(output/4, 1, 1)];
            } else if (!attention_tail) {
                [encoder setComputePipelineState:scale_pipeline];
                [encoder setBytes:&bin_args
                           length:sizeof(bin_args)
                          atIndex:0];
                [encoder setBuffer:norm_output offset:0 atIndex:1];
                [encoder setBuffer:layer_scale offset:0 atIndex:2];
                [encoder setBuffer:baseline_output offset:0 atIndex:3];
                [encoder dispatchThreadgroups:
                    MTLSizeMake(output, 1, 1)
                         threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            }
        };

        const auto encode_fused = ^(
                id<MTLComputeCommandEncoder> encoder,
                uint32_t epoch) {
            FusedArgs current = fused_args;
            current.epoch = epoch;
            [encoder setComputePipelineState:fused_pipeline];
            [encoder setBytes:&current
                       length:sizeof(current)
                      atIndex:0];
            [encoder setBuffer:weights offset:0 atIndex:1];
            [encoder setBuffer:activation offset:0 atIndex:2];
            [encoder setBuffer:raw offset:0 atIndex:3];
            [encoder setBuffer:state offset:0 atIndex:4];
            [encoder setBuffer:norm_weight offset:0 atIndex:5];
            [encoder setBuffer:residual offset:0 atIndex:6];
            [encoder setBuffer:
                attention_ffn_tail ? ffn_norm_weight : layer_scale
                         offset:0
                        atIndex:7];
            [encoder setBuffer:
                attention_ffn_tail ? fused_hidden : fused_output
                         offset:0
                        atIndex:8];
            if (attention_ffn_tail) {
                [encoder setBuffer:fused_output offset:0 atIndex:9];
            }
            [encoder setThreadgroupMemoryLength:
                fused_nsg*sizeof(float)
                                        atIndex:0];
            [encoder dispatchThreadgroups:
                MTLSizeMake(fused_groups, 1, 1)
                     threadsPerThreadgroup:
                MTLSizeMake(32, fused_nsg, 1)];
        };

        const auto run = ^bool(
                bool fused,
                int count,
                double * elapsed_seconds) {
            id<MTLCommandBuffer> command_buffer =
                [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder =
                [command_buffer computeCommandEncoder];
            for (int i = 0; i < count; ++i) {
                if (fused) {
                    encode_fused(encoder, next_epoch++);
                } else {
                    encode_baseline(encoder);
                }
            }
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.status !=
                MTLCommandBufferStatusCompleted) {
                print_error("execute", command_buffer.error);
                return false;
            }
            *elapsed_seconds =
                command_buffer.GPUEndTime -
                command_buffer.GPUStartTime;
            return *elapsed_seconds > 0.0;
        };

        double elapsed = 0.0;
        for (int warmup = 0; warmup < 4; ++warmup) {
            if (!run(false, 8, &elapsed) ||
                !run(true, 8, &elapsed)) {
                return EXIT_FAILURE;
            }
        }

        if (!run(false, 1, &elapsed)) {
            return EXIT_FAILURE;
        }
        std::vector<float> expected(output);
        std::vector<float> expected_raw(output);
        std::vector<float> expected_hidden(output);
        std::memcpy(
            expected.data(),
            baseline_output.contents,
            output_bytes);
        std::memcpy(
            expected_raw.data(),
            raw.contents,
            output_bytes);
        if (attention_ffn_tail) {
            std::memcpy(
                expected_hidden.data(),
                baseline_hidden.contents,
                output_bytes);
        }
        if (!run(true, 1, &elapsed)) {
            return EXIT_FAILURE;
        }
        const auto * actual =
            static_cast<const float *>(fused_output.contents);
        const auto * actual_raw =
            static_cast<const float *>(raw.contents);
        const auto * actual_hidden =
            static_cast<const float *>(fused_hidden.contents);
        double max_abs_error = 0.0;
        double rms_error = 0.0;
        double raw_max_abs_error = 0.0;
        double hidden_max_abs_error = 0.0;
        double hidden_rms_error = 0.0;
        for (int i = 0; i < output; ++i) {
            const double delta =
                static_cast<double>(actual[i]) - expected[i];
            max_abs_error =
                std::max(max_abs_error, std::abs(delta));
            rms_error += delta*delta;
            raw_max_abs_error = std::max(
                raw_max_abs_error,
                std::abs(
                    static_cast<double>(actual_raw[i]) -
                    expected_raw[i]));
            if (attention_ffn_tail) {
                const double hidden_delta =
                    static_cast<double>(actual_hidden[i]) -
                    expected_hidden[i];
                hidden_max_abs_error = std::max(
                    hidden_max_abs_error,
                    std::abs(hidden_delta));
                hidden_rms_error += hidden_delta*hidden_delta;
            }
        }
        rms_error = std::sqrt(rms_error/output);
        hidden_rms_error =
            std::sqrt(hidden_rms_error/output);

        std::vector<double> baseline_ms;
        std::vector<double> fused_ms;
        for (int sample = 0; sample < samples; ++sample) {
            const bool baseline_first = sample%2 == 0;
            for (int order = 0; order < 2; ++order) {
                const bool do_baseline =
                    (order == 0) == baseline_first;
                if (!run(!do_baseline, iterations, &elapsed)) {
                    return EXIT_FAILURE;
                }
                const double per_iteration_ms =
                    elapsed*1000.0/iterations;
                (do_baseline ? baseline_ms : fused_ms)
                    .push_back(per_iteration_ms);
            }
        }

        const double baseline_mean = mean(baseline_ms);
        const double baseline_median = median(baseline_ms);
        const double fused_mean = mean(fused_ms);
        const double fused_median = median(fused_ms);
        std::printf(
            "{"
            "\"device\":\"%s\","
            "\"tail\":\"%s\","
            "\"shape\":[%d,%d],"
            "\"groups\":%d,"
            "\"fused_nr0\":%d,"
            "\"fused_nsg\":%d,"
            "\"accumulators\":%d,"
            "\"baseline_dispatches\":%d,"
            "\"fused_dispatches\":1,"
            "\"samples\":%d,"
            "\"iterations_per_sample\":%d,"
            "\"baseline_mean_gpu_ms\":%.9f,"
            "\"baseline_median_gpu_ms\":%.9f,"
            "\"baseline_stddev_gpu_ms\":%.9f,"
            "\"fused_mean_gpu_ms\":%.9f,"
            "\"fused_median_gpu_ms\":%.9f,"
            "\"fused_stddev_gpu_ms\":%.9f,"
            "\"speedup\":%.6f,"
            "\"latency_reduction_pct\":%.6f,"
            "\"raw_max_abs_error\":%.9g,"
            "\"hidden_max_abs_error\":%.9g,"
            "\"hidden_rms_error\":%.9g,"
            "\"max_abs_error\":%.9g,"
            "\"rms_error\":%.9g,"
            "\"baseline_samples_ms\":[",
            device.name.UTF8String,
            down_exact_ab
                ? "down_exact_ab"
                : down_exact_tail
                ? "down_exact"
                : attention_ffn_tail
                ? "attn_ffn"
                : (attention_tail ? "attn" : "down"),
            input,
            output,
            fused_groups,
            fused_nr0,
            fused_nsg,
            accumulators,
            down_exact_ab
                ? 1
                : (attention_tail ? 2 : 3),
            samples,
            iterations,
            baseline_mean,
            baseline_median,
            sample_stddev(baseline_ms, baseline_mean),
            fused_mean,
            fused_median,
            sample_stddev(fused_ms, fused_mean),
            baseline_median/fused_median,
            (1.0 - fused_median/baseline_median)*100.0,
            raw_max_abs_error,
            hidden_max_abs_error,
            hidden_rms_error,
            max_abs_error,
            rms_error);
        for (size_t i = 0; i < baseline_ms.size(); ++i) {
            std::printf(
                "%s%.9f",
                i == 0 ? "" : ",",
                baseline_ms[i]);
        }
        std::printf("],\"fused_samples_ms\":[");
        for (size_t i = 0; i < fused_ms.size(); ++i) {
            std::printf(
                "%s%.9f",
                i == 0 ? "" : ",",
                fused_ms[i]);
        }
        std::printf("]}\n");
    }
    return EXIT_SUCCESS;
}
