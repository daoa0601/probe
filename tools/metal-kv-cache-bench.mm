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
constexpr NSUInteger kSharedBytes = 32*sizeof(float);

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

struct alignas(8) SetRowsArgs {
    int32_t nk0;
    int32_t ne01;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne11;
    int32_t ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
};

enum class Sequence {
    KBaseline,
    KFused,
    VBaseline,
    VFused,
};

bool parse_positive_int(const char * text, const char * name, int32_t * value) {
    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed <= 0 || parsed > INT32_MAX) {
        std::fprintf(stderr, "invalid %s: %s\n", name, text);
        return false;
    }
    *value = static_cast<int32_t>(parsed);
    return true;
}

double mean(const std::vector<double> & values) {
    return std::accumulate(values.begin(), values.end(), 0.0)/values.size();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size()/2;
    return values.size()%2
        ? values[middle]
        : (values[middle - 1] + values[middle])/2.0;
}

double stddev(const std::vector<double> & values) {
    const double average = mean(values);
    double sum = 0.0;
    for (const double value : values) {
        const double delta = value - average;
        sum += delta*delta;
    }
    return values.size() < 2 ? 0.0 : std::sqrt(sum/(values.size() - 1));
}

void print_samples(const std::vector<double> & values) {
    for (size_t index = 0; index < values.size(); ++index) {
        std::printf("%s%.9f", index == 0 ? "" : ",", values[index]);
    }
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

}  // namespace

int main(int argc, const char * argv[]) {
    if (argc < 4 || argc > 6) {
        std::fprintf(
            stderr,
            "usage: %s METALLIB HEADS HEAD_ELEMENTS "
            "[SAMPLES [SEQUENCES_PER_SAMPLE]]\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    int32_t heads = 0;
    int32_t head_elements = 0;
    int32_t samples = 11;
    int32_t sequences_per_sample = 64;
    if (!parse_positive_int(argv[2], "heads", &heads) ||
        !parse_positive_int(argv[3], "head elements", &head_elements) ||
        (argc >= 5 &&
         !parse_positive_int(argv[4], "samples", &samples)) ||
        (argc >= 6 &&
         !parse_positive_int(
             argv[5],
             "sequences per sample",
             &sequences_per_sample)) ||
        (head_elements != 256 && head_elements != 512)) {
        return EXIT_FAILURE;
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError * error = nil;
        id<MTLLibrary> library = [
            device
            newLibraryWithURL:[
                NSURL fileURLWithPath:[
                    NSString stringWithUTF8String:argv[1]]]
            error:&error];
        if (library == nil) {
            print_error(@"load metallib", error);
            return EXIT_FAILURE;
        }

        const auto make_pipeline =
            [&](NSString * name, bool rope) -> id<MTLComputePipelineState> {
            id<MTLFunction> function = nil;
            if (rope) {
                const bool false_value = false;
                MTLFunctionConstantValues * constants =
                    [[MTLFunctionConstantValues alloc] init];
                [constants setConstantValue:&false_value
                                       type:MTLDataTypeBool
                                    atIndex:kFunctionConstantRope + 0];
                [constants setConstantValue:&false_value
                                       type:MTLDataTypeBool
                                    atIndex:kFunctionConstantRope + 1];
                function = [
                    library
                    newFunctionWithName:name
                    constantValues:constants
                    error:&error];
            } else {
                function = [library newFunctionWithName:name];
            }
            if (function == nil) {
                print_error(name, error);
                return nil;
            }
            MTLComputePipelineDescriptor * descriptor =
                [[MTLComputePipelineDescriptor alloc] init];
            descriptor.computeFunction = function;
            descriptor.label = name;
            descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
            id<MTLComputePipelineState> pipeline = [
                device
                newComputePipelineStateWithDescriptor:descriptor
                options:MTLPipelineOptionNone
                reflection:nil
                error:&error];
            if (pipeline == nil) {
                print_error(name, error);
            }
            return pipeline;
        };

        id<MTLComputePipelineState> norm_mul =
            make_pipeline(@"kernel_rms_norm_mul_f32_4", false);
        id<MTLComputePipelineState> norm =
            make_pipeline(@"kernel_rms_norm_f32_4", false);
        id<MTLComputePipelineState> rope =
            make_pipeline(@"kernel_rope_neox_f32", true);
        id<MTLComputePipelineState> set_rows =
            make_pipeline(@"kernel_set_rows_f32_i64_f16", false);
        id<MTLComputePipelineState> k_fused =
            make_pipeline(
                @"kernel_rms_norm_mul_rope_neox_set_rows_f16_i64",
                true);
        id<MTLComputePipelineState> v_fused =
            make_pipeline(@"kernel_rms_norm_set_rows_f16_i64", false);
        if (norm_mul == nil || norm == nil || rope == nil ||
            set_rows == nil || k_fused == nil || v_fused == nil) {
            return EXIT_FAILURE;
        }

        const size_t head_bytes =
            static_cast<size_t>(head_elements)*sizeof(float);
        const size_t tensor_bytes = head_bytes*heads;
        const size_t cache_row_bytes =
            static_cast<size_t>(head_elements)*heads*sizeof(uint16_t);
        constexpr int32_t kCacheRows = 4;
        constexpr int64_t kCacheRow = 2;

        const auto make_buffer = [&](size_t bytes) {
            return [
                device
                newBufferWithLength:bytes
                options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> k_source = make_buffer(tensor_bytes);
        id<MTLBuffer> v_source = make_buffer(tensor_bytes);
        id<MTLBuffer> scale = make_buffer(head_bytes);
        id<MTLBuffer> positions = make_buffer(sizeof(int32_t));
        id<MTLBuffer> indices = make_buffer(sizeof(int64_t));
        id<MTLBuffer> norm_output = make_buffer(tensor_bytes);
        id<MTLBuffer> rope_output = make_buffer(tensor_bytes);
        id<MTLBuffer> k_baseline_cache =
            make_buffer(cache_row_bytes*kCacheRows);
        id<MTLBuffer> k_fused_cache =
            make_buffer(cache_row_bytes*kCacheRows);
        id<MTLBuffer> v_baseline_cache =
            make_buffer(cache_row_bytes*kCacheRows);
        id<MTLBuffer> v_fused_cache =
            make_buffer(cache_row_bytes*kCacheRows);
        if (k_source == nil || v_source == nil || scale == nil ||
            positions == nil || indices == nil || norm_output == nil ||
            rope_output == nil || k_baseline_cache == nil ||
            k_fused_cache == nil || v_baseline_cache == nil ||
            v_fused_cache == nil) {
            std::fprintf(stderr, "unable to allocate Metal buffers\n");
            return EXIT_FAILURE;
        }

        auto * k_values = static_cast<float *>(k_source.contents);
        auto * v_values = static_cast<float *>(v_source.contents);
        for (size_t index = 0; index < tensor_bytes/sizeof(float); ++index) {
            k_values[index] =
                static_cast<float>((index*13 + 11)%257 - 128)/64.0f;
            v_values[index] =
                static_cast<float>((index*17 + 7)%263 - 131)/64.0f;
        }
        auto * scale_values = static_cast<float *>(scale.contents);
        for (int32_t index = 0; index < head_elements; ++index) {
            scale_values[index] =
                0.75f + static_cast<float>(index%31)/64.0f;
        }
        *static_cast<int32_t *>(positions.contents) = 173;
        *static_cast<int64_t *>(indices.contents) = kCacheRow;
        std::memset(norm_output.contents, 0, tensor_bytes);
        std::memset(rope_output.contents, 0, tensor_bytes);
        std::memset(k_baseline_cache.contents, 0, cache_row_bytes*kCacheRows);
        std::memset(k_fused_cache.contents, 0, cache_row_bytes*kCacheRows);
        std::memset(v_baseline_cache.contents, 0, cache_row_bytes*kCacheRows);
        std::memset(v_fused_cache.contents, 0, cache_row_bytes*kCacheRows);

        const uint64_t head_stride = head_bytes;
        const uint64_t token_stride = tensor_bytes;
        NormArgs norm_args = {
            head_elements,
            head_elements/4,
            head_stride,
            token_stride,
            token_stride,
            1.0e-6f,
            {heads, 1, 1},
            {1, 1, 1},
            {1, 1, 1},
            {head_stride, head_stride, head_stride},
            {token_stride, head_stride, head_stride},
            {token_stride, head_stride, head_stride},
        };
        RopeArgs rope_args = {
            head_elements,
            heads,
            1,
            1,
            sizeof(float),
            head_stride,
            token_stride,
            token_stride,
            head_elements,
            heads,
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
        SetRowsArgs set_args = {
            head_elements*heads,
            1,
            tensor_bytes,
            tensor_bytes,
            tensor_bytes,
            1,
            1,
            sizeof(int64_t),
            sizeof(int64_t),
            sizeof(int64_t),
            cache_row_bytes,
            cache_row_bytes*kCacheRows,
            cache_row_bytes*kCacheRows,
        };

        const MTLSize head_groups =
            MTLSizeMake(static_cast<NSUInteger>(heads), 1, 1);
        const MTLSize head_threads = MTLSizeMake(
            static_cast<NSUInteger>(head_elements/4),
            1,
            1);
        const MTLSize rope_threads =
            MTLSizeMake(static_cast<NSUInteger>(head_elements), 1, 1);
        const NSUInteger set_thread_count = std::min(
            static_cast<NSUInteger>(1024),
            static_cast<NSUInteger>(head_elements*heads));
        const MTLSize set_groups = MTLSizeMake(1, 1, 1);
        const MTLSize set_threads =
            MTLSizeMake(set_thread_count, 1, 1);

        id<MTLCommandQueue> queue = [device newCommandQueue];
        const auto encode_norm =
            [&](id<MTLComputeCommandEncoder> encoder,
                id<MTLComputePipelineState> pipeline,
                id<MTLBuffer> source) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBytes:&norm_args length:sizeof(norm_args) atIndex:0];
            [encoder setBuffer:source offset:0 atIndex:1];
            [encoder setBuffer:scale offset:0 atIndex:2];
            [encoder setBuffer:scale offset:0 atIndex:3];
            [encoder setBuffer:norm_output offset:0 atIndex:4];
            [encoder setThreadgroupMemoryLength:kSharedBytes atIndex:0];
            [encoder dispatchThreadgroups:head_groups
                     threadsPerThreadgroup:head_threads];
        };
        const auto encode_set_rows =
            [&](id<MTLComputeCommandEncoder> encoder,
                id<MTLBuffer> source,
                id<MTLBuffer> cache) {
            [encoder setComputePipelineState:set_rows];
            [encoder setBytes:&set_args length:sizeof(set_args) atIndex:0];
            [encoder setBuffer:source offset:0 atIndex:1];
            [encoder setBuffer:indices offset:0 atIndex:2];
            [encoder setBuffer:cache offset:0 atIndex:3];
            [encoder dispatchThreadgroups:set_groups
                     threadsPerThreadgroup:set_threads];
        };
        const auto run = [&](Sequence sequence,
                             int32_t count,
                             double * seconds) -> bool {
            id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder =
                [command_buffer computeCommandEncoder];
            for (int32_t iteration = 0; iteration < count; ++iteration) {
                if (sequence == Sequence::KBaseline) {
                    encode_norm(encoder, norm_mul, k_source);
                    [encoder setComputePipelineState:rope];
                    [encoder
                        setBytes:&rope_args
                        length:sizeof(rope_args)
                        atIndex:0];
                    [encoder setBuffer:norm_output offset:0 atIndex:1];
                    [encoder setBuffer:positions offset:0 atIndex:2];
                    [encoder setBuffer:k_source offset:0 atIndex:3];
                    [encoder setBuffer:rope_output offset:0 atIndex:4];
                    [encoder dispatchThreadgroups:head_groups
                             threadsPerThreadgroup:rope_threads];
                    encode_set_rows(
                        encoder,
                        rope_output,
                        k_baseline_cache);
                } else if (sequence == Sequence::KFused) {
                    [encoder setComputePipelineState:k_fused];
                    [encoder
                        setBytes:&norm_args
                        length:sizeof(norm_args)
                        atIndex:0];
                    [encoder
                        setBytes:&rope_args
                        length:sizeof(rope_args)
                        atIndex:1];
                    [encoder
                        setBytes:&set_args
                        length:sizeof(set_args)
                        atIndex:2];
                    [encoder setBuffer:k_source offset:0 atIndex:3];
                    [encoder setBuffer:scale offset:0 atIndex:4];
                    [encoder setBuffer:positions offset:0 atIndex:5];
                    [encoder setBuffer:k_source offset:0 atIndex:6];
                    [encoder setBuffer:indices offset:0 atIndex:7];
                    [encoder setBuffer:k_fused_cache offset:0 atIndex:8];
                    [encoder setThreadgroupMemoryLength:kSharedBytes
                                               atIndex:0];
                    [encoder dispatchThreadgroups:head_groups
                             threadsPerThreadgroup:head_threads];
                } else if (sequence == Sequence::VBaseline) {
                    encode_norm(encoder, norm, v_source);
                    encode_set_rows(
                        encoder,
                        norm_output,
                        v_baseline_cache);
                } else {
                    [encoder setComputePipelineState:v_fused];
                    [encoder
                        setBytes:&norm_args
                        length:sizeof(norm_args)
                        atIndex:0];
                    [encoder
                        setBytes:&set_args
                        length:sizeof(set_args)
                        atIndex:1];
                    [encoder setBuffer:v_source offset:0 atIndex:2];
                    [encoder setBuffer:indices offset:0 atIndex:3];
                    [encoder setBuffer:v_fused_cache offset:0 atIndex:4];
                    [encoder setThreadgroupMemoryLength:kSharedBytes
                                               atIndex:0];
                    [encoder dispatchThreadgroups:head_groups
                             threadsPerThreadgroup:head_threads];
                }
            }
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.status != MTLCommandBufferStatusCompleted) {
                print_error(@"execute benchmark", command_buffer.error);
                return false;
            }
            *seconds =
                command_buffer.GPUEndTime - command_buffer.GPUStartTime;
            return *seconds > 0.0;
        };

        for (int warmup = 0; warmup < 4; ++warmup) {
            for (const Sequence sequence : {
                     Sequence::KBaseline,
                     Sequence::KFused,
                     Sequence::VBaseline,
                     Sequence::VFused}) {
                double seconds = 0.0;
                if (!run(sequence, 64, &seconds)) {
                    return EXIT_FAILURE;
                }
            }
        }

        std::vector<double> k_base;
        std::vector<double> k_fuse;
        std::vector<double> v_base;
        std::vector<double> v_fuse;
        for (int32_t sample = 0; sample < samples; ++sample) {
            for (const auto pair : {
                     std::pair{Sequence::KBaseline, Sequence::KFused},
                     std::pair{Sequence::VBaseline, Sequence::VFused}}) {
                const bool baseline_first = sample%2 == 0;
                double baseline_seconds = 0.0;
                double fused_seconds = 0.0;
                for (int order = 0; order < 2; ++order) {
                    const bool run_baseline =
                        (order == 0) == baseline_first;
                    if (!run(
                            run_baseline ? pair.first : pair.second,
                            sequences_per_sample,
                            run_baseline
                                ? &baseline_seconds
                                : &fused_seconds)) {
                        return EXIT_FAILURE;
                    }
                }
                const double divisor =
                    1000.0/sequences_per_sample;
                auto & baseline =
                    pair.first == Sequence::KBaseline ? k_base : v_base;
                auto & fused =
                    pair.first == Sequence::KBaseline ? k_fuse : v_fuse;
                baseline.push_back(baseline_seconds*divisor);
                fused.push_back(fused_seconds*divisor);
            }
        }

        double seconds = 0.0;
        for (const Sequence sequence : {
                 Sequence::KBaseline,
                 Sequence::KFused,
                 Sequence::VBaseline,
                 Sequence::VFused}) {
            if (!run(sequence, 1, &seconds)) {
                return EXIT_FAILURE;
            }
        }

        const size_t cache_offset = kCacheRow*cache_row_bytes;
        const auto compare_cache =
            [&](id<MTLBuffer> baseline,
                id<MTLBuffer> fused,
                size_t * exact,
                double * max_abs) {
            const auto * lhs = reinterpret_cast<const uint16_t *>(
                static_cast<const uint8_t *>(baseline.contents) +
                cache_offset);
            const auto * rhs = reinterpret_cast<const uint16_t *>(
                static_cast<const uint8_t *>(fused.contents) +
                cache_offset);
            const size_t elements = cache_row_bytes/sizeof(uint16_t);
            *exact = 0;
            *max_abs = 0.0;
            for (size_t index = 0; index < elements; ++index) {
                *exact += lhs[index] == rhs[index];
                __fp16 lhs_half;
                __fp16 rhs_half;
                std::memcpy(&lhs_half, lhs + index, sizeof(lhs_half));
                std::memcpy(&rhs_half, rhs + index, sizeof(rhs_half));
                *max_abs = std::max(
                    *max_abs,
                    std::abs(
                        static_cast<double>(lhs_half) -
                        static_cast<double>(rhs_half)));
            }
        };
        size_t k_exact = 0;
        size_t v_exact = 0;
        double k_max_abs = 0.0;
        double v_max_abs = 0.0;
        compare_cache(
            k_baseline_cache,
            k_fused_cache,
            &k_exact,
            &k_max_abs);
        compare_cache(
            v_baseline_cache,
            v_fused_cache,
            &v_exact,
            &v_max_abs);
        const size_t cache_elements = cache_row_bytes/sizeof(uint16_t);

        const auto print_metric =
            [&](const char * prefix,
                const std::vector<double> & baseline,
                const std::vector<double> & fused) {
            const double baseline_median = median(baseline);
            const double fused_median = median(fused);
            std::printf(
                "\"%s_baseline_mean_gpu_ms\":%.9f,"
                "\"%s_baseline_median_gpu_ms\":%.9f,"
                "\"%s_baseline_stddev_gpu_ms\":%.9f,"
                "\"%s_fused_mean_gpu_ms\":%.9f,"
                "\"%s_fused_median_gpu_ms\":%.9f,"
                "\"%s_fused_stddev_gpu_ms\":%.9f,"
                "\"%s_latency_change_pct\":%.6f,"
                "\"%s_baseline_samples_gpu_ms\":[",
                prefix,
                mean(baseline),
                prefix,
                baseline_median,
                prefix,
                stddev(baseline),
                prefix,
                mean(fused),
                prefix,
                fused_median,
                prefix,
                stddev(fused),
                prefix,
                (fused_median/baseline_median - 1.0)*100.0,
                prefix);
            print_samples(baseline);
            std::printf("],\"%s_fused_samples_gpu_ms\":[", prefix);
            print_samples(fused);
            std::printf("]");
        };

        std::printf(
            "{"
            "\"device\":%s,"
            "\"heads\":%d,"
            "\"head_elements\":%d,"
            "\"samples\":%d,"
            "\"sequences_per_sample\":%d,"
            "\"timing\":\"command_buffer_gpu_timestamp\","
            "\"k_baseline_dispatches\":3,"
            "\"k_fused_dispatches\":1,"
            "\"v_baseline_dispatches\":2,"
            "\"v_fused_dispatches\":1,"
            "\"k_exact_elements\":%zu,"
            "\"v_exact_elements\":%zu,"
            "\"cache_elements\":%zu,"
            "\"k_max_abs_error\":%.9g,"
            "\"v_max_abs_error\":%.9g,",
            json_escape(device.name).UTF8String,
            heads,
            head_elements,
            samples,
            sequences_per_sample,
            k_exact,
            v_exact,
            cache_elements,
            k_max_abs,
            v_max_abs);
        print_metric("k", k_base, k_fuse);
        std::printf(",");
        print_metric("v", v_base, v_fuse);
        std::printf("}\n");
    }

    return EXIT_SUCCESS;
}
