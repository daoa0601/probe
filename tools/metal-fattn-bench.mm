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

constexpr NSUInteger kFunctionConstantFlash = 400;
constexpr NSUInteger kFunctionConstantReduce = 500;
constexpr int32_t kQueryHeads = 16;
constexpr int32_t kCacheChunk = 32;

struct alignas(8) FlashArgs {
    int32_t ne01;
    int32_t ne02;
    int32_t ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne11;
    int32_t ne_12_2;
    int32_t ne_12_3;
    int32_t ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t ne31;
    int32_t ne32;
    int32_t ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t ne1;
    int32_t ne2;
    int32_t ne3;
    float scale;
    float max_bias;
    float m0;
    float m1;
    int32_t n_head_log2;
    float logit_softcap;
};

struct ReduceArgs {
    int32_t nrows;
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

int32_t simdgroups_for(int32_t workgroups, int32_t cache_tokens) {
    int32_t simdgroups = 1;
    while (2*workgroups*simdgroups*kCacheChunk < cache_tokens &&
           simdgroups < 4) {
        simdgroups *= 2;
    }
    return simdgroups;
}

size_t shared_bytes(int32_t head_elements, int32_t simdgroups) {
    const int32_t padded = (head_elements + 127)/128*128;
    const size_t half_elements =
        static_cast<size_t>(
            padded + 4*kCacheChunk + 2*padded)*simdgroups;
    return (half_elements*sizeof(float)/2 + 15)/16*16;
}

}  // namespace

int main(int argc, const char * argv[]) {
    if (argc < 7 || argc > 12) {
        std::fprintf(
            stderr,
            "usage: %s METALLIB HEAD_ELEMENTS KV_HEADS QUERY_TOKENS "
            "ACTIVE_TOKENS CACHE_TOKENS "
            "[SAMPLES [SEQUENCES_PER_SAMPLE [CANDIDATE_NWG "
            "[CACHE_SETS [CANDIDATE_NSG]]]]]\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    int32_t head_elements = 0;
    int32_t kv_heads = 0;
    int32_t query_tokens = 0;
    int32_t active_tokens = 0;
    int32_t cache_tokens = 0;
    int32_t samples = 11;
    int32_t sequences_per_sample = 64;
    int32_t candidate_nwg = 1;
    int32_t cache_sets = 1;
    int32_t candidate_nsg = 0;
    if (!parse_positive_int(argv[2], "head elements", &head_elements) ||
        !parse_positive_int(argv[3], "KV heads", &kv_heads) ||
        !parse_positive_int(argv[4], "query tokens", &query_tokens) ||
        !parse_positive_int(argv[5], "active tokens", &active_tokens) ||
        !parse_positive_int(argv[6], "cache tokens", &cache_tokens) ||
        (argc >= 8 && !parse_positive_int(argv[7], "samples", &samples)) ||
        (argc >= 9 &&
         !parse_positive_int(
             argv[8],
             "sequences per sample",
             &sequences_per_sample)) ||
        (argc >= 10 &&
         !parse_positive_int(
             argv[9],
             "candidate workgroups",
             &candidate_nwg)) ||
        (argc >= 11 &&
         !parse_positive_int(
             argv[10],
             "cache sets",
             &cache_sets)) ||
        (argc >= 12 &&
         !parse_positive_int(
             argv[11],
             "candidate SIMD groups",
             &candidate_nsg)) ||
        (head_elements != 256 && head_elements != 512) ||
        (kv_heads != 1 && kv_heads != 8) ||
        kQueryHeads%kv_heads != 0 ||
        active_tokens > cache_tokens ||
        cache_tokens%kCacheChunk != 0 ||
        (candidate_nwg != 1 &&
         candidate_nwg != 2 &&
         candidate_nwg != 4 &&
         candidate_nwg != 8 &&
         candidate_nwg != 16) ||
        cache_sets > 32) {
        return EXIT_FAILURE;
    }
    if (candidate_nsg == 0) {
        candidate_nsg =
            simdgroups_for(candidate_nwg, cache_tokens);
    }
    if ((candidate_nsg != 1 &&
         candidate_nsg != 2 &&
         candidate_nsg != 4 &&
         candidate_nsg != 8) ||
        candidate_nsg*32 > 1024) {
        return EXIT_FAILURE;
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            std::fprintf(stderr, "no Metal device is available\n");
            return EXIT_FAILURE;
        }
        if (shared_bytes(head_elements, candidate_nsg) >
            device.maxThreadgroupMemoryLength) {
            std::fprintf(
                stderr,
                "candidate needs %zu threadgroup bytes; device exposes %zu\n",
                shared_bytes(head_elements, candidate_nsg),
                device.maxThreadgroupMemoryLength);
            return EXIT_FAILURE;
        }

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

        const int32_t cache_stride = head_elements*kv_heads;
        const int32_t rows = kQueryHeads*query_tokens;

        const auto make_main_pipeline =
            [&](int32_t workgroups,
                int32_t simdgroups) -> id<MTLComputePipelineState> {
            const bool true_value = true;
            const bool false_value = false;
            MTLFunctionConstantValues * constants =
                [[MTLFunctionConstantValues alloc] init];
            [constants setConstantValue:&true_value
                                   type:MTLDataTypeBool
                                atIndex:kFunctionConstantFlash + 0];
            for (NSUInteger index = 1; index <= 4; ++index) {
                [constants setConstantValue:&false_value
                                       type:MTLDataTypeBool
                                    atIndex:kFunctionConstantFlash + index];
            }
            [constants setConstantValue:&cache_stride
                                   type:MTLDataTypeInt
                                atIndex:kFunctionConstantFlash + 20];
            [constants setConstantValue:&cache_stride
                                   type:MTLDataTypeInt
                                atIndex:kFunctionConstantFlash + 21];
            [constants setConstantValue:&simdgroups
                                   type:MTLDataTypeInt
                                atIndex:kFunctionConstantFlash + 22];
            [constants setConstantValue:&workgroups
                                   type:MTLDataTypeInt
                                atIndex:kFunctionConstantFlash + 23];

            NSString * name = [
                NSString
                stringWithFormat:@"kernel_flash_attn_ext_vec_f16_dk%d_dv%d",
                                 head_elements,
                                 head_elements];
            id<MTLFunction> function = [
                library
                newFunctionWithName:name
                constantValues:constants
                error:&error];
            if (function == nil) {
                print_error(name, error);
                return nil;
            }
            MTLComputePipelineDescriptor * descriptor =
                [[MTLComputePipelineDescriptor alloc] init];
            descriptor.computeFunction = function;
            descriptor.label = [
                NSString
                stringWithFormat:@"%@_nwg%d_nsg%d",
                                 name,
                                 workgroups,
                                 simdgroups];
            descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
            id<MTLComputePipelineState> pipeline = [
                device
                newComputePipelineStateWithDescriptor:descriptor
                options:MTLPipelineOptionNone
                reflection:nil
                error:&error];
            if (pipeline == nil) {
                print_error(descriptor.label, error);
            }
            return pipeline;
        };

        const auto make_reduce_pipeline =
            [&](int32_t workgroups) -> id<MTLComputePipelineState> {
            MTLFunctionConstantValues * constants =
                [[MTLFunctionConstantValues alloc] init];
            [constants setConstantValue:&head_elements
                                   type:MTLDataTypeInt
                                atIndex:kFunctionConstantReduce + 0];
            [constants setConstantValue:&workgroups
                                   type:MTLDataTypeInt
                                atIndex:kFunctionConstantReduce + 1];
            NSString * name = @"kernel_flash_attn_ext_vec_reduce";
            id<MTLFunction> function = [
                library
                newFunctionWithName:name
                constantValues:constants
                error:&error];
            if (function == nil) {
                print_error(name, error);
                return nil;
            }
            MTLComputePipelineDescriptor * descriptor =
                [[MTLComputePipelineDescriptor alloc] init];
            descriptor.computeFunction = function;
            descriptor.label = [
                NSString stringWithFormat:@"%@_dv%d_nwg%d",
                                         name,
                                         head_elements,
                                         workgroups];
            descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
            id<MTLComputePipelineState> pipeline = [
                device
                newComputePipelineStateWithDescriptor:descriptor
                options:MTLPipelineOptionNone
                reflection:nil
                error:&error];
            if (pipeline == nil) {
                print_error(descriptor.label, error);
            }
            return pipeline;
        };

        const int32_t control_nsg =
            simdgroups_for(32, cache_tokens);
        id<MTLComputePipelineState> main_32 =
            make_main_pipeline(32, control_nsg);
        id<MTLComputePipelineState> main_candidate =
            make_main_pipeline(candidate_nwg, candidate_nsg);
        id<MTLComputePipelineState> reduce_32 =
            make_reduce_pipeline(32);
        id<MTLComputePipelineState> reduce_candidate =
            candidate_nwg == 1
                ? nil
                : make_reduce_pipeline(candidate_nwg);
        if (main_32 == nil ||
            main_candidate == nil ||
            reduce_32 == nil ||
            (candidate_nwg != 1 && reduce_candidate == nil)) {
            return EXIT_FAILURE;
        }
        if (static_cast<NSUInteger>(candidate_nsg*32) >
            main_candidate.maxTotalThreadsPerThreadgroup) {
            std::fprintf(
                stderr,
                "candidate needs %d threads; pipeline exposes %zu\n",
                candidate_nsg*32,
                main_candidate.maxTotalThreadsPerThreadgroup);
            return EXIT_FAILURE;
        }

        const size_t query_bytes =
            static_cast<size_t>(rows)*head_elements*sizeof(float);
        const size_t cache_set_bytes =
            static_cast<size_t>(cache_tokens)*cache_stride*sizeof(uint16_t);
        const size_t cache_bytes =
            cache_set_bytes*static_cast<size_t>(cache_sets);
        const size_t mask_bytes =
            static_cast<size_t>(cache_tokens)*query_tokens*sizeof(uint16_t);
        const size_t output_bytes =
            static_cast<size_t>(rows)*head_elements*sizeof(float);
        const size_t temp_bytes =
            static_cast<size_t>(rows)*32*(head_elements + 2)*sizeof(float);
        const auto make_buffer = [&](size_t bytes) {
            return [
                device
                newBufferWithLength:bytes
                options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> query = make_buffer(query_bytes);
        id<MTLBuffer> key = make_buffer(cache_bytes);
        id<MTLBuffer> value = make_buffer(cache_bytes);
        id<MTLBuffer> mask = make_buffer(mask_bytes);
        id<MTLBuffer> pad = make_buffer(16);
        id<MTLBuffer> temp = make_buffer(temp_bytes);
        id<MTLBuffer> output_32 = make_buffer(output_bytes);
        id<MTLBuffer> output_1 = make_buffer(output_bytes);
        if (query == nil || key == nil || value == nil || mask == nil ||
            pad == nil || temp == nil || output_32 == nil ||
            output_1 == nil) {
            std::fprintf(stderr, "unable to allocate Metal buffers\n");
            return EXIT_FAILURE;
        }

        auto * query_values = static_cast<float *>(query.contents);
        for (size_t index = 0; index < query_bytes/sizeof(float); ++index) {
            query_values[index] =
                static_cast<float>(static_cast<int64_t>((index*17)%257) - 128)/
                256.0f;
        }
        auto * key_values = static_cast<__fp16 *>(key.contents);
        auto * value_values = static_cast<__fp16 *>(value.contents);
        const size_t cache_set_elements =
            cache_set_bytes/sizeof(uint16_t);
        for (size_t index = 0; index < cache_bytes/sizeof(uint16_t); ++index) {
            const size_t local_index = index%cache_set_elements;
            key_values[index] = static_cast<__fp16>(
                static_cast<float>(
                    static_cast<int64_t>((local_index*13)%263) - 131)/
                256.0f);
            value_values[index] = static_cast<__fp16>(
                static_cast<float>(
                    static_cast<int64_t>((local_index*19)%269) - 134)/
                256.0f);
        }
        auto * mask_values = static_cast<__fp16 *>(mask.contents);
        for (int32_t query_index = 0; query_index < query_tokens;
             ++query_index) {
            for (int32_t token = 0; token < cache_tokens; ++token) {
                mask_values[query_index*cache_tokens + token] =
                    token < active_tokens
                    ? static_cast<__fp16>(0.0f)
                    : static_cast<__fp16>(-65504.0f);
            }
        }
        std::memset(temp.contents, 0, temp_bytes);
        std::memset(output_32.contents, 0, output_bytes);
        std::memset(output_1.contents, 0, output_bytes);

        const uint64_t query_head_bytes =
            static_cast<uint64_t>(head_elements)*sizeof(float);
        const uint64_t query_token_bytes =
            query_head_bytes*kQueryHeads;
        const uint64_t cache_head_bytes =
            static_cast<uint64_t>(head_elements)*sizeof(uint16_t);
        const uint64_t cache_token_bytes =
            static_cast<uint64_t>(cache_stride)*sizeof(uint16_t);
        const uint64_t cache_layer_bytes = cache_set_bytes;
        const uint64_t mask_query_bytes =
            static_cast<uint64_t>(cache_tokens)*sizeof(uint16_t);
        FlashArgs args = {
            query_tokens,
            kQueryHeads,
            1,
            query_token_bytes,
            query_head_bytes,
            query_token_bytes*query_tokens,
            cache_tokens,
            kv_heads,
            1,
            cache_stride,
            cache_token_bytes,
            cache_head_bytes,
            cache_layer_bytes,
            cache_stride,
            cache_token_bytes,
            cache_head_bytes,
            cache_layer_bytes,
            cache_tokens,
            1,
            1,
            mask_query_bytes,
            mask_query_bytes*query_tokens,
            mask_query_bytes*query_tokens,
            kQueryHeads,
            query_tokens,
            1,
            1.0f/std::sqrt(static_cast<float>(head_elements)),
            0.0f,
            0.0f,
            0.0f,
            0,
            0.0f,
        };
        ReduceArgs reduce_args = {rows};

        id<MTLCommandQueue> queue = [device newCommandQueue];
        const auto run =
            [&](int32_t workgroups,
                int32_t cache_set,
                int32_t count,
                double * seconds) -> bool {
            const int32_t simdgroups =
                workgroups == 32
                    ? control_nsg
                    : candidate_nsg;
            id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder =
                [command_buffer computeCommandEncoder];
            for (int32_t iteration = 0; iteration < count; ++iteration) {
                [encoder setComputePipelineState:
                    workgroups == 32
                        ? main_32
                        : main_candidate];
                [encoder setBytes:&args length:sizeof(args) atIndex:0];
                [encoder setBuffer:query offset:0 atIndex:1];
                [encoder
                    setBuffer:key
                    offset:static_cast<NSUInteger>(cache_set)*cache_set_bytes
                    atIndex:2];
                [encoder
                    setBuffer:value
                    offset:static_cast<NSUInteger>(cache_set)*cache_set_bytes
                    atIndex:3];
                [encoder setBuffer:mask offset:0 atIndex:4];
                [encoder setBuffer:nil offset:0 atIndex:5];
                [encoder setBuffer:pad offset:0 atIndex:6];
                [encoder
                    setBuffer:workgroups == 1 ? output_1 : temp
                    offset:0
                    atIndex:7];
                    [encoder
                        setThreadgroupMemoryLength:
                        shared_bytes(head_elements, simdgroups)
                        atIndex:0];
                [encoder
                    dispatchThreadgroups:
                        MTLSizeMake(
                            query_tokens,
                            kQueryHeads,
                            workgroups)
                    threadsPerThreadgroup:
                        MTLSizeMake(32, simdgroups, 1)];

                if (workgroups != 1) {
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [encoder setComputePipelineState:
                        workgroups == 32
                            ? reduce_32
                            : reduce_candidate];
                    [encoder
                        setBytes:&reduce_args
                        length:sizeof(reduce_args)
                        atIndex:0];
                    [encoder setBuffer:temp offset:0 atIndex:1];
                    [encoder
                        setBuffer:
                            workgroups == 32
                                ? output_32
                                : output_1
                        offset:0
                        atIndex:2];
                    [encoder
                        dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                        threadsPerThreadgroup:
                            MTLSizeMake(32*workgroups, 1, 1)];
                }

                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
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
            double seconds = 0.0;
            const int32_t control_cache_set =
                (2*warmup)%cache_sets;
            const int32_t candidate_cache_set =
                cache_sets > 1
                    ? (control_cache_set + 1)%cache_sets
                    : control_cache_set;
            if (!run(32, control_cache_set, 64, &seconds) ||
                !run(candidate_nwg, candidate_cache_set, 64, &seconds)) {
                return EXIT_FAILURE;
            }
        }

        std::vector<double> control;
        std::vector<double> candidate;
        for (int32_t sample = 0; sample < samples; ++sample) {
            const bool control_first = sample%2 == 0;
            const int32_t control_cache_set =
                (2*sample)%cache_sets;
            const int32_t candidate_cache_set =
                cache_sets > 1
                    ? (control_cache_set + 1)%cache_sets
                    : control_cache_set;
            double control_seconds = 0.0;
            double candidate_seconds = 0.0;
            for (int order = 0; order < 2; ++order) {
                const bool run_control = (order == 0) == control_first;
                if (!run(
                        run_control ? 32 : candidate_nwg,
                        run_control
                            ? control_cache_set
                            : candidate_cache_set,
                        sequences_per_sample,
                        run_control
                            ? &control_seconds
                            : &candidate_seconds)) {
                    return EXIT_FAILURE;
                }
            }
            const double divisor = 1000.0/sequences_per_sample;
            control.push_back(control_seconds*divisor);
            candidate.push_back(candidate_seconds*divisor);
        }

        double seconds = 0.0;
        if (!run(32, 0, 1, &seconds) ||
            !run(candidate_nwg, 0, 1, &seconds)) {
            return EXIT_FAILURE;
        }

        const auto * lhs = static_cast<const float *>(output_32.contents);
        const auto * rhs = static_cast<const float *>(output_1.contents);
        const size_t output_elements = output_bytes/sizeof(float);
        double max_abs = 0.0;
        double squared_error = 0.0;
        for (size_t index = 0; index < output_elements; ++index) {
            const double error =
                std::abs(static_cast<double>(lhs[index]) - rhs[index]);
            max_abs = std::max(max_abs, error);
            squared_error += error*error;
        }
        const double rmse = std::sqrt(squared_error/output_elements);
        const double control_median = median(control);
        const double candidate_median = median(candidate);

        std::printf(
            "{"
            "\"device\":%s,"
            "\"head_elements\":%d,"
            "\"kv_heads\":%d,"
            "\"query_heads\":%d,"
            "\"query_tokens\":%d,"
            "\"cache_tokens\":%d,"
            "\"active_tokens\":%d,"
            "\"control_nwg\":32,"
            "\"control_nsg\":%d,"
            "\"candidate_nwg\":%d,"
            "\"candidate_nsg\":%d,"
            "\"control_dispatches\":2,"
            "\"candidate_dispatches\":%d,"
            "\"control_reduce_threads\":1024,"
            "\"candidate_reduce_threads\":%d,"
            "\"candidate_smem_bytes\":%zu,"
            "\"max_threadgroup_memory_bytes\":%zu,"
            "\"cache_sets\":%d,"
            "\"separate_cache_sets\":%s,"
            "\"samples\":%d,"
            "\"sequences_per_sample\":%d,"
            "\"timing\":\"command_buffer_gpu_timestamp\","
            "\"output_elements\":%zu,"
            "\"max_abs_error\":%.9g,"
            "\"rmse\":%.9g,"
            "\"control_mean_gpu_ms\":%.9f,"
            "\"control_median_gpu_ms\":%.9f,"
            "\"control_stddev_gpu_ms\":%.9f,"
            "\"candidate_mean_gpu_ms\":%.9f,"
            "\"candidate_median_gpu_ms\":%.9f,"
            "\"candidate_stddev_gpu_ms\":%.9f,"
            "\"latency_change_pct\":%.6f,"
            "\"control_samples_gpu_ms\":[",
            json_escape(device.name).UTF8String,
            head_elements,
            kv_heads,
            kQueryHeads,
            query_tokens,
            cache_tokens,
            active_tokens,
            control_nsg,
            candidate_nwg,
            candidate_nsg,
            candidate_nwg == 1 ? 1 : 2,
            candidate_nwg == 1
                ? 0
                : 32*candidate_nwg,
            shared_bytes(head_elements, candidate_nsg),
            device.maxThreadgroupMemoryLength,
            cache_sets,
            cache_sets > 1 ? "true" : "false",
            samples,
            sequences_per_sample,
            output_elements,
            max_abs,
            rmse,
            mean(control),
            control_median,
            stddev(control),
            mean(candidate),
            candidate_median,
            stddev(candidate),
            (candidate_median/control_median - 1.0)*100.0);
        print_samples(control);
        std::printf("],\"candidate_samples_gpu_ms\":[");
        print_samples(candidate);
        std::printf("]}\n");
    }

    return EXIT_SUCCESS;
}
