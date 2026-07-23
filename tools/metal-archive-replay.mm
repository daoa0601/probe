#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr NSUInteger kElementCount = 256;

void print_usage(const char * program) {
    std::fprintf(
        stderr,
        "usage: %s METALLIB BINARY_ARCHIVE FUNCTION EXPECTED_ADDEND\n",
        program);
}

bool parse_uint32(const char * text, uint32_t * value) {
    char * end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || parsed > UINT32_MAX) {
        std::fprintf(stderr, "invalid expected addend: %s\n", text);
        return false;
    }
    *value = static_cast<uint32_t>(parsed);
    return true;
}

void print_error(NSString * context, NSError * error) {
    std::fprintf(
        stderr,
        "%s: domain=%s code=%ld description=%s\n",
        context.UTF8String,
        error.domain.UTF8String,
        static_cast<long>(error.code),
        error.localizedDescription.UTF8String);
}

NSString * json_escape(NSString * value) {
    NSData * data = [NSJSONSerialization dataWithJSONObject:@[value] options:0 error:nil];
    NSString * array = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    return [array substringWithRange:NSMakeRange(1, array.length - 2)];
}

}  // namespace

int main(int argc, const char * argv[]) {
    if (argc != 5) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    uint32_t expected_addend = 0;
    if (!parse_uint32(argv[4], &expected_addend)) {
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
        NSURL * archive_url =
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[2]]];
        NSString * function_name = [NSString stringWithUTF8String:argv[3]];

        id<MTLLibrary> library = [device newLibraryWithURL:library_url error:&error];
        if (library == nil) {
            print_error(@"load metallib", error);
            return EXIT_FAILURE;
        }
        id<MTLFunction> function = [library newFunctionWithName:function_name];
        if (function == nil) {
            std::fprintf(stderr, "function not found: %s\n", argv[3]);
            return EXIT_FAILURE;
        }

        MTLBinaryArchiveDescriptor * archive_descriptor =
            [[MTLBinaryArchiveDescriptor alloc] init];
        archive_descriptor.url = archive_url;
        id<MTLBinaryArchive> archive =
            [device newBinaryArchiveWithDescriptor:archive_descriptor error:&error];
        if (archive == nil) {
            print_error(@"load binary archive", error);
            return EXIT_FAILURE;
        }

        MTLComputePipelineDescriptor * pipeline_descriptor =
            [[MTLComputePipelineDescriptor alloc] init];
        pipeline_descriptor.label = function_name;
        pipeline_descriptor.computeFunction = function;
        pipeline_descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
        pipeline_descriptor.binaryArchives = @[archive];

        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithDescriptor:pipeline_descriptor
                                                  options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                               reflection:nil
                                                    error:&error];
        if (pipeline == nil) {
            print_error(@"create pipeline from required archive hit", error);
            return EXIT_FAILURE;
        }

        const NSUInteger bytes = kElementCount * sizeof(uint32_t);
        id<MTLBuffer> source =
            [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> destination =
            [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (source == nil || destination == nil || queue == nil) {
            std::fprintf(stderr, "unable to allocate Metal resources\n");
            return EXIT_FAILURE;
        }

        auto * source_values = static_cast<uint32_t *>(source.contents);
        auto * destination_values = static_cast<uint32_t *>(destination.contents);
        for (NSUInteger index = 0; index < kElementCount; ++index) {
            source_values[index] =
                static_cast<uint32_t>(index * 0x01010101u) ^ 0x5a5aa5a5u;
            destination_values[index] = 0xdeadbeefu;
        }

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:source offset:0 atIndex:0];
        [encoder setBuffer:destination offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(kElementCount, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            print_error(@"execute archived pipeline", command_buffer.error);
            return EXIT_FAILURE;
        }

        NSUInteger mismatches = 0;
        uint32_t first_actual = 0;
        uint32_t first_expected = 0;
        for (NSUInteger index = 0; index < kElementCount; ++index) {
            const uint32_t expected = source_values[index] + expected_addend;
            if (destination_values[index] != expected) {
                if (mismatches == 0) {
                    first_actual = destination_values[index];
                    first_expected = expected;
                }
                ++mismatches;
            }
        }

        std::printf(
            "{"
            "\"device\":%s,"
            "\"function\":%s,"
            "\"required_archive_hit\":true,"
            "\"expected_addend\":%u,"
            "\"elements\":%lu,"
            "\"mismatches\":%lu,"
            "\"first_expected\":%u,"
            "\"first_actual\":%u,"
            "\"thread_execution_width\":%lu"
            "}\n",
            json_escape(device.name).UTF8String,
            json_escape(function_name).UTF8String,
            expected_addend,
            static_cast<unsigned long>(kElementCount),
            static_cast<unsigned long>(mismatches),
            first_expected,
            first_actual,
            static_cast<unsigned long>(pipeline.threadExecutionWidth));
        return mismatches == 0 ? EXIT_SUCCESS : 2;
    }
}
