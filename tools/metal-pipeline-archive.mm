#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr NSUInteger kFunctionConstantMulMv = 600;

void print_usage(const char * program) {
    std::fprintf(
        stderr,
        "usage: %s METALLIB ARCHIVE FUNCTION NSG NE12 R2 R3\n"
        "       NSG=0 loads an unspecialized function without Q4 constants\n",
        program);
}

bool parse_int16(const char * text, const char * name, int16_t * value) {
    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < INT16_MIN || parsed > INT16_MAX) {
        std::fprintf(stderr, "invalid %s value: %s\n", name, text);
        return false;
    }
    *value = static_cast<int16_t>(parsed);
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

}  // namespace

int main(int argc, const char * argv[]) {
    if (argc != 8) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    int16_t nsg = 0;
    int16_t ne12 = 0;
    int16_t r2 = 0;
    int16_t r3 = 0;
    if (!parse_int16(argv[4], "nsg", &nsg) ||
        !parse_int16(argv[5], "ne12", &ne12) ||
        !parse_int16(argv[6], "r2", &r2) ||
        !parse_int16(argv[7], "r3", &r3)) {
        return EXIT_FAILURE;
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            std::fprintf(stderr, "no Metal device is available\n");
            return EXIT_FAILURE;
        }

        NSString * library_path = [NSString stringWithUTF8String:argv[1]];
        NSString * archive_path = [NSString stringWithUTF8String:argv[2]];
        NSString * function_name = [NSString stringWithUTF8String:argv[3]];
        NSURL * library_url = [NSURL fileURLWithPath:library_path];
        NSURL * archive_url = [NSURL fileURLWithPath:archive_path];

        NSError * error = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:library_url error:&error];
        if (library == nil) {
            print_error(@"load metallib", error);
            return EXIT_FAILURE;
        }

        id<MTLFunction> function = nil;
        if (nsg == 0) {
            function = [library newFunctionWithName:function_name];
        } else {
            MTLFunctionConstantValues * constants = [[MTLFunctionConstantValues alloc] init];
            [constants setConstantValue:&nsg
                                   type:MTLDataTypeShort
                                atIndex:kFunctionConstantMulMv + 0];
            [constants setConstantValue:&ne12
                                   type:MTLDataTypeShort
                                atIndex:kFunctionConstantMulMv + 2];
            [constants setConstantValue:&r2
                                   type:MTLDataTypeShort
                                atIndex:kFunctionConstantMulMv + 3];
            [constants setConstantValue:&r3
                                   type:MTLDataTypeShort
                                atIndex:kFunctionConstantMulMv + 4];
            function = [library newFunctionWithName:function_name
                                      constantValues:constants
                                               error:&error];
        }
        if (function == nil) {
            print_error(@"specialize function", error);
            return EXIT_FAILURE;
        }

        MTLComputePipelineDescriptor * pipeline_descriptor =
            [[MTLComputePipelineDescriptor alloc] init];
        pipeline_descriptor.label = function_name;
        pipeline_descriptor.computeFunction = function;
        pipeline_descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;

        MTLComputePipelineReflection * reflection = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithDescriptor:pipeline_descriptor
                                                  options:MTLPipelineOptionBindingInfo
                                               reflection:&reflection
                                                    error:&error];
        if (pipeline == nil) {
            print_error(@"compile pipeline", error);
            return EXIT_FAILURE;
        }

        MTLBinaryArchiveDescriptor * archive_descriptor =
            [[MTLBinaryArchiveDescriptor alloc] init];
        id<MTLBinaryArchive> archive =
            [device newBinaryArchiveWithDescriptor:archive_descriptor error:&error];
        if (archive == nil) {
            print_error(@"create binary archive", error);
            return EXIT_FAILURE;
        }
        archive.label = function_name;

        if (![archive addComputePipelineFunctionsWithDescriptor:pipeline_descriptor error:&error]) {
            print_error(@"add pipeline to binary archive", error);
            return EXIT_FAILURE;
        }

        [[NSFileManager defaultManager] removeItemAtURL:archive_url error:nil];
        if (![archive serializeToURL:archive_url error:&error]) {
            print_error(@"serialize binary archive", error);
            return EXIT_FAILURE;
        }

        std::printf(
            "{"
            "\"device\":%s,"
            "\"function\":%s,"
            "\"nsg\":%d,"
            "\"ne12\":%d,"
            "\"r2\":%d,"
            "\"r3\":%d,"
            "\"thread_execution_width\":%lu,"
            "\"max_threads_per_threadgroup\":%lu,"
            "\"static_threadgroup_memory_bytes\":%lu,"
            "\"counter_sampling_stage_boundary\":%s,"
            "\"counter_sampling_dispatch_boundary\":%s"
            "}\n",
            json_escape(device.name).UTF8String,
            json_escape(function_name).UTF8String,
            nsg,
            ne12,
            r2,
            r3,
            static_cast<unsigned long>(pipeline.threadExecutionWidth),
            static_cast<unsigned long>(pipeline.maxTotalThreadsPerThreadgroup),
            static_cast<unsigned long>(pipeline.staticThreadgroupMemoryLength),
            [device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary] ?
                "true" : "false",
            [device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary] ?
                "true" : "false");
    }

    return EXIT_SUCCESS;
}
