#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>

namespace {

NSString * json_escape(NSString * value) {
    NSData * data = [NSJSONSerialization dataWithJSONObject:@[value] options:0 error:nil];
    NSString * array = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    return [array substringWithRange:NSMakeRange(1, array.length - 2)];
}

}  // namespace

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            std::fprintf(stderr, "no Metal device is available\n");
            return EXIT_FAILURE;
        }

        std::printf(
            "{"
            "\"device\":%s,"
            "\"sampling_points\":{"
            "\"stage_boundary\":%s,"
            "\"draw_boundary\":%s,"
            "\"dispatch_boundary\":%s,"
            "\"tile_dispatch_boundary\":%s,"
            "\"blit_boundary\":%s"
            "},"
            "\"counter_sets\":[",
            json_escape(device.name).UTF8String,
            [device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary] ? "true" : "false",
            [device supportsCounterSampling:MTLCounterSamplingPointAtDrawBoundary] ? "true" : "false",
            [device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary] ? "true" : "false",
            [device supportsCounterSampling:MTLCounterSamplingPointAtTileDispatchBoundary] ? "true" : "false",
            [device supportsCounterSampling:MTLCounterSamplingPointAtBlitBoundary] ? "true" : "false");

        NSArray<id<MTLCounterSet>> * counter_sets = device.counterSets;
        for (NSUInteger set_index = 0; set_index < counter_sets.count; ++set_index) {
            id<MTLCounterSet> counter_set = counter_sets[set_index];
            std::printf(
                "%s{\"name\":%s,\"counters\":[",
                set_index == 0 ? "" : ",",
                json_escape(counter_set.name).UTF8String);
            for (NSUInteger counter_index = 0;
                 counter_index < counter_set.counters.count;
                 ++counter_index) {
                id<MTLCounter> counter = counter_set.counters[counter_index];
                std::printf(
                    "%s%s",
                    counter_index == 0 ? "" : ",",
                    json_escape(counter.name).UTF8String);
            }
            std::printf("]}");
        }
        std::printf("]}\n");
    }

    return EXIT_SUCCESS;
}
