#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstring>

static void printObserved(const uint8_t bytes[16]) {
    for (int i = 0; i < 16; ++i) {
        std::printf("%02x", bytes[i]);
    }
}

int main(int argc, const char *argv[]) {
    (void)argc;
    (void)argv;

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            std::printf("icb_probe: SKIP (no Metal device)\n");
            return 0;
        }

        // The selector the Swift overlay cannot express at compile time. If it is
        // absent on the ObjC runtime too, that is a platform block, not a runtime
        // failure: report UNAVAILABLE and exit nonzero so a CI step cannot turn it
        // into a false PASS.
        SEL newIcbSelector = @selector(newIndirectCommandBufferWithDescriptor:maxCommandCount:options:);
        if (![device respondsToSelector:newIcbSelector]) {
            std::printf("icb_probe: UNAVAILABLE (selector missing: "
                        "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:)\n");
            return 2;
        }

        const char *shaderPath = "conformance/shaders/render_offscreen_2x2.metal";
        NSString *path = [NSString stringWithUTF8String:shaderPath];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithFile:path error:&error];
        if (library == nil) {
            std::printf("icb_probe: FAIL (cannot build shader library from %s: %s)\n",
                        shaderPath,
                        error.localizedDescription.UTF8String ?: "(no detail)");
            return 1;
        }

        id<MTLFunction> vertexFunction = [library newFunctionWithName:@"render_fullscreen_triangle"];
        id<MTLFunction> fragmentFunction = [library newFunctionWithName:@"render_solid_rgba8"];
        if (vertexFunction == nil || fragmentFunction == nil) {
            std::printf("icb_probe: FAIL (reviewed stage entries were not found)\n");
            return 1;
        }

        MTLRenderPipelineDescriptor *pipelineDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
        pipelineDescriptor.vertexFunction = vertexFunction;
        pipelineDescriptor.fragmentFunction = fragmentFunction;
        pipelineDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        NSError *pipelineError = nil;
        id<MTLRenderPipelineState> pipeline =
            [device newRenderPipelineStateWithDescriptor:pipelineDescriptor error:&pipelineError];
        if (pipeline == nil) {
            std::printf("icb_probe: FAIL (cannot build the render pipeline: %s)\n",
                        pipelineError.localizedDescription.UTF8String ?: "(no detail)");
            return 1;
        }

        MTLTextureDescriptor *textureDescriptor =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                               width:2
                                                              height:2
                                                           mipmapped:NO];
        textureDescriptor.usage = MTLTextureUsageRenderTarget;
        textureDescriptor.storageMode = MTLStorageModeShared;
        id<MTLTexture> target = [device newTextureWithDescriptor:textureDescriptor];
        if (target == nil) {
            std::printf("icb_probe: FAIL (cannot allocate the 2x2 rgba8 attachment)\n");
            return 1;
        }

        MTLIndirectCommandBufferDescriptor *icbDescriptor =
            [[MTLIndirectCommandBufferDescriptor alloc] init];
        icbDescriptor.commandTypes = MTLIndirectCommandTypeDraw;
        icbDescriptor.inheritBuffers = NO;
        icbDescriptor.inheritPipelineState = NO;
        icbDescriptor.maxVertexBufferBindCount = 0;
        icbDescriptor.maxFragmentBufferBindCount = 0;
        id<MTLIndirectCommandBuffer> icb =
            [device newIndirectCommandBufferWithDescriptor:icbDescriptor
                                           maxCommandCount:1
                                                   options:MTLResourceStorageModeShared];
        if (icb == nil) {
            std::printf("icb_probe: FAIL (cannot allocate the draw indirect command buffer)\n");
            return 1;
        }

        SEL renderCommandSelector = @selector(indirectRenderCommandAtIndex:);
        if (![icb respondsToSelector:renderCommandSelector]) {
            std::printf("icb_probe: UNAVAILABLE (selector missing: indirectRenderCommandAtIndex:)\n");
            return 2;
        }

        id<MTLIndirectRenderCommand> command = [icb indirectRenderCommandAtIndex:0];
        if (command == nil) {
            std::printf("icb_probe: FAIL (indirectRenderCommandAtIndex:0 returned nil)\n");
            return 1;
        }

        SEL setPipelineSelector = @selector(setRenderPipelineState:);
        SEL drawSelector = @selector(drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:);
        if (![command respondsToSelector:setPipelineSelector] ||
            ![command respondsToSelector:drawSelector]) {
            std::printf("icb_probe: UNAVAILABLE (selector missing: "
                        "setRenderPipelineState: or "
                        "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:)\n");
            return 2;
        }

        [command setRenderPipelineState:pipeline];
        [command drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:0
                    vertexCount:3
                  instanceCount:1
                   baseInstance:0];

        MTLRenderPassDescriptor *renderPassDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];
        renderPassDescriptor.colorAttachments[0].texture = target;
        renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
        renderPassDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
        // fefefefe is the sentinel the fixture must not read back: if the draw
        // never ran, the attachment would still be this clear colour.
        renderPassDescriptor.colorAttachments[0].clearColor =
            MTLClearColorMake(254.0 / 255.0, 254.0 / 255.0, 254.0 / 255.0, 254.0 / 255.0);

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLRenderCommandEncoder> encoder =
            [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
        if (encoder == nil) {
            std::printf("icb_probe: FAIL (cannot create the render encoder)\n");
            return 1;
        }

        SEL executeSelector = @selector(executeCommandsInBuffer:withRange:);
        if (![encoder respondsToSelector:executeSelector]) {
            std::printf("icb_probe: UNAVAILABLE (selector missing: executeCommandsInBuffer:withRange:)\n");
            [encoder endEncoding];
            return 2;
        }

        [encoder setViewport:(MTLViewport){0.0, 0.0, 2.0, 2.0, 0.0, 1.0}];
        [encoder executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.status != MTLCommandBufferStatusCompleted || commandBuffer.error != nil) {
            std::printf("icb_probe: FAIL (Metal execution status %ld: %s)\n",
                        (long)commandBuffer.status,
                        commandBuffer.error.localizedDescription.UTF8String ?: "(no detail)");
            return 1;
        }

        uint8_t observed[16] = {0};
        [target getBytes:observed
              bytesPerRow:8
               fromRegion:MTLRegionMake2D(0, 0, 2, 2)
             mipmapLevel:0];

        static const uint8_t expected[16] = {
            0x40, 0x80, 0xc0, 0xff, 0x40, 0x80, 0xc0, 0xff,
            0x40, 0x80, 0xc0, 0xff, 0x40, 0x80, 0xc0, 0xff,
        };
        static const uint8_t sentinel[16] = {
            0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe,
            0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe,
        };

        if (std::memcmp(observed, sentinel, sizeof(sentinel)) == 0) {
            std::printf("icb_probe: FAIL (attachment still holds the fefefefe sentinel)\n");
            return 1;
        }
        if (std::memcmp(observed, expected, sizeof(expected)) != 0) {
            std::printf("icb_probe: FAIL (observed ");
            printObserved(observed);
            std::printf(", expected 4080c0ff4080c0ff4080c0ff4080c0ff)\n");
            return 1;
        }

        std::printf("icb_probe: PASS (4080c0ff4080c0ff4080c0ff4080c0ff)\n");
        return 0;
    }
}
