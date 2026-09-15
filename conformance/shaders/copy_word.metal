#include <metal_stdlib>
using namespace metal;

// Owned synthetic source counterpart to kernel_copy_word.ll.
kernel void copy_word(device const uint *input [[buffer(0)]],
                      device uint *output [[buffer(1)]]) {
    output[0] = input[0];
}
