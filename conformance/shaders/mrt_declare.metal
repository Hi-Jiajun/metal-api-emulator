#include <metal_stdlib>
using namespace metal;

// Owned synthetic source counterpart to mrt_declare.ll: one invocation xors
// the two attachment words and writes the result to the output view.
kernel void mrt_declare(device const uint *left [[buffer(0)]],
                        device const uint *right [[buffer(1)]],
                        device uint *output [[buffer(2)]]) {
    output[0] = left[0] ^ right[0];
}
