#include <metal_stdlib>
using namespace metal;

// Owned synthetic source counterpart to mrt_declare4.ll: the four-attachment
// declaring pass (`research/docs/23` §3.3, v24). One invocation reads one word
// from each attachment view and writes their xor into its own output view, so a
// single compute pass declares up to four attachment views as read-only — the
// v18 `mrt_declare` shape widened from two reads to four. A render case whose
// attachment list is shorter than four still resolves every one of its views
// against this one declaring pass; the extra reads are declared scratch the
// case is free not to attach.
kernel void mrt_declare4(device const uint *a [[buffer(0)]],
                         device const uint *b [[buffer(1)]],
                         device const uint *c [[buffer(2)]],
                         device const uint *d [[buffer(3)]],
                         device uint *output [[buffer(4)]]) {
    output[0] = a[0] ^ b[0] ^ c[0] ^ d[0];
}
