#include <metal_stdlib>
using namespace metal;

// Source counterpart to the attributed indexed LLVM fixture; see NOTICE.md.
kernel void kernel_dispatch_threads_boundary_barrier(
    device uint *output [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]]) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    output[gid.y * 10 + gid.x] = groupSize.y * 100 + groupSize.x;
}
