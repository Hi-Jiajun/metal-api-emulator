# 每 stage 各自的 stage-buffer 上限（E-SB2）

本文记录本仓一个增量：`MAX_RENDER_STAGE_BUFFERS` 的计数轴从"两个 stage 合计"改成"**一个
stage 自己的**"，以及 provider 怎么用设备的真实窗口声明它执行到哪一条。叙事全貌在
`research/docs/23` §117；这篇只写 E 仓自己的契约、边界与可复跑的验证命令。

## 形状与今天的失败面

census v39 的 `render_provider_out_of_class_stage_buffer_shape` 仍读 92 行
（`evidence/gate3-census-v39-2026-09-19/v39-summary.txt`）。R9q 的只读侦察
（`.agents/tasks/root/r_stage_buffer_gt4-report.md`）把那批行的原文读完：它们是**两 stage
合计 13 条** `[[buffer(n)]]` 声明，不是重复槽、也不是 vertex layout 冲突。E-SB1（§108）把
常量从 4 抬到 8 时口径是 pipeline 级列表，所以 13 条在 8 之后仍然被拦。

Metal 的索引空间本来就是每个 stage 各自一份，因此正确的轴是 stage 自己：

```rust
// crates/metal-api-core/src/provider.rs
pub const MAX_RENDER_STAGE_BUFFERS: usize = 8;              // 一个 stage
pub const MAX_RENDER_STAGE_BUFFER_DECLARATIONS: usize = 16; // 两个 stage 的推论
```

## 设备事实：8 是 review ceiling，不是 per-stage 下限

Vulkan 的 Required Limits 表里，两条 stage 的 pipeline 下：

| 轴 | core 下限 | Lavapipe 读数 |
|---|---|---|
| `maxDescriptorSetStorageBuffers`（每 set） | `min(24, n × per-stage)` = 8 | 1 015 808 |
| `maxPerStageDescriptorStorageBuffers`（每 stage） | **4** | 1 015 808 |
| `maxPerStageResources` | 128 | 1 015 808 |

所以"每 stage 8"只能写成 review ceiling，执行窗口由设备回答——和 `attachment_dimension_window`
对 2048 的处理同形：

```rust
// crates/metal-api-vulkan/src/provider.rs
pub(crate) fn stage_buffer_window(limits: &vk::PhysicalDeviceLimits) -> StageBufferWindow
// per_stage = min(MAX_RENDER_STAGE_BUFFERS, maxPerStageDescriptorStorageBuffers)
// list      = 2 × per_stage
// per_set   = maxDescriptorSetStorageBuffers（arrangement 检查用）
```

## 能力位与 wire

| 面 | 内容 |
|---|---|
| `ProviderCapabilities` | 新字段 `max_render_stage_buffers_per_stage`（`0` = 不声明）+ 谓词 `declares_render_stage_buffer_per_stage_ceiling()` |
| Vulkan rail | `max_render_stage_buffers` = 16、`max_render_stage_buffers_per_stage` = 8（本机设备窗口） |
| native rail | `max_render_stage_buffers` = 8、每 stage 窗口不声明：13 条声明按名拒绝 |
| capability frame | 第二 tag 族新增 `0x00 0x07 <u32 BE>`；旧 frame 读作 `0`（更严的旧读法） |
| 声明块 | 长度前缀上限 8 → 16，解码后仍按 stage 复核 8；`RenderStageBufferCount` 带 stage |

一个诚实的边界：新 frame 的 window 块若被**旧解码器**读到，会以 `TrailingPayload` 整帧拒绝
（`decode_response_payload` 末尾的 `decoder.finish()`），不是静默忽略；方向是 fail-closed，
与 E-TX9 的 tail 块同一性质。

## 两条 rail 的执行语义

- Vulkan：每 stage ≤ 窗口、每个 set 的 storage buffer 条数 ≤ 设备 per-set 窗口，越界按名拒绝
  （直构请求的防御臂）；
- arrangement：reviewed 对是 set 1（vertex）/set 2（fragment），translated 阶段用反射自己的
  slot；两段落同一个 set 时按 set 检查（转译默认布局 + 合并 set 0 的那条路）；
- native：reviewed 模块每 stage 只绑一个槽，没有 Apple 读数覆盖更宽的窗口，因此不声明。

## 证据与可复跑命令

| 面 | 位置 |
|---|---|
| 13 条声明的执行读数（Lavapipe） | `crates/metal-api-vulkan/tests/render_e2e.rs::a_per_stage_stage_buffer_shape_enters_the_rail_and_lands_its_bytes` |
| 9 条单 stage 的按名拒 | 同文件 `a_stage_list_above_the_per_stage_ceiling_is_refused_by_name` |
| 设备窗口的纯函数读数 | `crates/metal-api-vulkan/src/provider.rs` 的单测（core 下限 4/8、Lavapipe 8/16、中间值 6/12） |
| 契约三态（形状 / 旧读法 / 设备窗口） | `crates/metal-api-core/src/provider.rs::a_per_stage_window_narrows_the_stage_buffer_shape_a_snapshot_admits` |
| conformance | `conformance/suite-v42.json`、`conformance/test_suite_v42.py` |
| fixture | `crates/metal-api-vulkan/tests/fixtures/render_stage_buffer_seven.vert.ll`（llvm-as 22.1.8 可复现） |

```bash
cd /home/hiliang/hackintosh/worktrees/metal-stage-ceiling
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  cargo test --locked -p metal-api-vulkan --test render_e2e -- \
    --exact a_per_stage_stage_buffer_shape_enters_the_rail_and_lands_its_bytes --nocapture

VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  cargo run --locked -p metal-smoke --bin provider-capture -- \
    --suite conformance/suite-v42.json --output /tmp/v42.json
python3 conformance/compare.py --suite conformance/suite-v42.json --check /tmp/v42.json
```

期望帧钉在 `e0`×16 上：13 份 view 的字节里，fragment 的六个槽是 5 × 32/255 + 64/255，vertex
的七个槽把 `b0` 的三个 `float2` 位置加六个零偏移——丢掉任一声明都会换帧，改 vertex 的偏移
会把三角形推出 2×2 viewport 而留下 clear sentinel。

## 消费方（另一仓，本 change 不含）

reims 的随动清单写在 `.agents/tasks/root/e_stage_buffer_per_stage_ceiling-report.md` §R：读
`max_render_stage_buffers_per_stage`（块缺失时保持 R9q 的合并规则）、按 stage 计数、并把
`stage_buffer_shape` 的读数从"合计 > 8"改成"某个 stage > 窗口"。
