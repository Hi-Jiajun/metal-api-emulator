# 模块自己声明的 16-bit 能力对：设备自答的 `Float16` + `Int16`（E-SH1）

本文记录本仓一个增量：census v48 残余的 LPF pipeline（`pipe=58` =
`fixed_vert_lpf_gen` × `fixed_frag_lpf_cpf`）翻译出的 fragment 模块**自己**声明
`OpCapability Float16` 与 `OpCapability Int16`，而 rail 的能力门把它们按名拒掉，pipeline
连翻译都过不去。它是 `docs/FRAGMENT-OUTPUT-SUPERSET.md`（E-FOS1）的姊妹篇：同样是
"设备自己答"的一个面，只是这次答案是 **SPIR-V 能力子集**而不是管线接口。

## 形状与今天的失败面

侦察报告 `.agents/tasks/root/e_texture_sampler_form-report.md` §2 的门 2 与
`evidence/texture-sampler-d94d8da-2026-09-20/` 的读数：

* 该模块的 SPIR-V 声明 `Shader / Int64 / Int8 / Float16 / Int16`
  （`12-pipe58-spirv-capabilities.txt`）；
* `TranslatedRenderStage` 的能力扫描当时只放行
  `Shader / ImageQuery / Int8 / Int64 / Sampled1D / SampledBuffer / FloatControls2`，
  于是翻译直接以
  `SPIR-V capability 9 requires a Vulkan feature outside the Phase 1 subset`
  结束（`11-pipe58-translatability-probe.log`，capability 9 = `Float16`）；
* 注册门（`render.rs::validate_module_capabilities`）读同一份策略，所以即便翻译改成
  `ADMITTING`，注册一样会按名拒 —— 这条 pipeline 的 29 条 draw 只是换一个桶，**不会**进类。

Vulkan 的规则是设备的：`Float16` 只在设备建 device 时开了 `shaderFloat16` 时合法，
`Int16` 只在开了核心 `shaderInt16` 时合法（给未启用的能力声明是 invalid usage，不是
"慢一点"）。所以这一刀把两个读数变成**设备自答**，并把答案放进能力帧。

## 政策与能力的按方向语义（唯一真源）

```rust
// crates/metal-api-vulkan/src/lib.rs
pub struct SpirvFeaturePolicy { float_controls2: bool, float16: bool, int16: bool }
pub const fn with_float16(self, admitted: bool) -> Self
pub const fn with_int16(self, admitted: bool) -> Self
pub const fn half(self) -> bool            // float16 && int16
pub const ADMITTING: Self                  // 提问面：把这个门能表达的能力全开

pub struct HalfShaderSupport { float16: bool, int16: bool }   // 设备读数
// float16_reported() / int16_reported() / enabled() (=合取) / policy()
```

| 面 | 语义 |
|---|---|
| `PHASE1` | 两个位全关：设备没报能力时的 fail-closed 答案，句子与之前逐字相同 |
| `ADMITTING` | **不是设备答案**，是"模块自己声明了什么"的提问面，只被 `declared_shader_capabilities` 用来读模块 |
| `HalfShaderSupport::enabled()` | 两个读数都真才为真，就是能力帧那**一位** |
| 能力扫描 | 逐个 `OpCapability` 判定：`Float16` 只看 `float16` 位，`Int16` 只看 `int16` 位 |

两个位与一位的关系是有意的：**门按能力逐条查**（一个设备可能只有 `shaderFloat16`），
而**帧只发布合取**（需要这个面的模块两个能力都声明，只答一个的设备答不了它）。一位的
缺段读法是 `false`，也就是"这个 provider 的子集里没有这一对"。

## capability frame

| 面 | 内容 |
|---|---|
| 位与谓词 | `crates/metal-api-core/src/provider.rs` 的 `supports_render_half_capabilities` / `declares_render_half_capabilities()` |
| 族内 tag | `CAPABILITY_RENDER_HALF_CAPABILITIES_TAIL = 0x10`（`command_codec.rs`，紧随 `0x0F`） |
| 编码 | `0x00`（转义）+ `0x10`（族内 tag）+ 一个 bool；不声明就一个字节都不多写 |
| 旧帧 | 新解码器读 `false`（fail-closed）；旧解码器读到 `0x10` 会以 `UnknownCapabilityTail` 拒整帧 |
| 未知族内 tag | 仍然 `UnknownCapabilityTail`（乱序/重复/超集一律拒） |

## 两条轨的取值是事实

| 轨 | 取值 | 依据 |
|---|---|---|
| Vulkan | 设备读数（`shaderFloat16` ∧ `shaderInt16`） | `HalfShaderSupport` 从与 mirror-clamp 同一个 `VkPhysicalDeviceFeatures2` 链里读出；两个 feature 都**只在设备报了才 enable** |
| native | `false`（保持） | 该轨执行的是它自己写的 reviewed MSL，没有任何一个模块把 float 收窄成 `half` 再读位；它也没有 SPIR-V 前端，声明这一位的模块命中不了任何 reviewed arm，注册按名拒（`native_render_source_not_reviewed`，自己的句子） |

`capabilities_from_limits` **不**声明这一位（limit 说不出 feature），
`VulkanExecutor::provider_capabilities()` 用设备读数覆盖它 —— 与 depth/stencil resolve、
whole-binding arm 的写法同形。

## 证据与可复跑命令

| 面 | 位置 |
|---|---|
| 设备读数（`shaderFloat16`/`shaderInt16`、policy 两位、帧一位） | `crates/metal-api-vulkan/tests/render_half_capabilities_e2e.rs` |
| fixture（声明这一对的模块 + 它自己的孪生） | `tests/fixtures/render_half_truncated_rgba8.frag.ll`、`..._off_rgba8.frag.ll` |
| 能力扫描的两条臂（含注入的假读数） | `crates/metal-api-vulkan/src/lib.rs` 的单测 `the_16_bit_pair_rides_the_device_policy_one_capability_at_a_time`、`half_shader_support_answers_only_for_the_pair` |
| 已声明能力的 walk（在第一个 `OpFunction` 停下） | 同文件的 `the_declared_capabilities_walk_stops_at_the_first_function` |
| wire 往返 / 只声明这一对 / 旧帧 / 未知族内 tag | `crates/metal-api-ipc/src/command.rs` 的单测 `the_half_capability_pair_block_is_the_tail_familys_next_tag`、`an_only_half_capability_declaration_still_writes_the_extended_payload` |
| 默认快照不声明这一位 | `crates/metal-api-core/src/provider.rs` 的单测 |
| 两条轨快照声明（Vulkan=设备读数 / native=`false`） | `crates/metal-api-vulkan/src/provider.rs`、`crates/metal-api-native/src/render.rs` |

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  cargo test -p metal-api-vulkan --test render_half_capabilities_e2e -- --nocapture
# RTX 5060：交叉编译后用 Windows 侧执行同一个测试二进制
cargo test -p metal-api-vulkan --locked --target x86_64-pc-windows-gnu \
  --test render_half_capabilities_e2e --no-run
```

fixture 的语义是可证伪的：模块把 `half(64/255)` 的位模式右移一位得到的 `i16`
（`0x3404 >> 1 == 0x1a02 == 6658`）与它自己语句里写的值比较，命中就落
`40 80 c0 ff`（reviewed offscreen fixture 自己的 texel），不命中就落 `ff 80 c0 00`。
孪生模块只改那个被比较的数，所以"帧确实由这条收窄路径决定"是可读的，而不是"跑过就行"。

## 消费方（另一仓，本 change 不含）

reims 侧的随动任务书由主代理另派：读新 tag，**按名**兜底那道门 —— 模块自己声明了这一对
而设备帧不答它时，class 直接按名把 draw 留在 engine（slug
`render_provider_out_of_class_module_capability`），**不发生 submission**；位在位时照旧
交给 provider。顺序纪律是硬的：**E 先 R 后** —— 反过来就是"class 放行、provider 拒"的
丢画形状（census 红线 `draws_skipped_after_engine_refusal`）。

## 未决

1. **`Float16Buffer` 与 16-bit storage 面**：本增量只开 `Float16`/`Int16` 两个
   capability，`Float16Buffer`（8）、`StorageBuffer16BitAccess`（4433）等一个都没开，
   遇到时仍然按名拒（单测固定）。
2. **一能力一位 vs 一位合取**：帧只发布合取，所以"设备只有 `shaderFloat16` 而模块只声明
   `Float16`"这种形状会被 class 保守地留在 engine（fail-closed 方向）。真机上两个读数
   至今一致，等 Apple/驱动侧出现分歧再评估要不要拆成两位。
3. **`FloatControls2` 没有帧面**：它今天同样只在设备策略里，class 侧没有对应的按名门 ——
   要补的话是同一个模式（新 tag + 新门），本增量不做。
