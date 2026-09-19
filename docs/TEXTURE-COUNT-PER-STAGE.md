# 每 stage 各自的采样纹理上限（E-TC1）

本文记录本仓一个增量：`MAX_RENDER_TEXTURES` 的计数轴从"pass 的整张列表"改成"**一个
stage 自己的**"，以及 provider 怎么用设备的真实窗口声明它执行到哪一条。它和
`docs/STAGE-BUFFER-PER-STAGE-CEILING.md`（E-SB2）是同一个成法，差别只在纹理这一面今天
只有一个 stage 会声明。

## 形状与今天的失败面

真机预览轮（`/mnt/c/tmp/reims-vgpu-fail.log`，02:0x）把 LPF pipeline 的残桶从
`stage_buffer_footprint` 推进到了新的一道门：

```
OFF render_provider_out_of_class pipe=58 reason=a fragment stage that declares 13 sampled
textures stays on the engine: the canonical contract states 8 (`MAX_RENDER_TEXTURES`), and a
longer list is refused by name (`render_texture_limit`) rather than executed with the rest
dropped t=112992
```

整轮共 42 行 `render_provider_out_of_class_texture_count`，**每一行的数都是 13**
（`rg -o 'reason=a fragment stage that declares [0-9]+ sampled textures' | sort | uniq -c`
只读出 `13` 一种）。也就是说：这不是"很多档宽度"，而是一个具体的声明形状——一个 fragment
stage 自己的 `[[texture(n)]]` 自变量表里有 13 条。

## 契约轴：从"整张列表"到一个 stage 自己的窗口

Metal 的纹理索引空间本来就是每个 stage 各自一份：`[[texture(n)]]` 是该 stage 自己的
texture argument table 里的自变量。八是第一个 widen 增量在**pass 的整张列表**上写的数，
而 13 条声明在 8 之后仍然被拦，正是因为两条 stage 的合计口径与 stage 自己的口径是两件事。

```rust
// crates/metal-api-core/src/provider.rs
pub const MAX_RENDER_TEXTURES: usize = 16;                 // 一个 stage
pub const MAX_RENDER_TEXTURE_DECLARATIONS: usize = MAX_RENDER_TEXTURES * 2; // 两条 stage 的推论
```

16 是 **review ceiling 而不是本契约的发明**：Vulkan 的 Required Limits 表里
`maxPerStageDescriptorSampledImages` 的 core 下限正是 **16**，所以任何 conformant 设备都不
必拒绝一个停在 16 以内的 stage；13 是 census 读到过的最宽形状，16 给出三条声明余量；同一张
自变量表上的另一条轴 `MAX_RENDER_SAMPLERS` 也已经是 16。纹理由此从"设备事实之外的 8"
变成"平台自己已经承诺的 16"。

列表侧的 32 是**推论而不是第二条轴**：render pipeline 恰好两条 stage，每条最多 16 条声明，
所以更长的列表必然把某一条 stage 数了两遍。它作为独立常量的唯一用途是 wire 的长度前缀
（`command_codec.rs` 的纹理块在读到任何 stage 之前就要判上界）。今天 canonical 契约里只有
fragment stage 会声明采样纹理（`RenderPassDescriptor::textures` 与
`RenderPipelineContract::textures` 都是它自己的列表），所以运行时的截断点是
`MAX_RENDER_TEXTURES`，而 `MAX_RENDER_TEXTURE_DECLARATIONS` 是 wire 的宽度与两条 stage
算术上的上界。

结构校验因此是两条规则，顺序固定：

1. 列表长度 > `MAX_RENDER_TEXTURE_DECLARATIONS` ⇒ `RenderTextureLimitExceeded { stage: None, .. }`
   （"整张列表"那一臂，wire 前缀在这一臂被拒）；
2. 列表长度 > `MAX_RENDER_TEXTURES` ⇒ `RenderTextureLimitExceeded { stage: Some(Fragment), .. }`
   （stage 自己那一臂，拒绝里带 stage、requested、maximum）。

17 条声明走第二臂、33 条走第一臂，两者都是具名拒绝而不是"悄悄丢掉多余的"。

## 设备窗口与能力位

| 面 | 内容 |
|---|---|
| `ProviderCapabilities` | `max_render_textures` 变成 pass 的列表界；新字段 `max_render_textures_per_stage`（`0` = 不声明）+ 谓词 `declares_render_texture_per_stage_ceiling()` |
| Vulkan rail | `render_texture_window(limits)`：`per_stage = min(MAX_RENDER_TEXTURES, maxPerStageDescriptorSampledImages)`、`list = 2 × per_stage`、`per_set = maxDescriptorSetSampledImages` |
| native rail | `max_render_textures` 保持 1（reviewed 模块只读一个自变量），每 stage 窗口不声明：13 条按名拒 |
| capability frame | 第二家族新 tag **`0x0F`**，载荷一个 big-endian `u32`（与 `0x07` 的 stage-buffer per-stage 窗口同形：都是**声明条数**而不是 extent，所以是 `u32` 不是 `u64`） |
| 缺段读法 | 旧读法：列表界管整张列表，13 条继续按名拒，句子逐字不变 |

缺段之所以是**更严**的一侧：一个还没声明这条轴的 provider，它自己的 `max_render_textures`
就是它当年的整条规则（Vulkan rail 当年是 8），所以"位缺席 ⇒ 按旧读法拒"与"它当年会拒"
是同一件事。

两个载荷守卫都要带上这个位（响应级 frame 标签守卫 + `put_capabilities` 里 heap/ICB 半边的
守卫）：只加一处会让"只声明这个窗口"的快照静默退回 legacy 载荷，段就掉在 wire 上。

## 两条 rail 的执行语义

- Vulkan：`resolve_render_textures` 复述契约的两条规则（直构请求的防御臂），
  `create_render_textures` 再用设备自己的窗口判一次——canonical arrangement 把一个 pass 的
  采样纹理放在同一个 set（set 0，与 stage buffer 共用），所以检查是 per set + per stage，
  越界按名拒（`render_texture_limit`，带 `set`/`stage`/`requested`/`maximum`）；
- translated 臂的槽位来自模块自己的反射（13 个自变量 ⇒ 13 个 combined image sampler），
  descriptor set layout 与 pool 尺寸都从这条计划读出，所以窗口内没有任何隐藏的 8 槽上限；
- native：reviewed MSL 模块只读一个纹理自变量，`render_texture_stage_unsupported` 是它真正
  的执行窗口；声明块里不声明 per-stage 窗口，13 条由 core admission 按名拒。

## 证据与可复跑命令

| 面 | 位置 |
|---|---|
| 13 条声明的执行读数（Lavapipe） | `crates/metal-api-vulkan/tests/render_texture_count_e2e.rs`（`5b 01 0d ff`，两个换纹理读数 `53 01 0d ff` / `4e 01 00 ff`） |
| fixture | `crates/metal-api-vulkan/tests/fixtures/render_sample_thirteen_textures.frag.ll`（13 个 `[[texture(n)]]` + 13 个 AIR constexpr sampler state） |
| 设备窗口的纯函数读数 | `crates/metal-api-vulkan/src/provider.rs` 的单测（平台下限 16/32、Lavapipe 16/32、更窄设备 9/18） |
| 契约三态（形状 / 旧读法 / 设备窗口） | `crates/metal-api-core/src/provider.rs::a_per_stage_texture_window_admits_the_census_shape_and_narrows_it` |
| wire 形状 | `crates/metal-api-ipc/src/command.rs::the_per_stage_texture_window_travels_in_its_own_extended_block`（`00 0f <u32 BE>` 是家族最后一段、缺段读 0、逐字节重编） |
| conformance | 见"留给主代理"一节 |

```bash
cd /home/hiliang/hackintosh/metal-api-emulator/worktrees/metal-texture-count
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  cargo test -p metal-api-vulkan --locked --test render_texture_count_e2e -- --nocapture
cargo test -p metal-api-vulkan --locked --target x86_64-pc-windows-gnu \
  --test render_texture_count_e2e --no-run   # 再用 powershell.exe 跑那个 .exe（RTX 5060）
```

## 一个诚实的边界

新 frame 的 `00 0f` 段若被**旧解码器**读到，会以 `UnknownCapabilityTail(0x0f)` 整帧拒绝
（`decode_capability_extended_tail` 的白名单 + 严格升序检查），不是静默忽略；方向是
fail-closed，与 E-SB2 的 `0x07`、E-SB3 的 `0x0c` 同一性质。

另一条边界属于 fixture：13 个自变量各带一个 AIR constexpr sampler state（13 个同值
global）。rail 把模块的静态 sampler 与纹理声明**按位置**配对（C1b 的规则：一个采样纹理读
一个 AIR 静态 sampler），所以"13 个自变量共用一个 state"的模块不是这条 reviewed 形状；
fixture 里 13 个 state 的写法就是前端对每个自变量各写一份 state 的形状。
