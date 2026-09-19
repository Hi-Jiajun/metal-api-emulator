# stage buffer 的"整段绑定"footprint（E-SB3）

本文记录本仓一个增量：`FootprintProof` 新增第三条**可执行**的臂
`BindingRange` —— 翻译没有述说 reach 时，provider 按**调用方自己绑定的那一段**执行，
契约不发布任何字节上限。叙事全貌在 `research/docs/23` §3.3；这篇只写 E 仓自己的契约、
边界与可复跑的验证命令。

## 形状与今天的失败面

air-reach 侦察（`.agents/tasks/root/e_stage_buffer_unbounded_reach-report.md`）把 census v46
的 58 行 `render_provider_out_of_class_stage_buffer_footprint` 钉到一条 pipeline
（`fixed_vert_lpf_gen` × `fixed_frag_lpf_cpf`），并把根因钉到 translator 的
`site=offset-none`：`[[buffer(7)]]` / `[[buffer(8)]]` 的**索引值本身不可述**
（浮点→half→位域截断的数据通路），不是越界、不是记录超限、也不是指针逃逸。

translator 自己对这件事的说法就在反射里
（`metal2vulkan::reflect::BufferFootprint`）：

> A true unbounded flag means at least one reachable dereference could not be expressed by
> this schema; **the complete caller-provided buffer window must then remain available.**

即"**整段调用方给的窗口要留着**"。在此之前契约对这句话只有一个落点：
`FootprintProof::Unbounded` —— 按名拒绝。于是这条 pipeline 的 29 条 draw 全部停在引擎上。

## 契约面

```rust
// crates/metal-api-core/src/provider.rs
pub enum FootprintProof {
    Static { max_bytes: u64 },
    Affine { accesses: Vec<AffineAccess> },
    BindingRange,   // 新：没有述说 reach，按 pass 绑定的整段字节执行
    Unbounded,      // 不变：没有证明就不执行，句子/slug 逐字不动
}
```

三条不变量：

1. **published 的事实不多一个**。范围内访问两边同字节；范围外访问 Metal 对越过绑定长度的
   访问本来就是未定义（不保证值、也不保证不崩），Vulkan 侧只有 `robustBufferAccess`
   打开时才是"clamp 读 / 丢弃写、不发散"。所以新臂发布的是"我按你声明的整段执行，
   越界依旧未定义"，比 Metal 不多说一个字。
2. **不是 `Unbounded` 的放宽**。`Unbounded` 的 slug
   （`render_stage_buffer_footprint_unsupported`）、句子与第一增量完全相同；新臂只是给它
   一个**新的、可执行的**落点。空绑定照旧 `MissingStageBufferBinding`，access 照旧逐字段
   对齐（`StageBufferAccessMismatch`）——被放开的只有"没有上限可比较"这一件事。
3. **整段绑定**。`BufferView::validate_shape` 早已把 trace 自带的源与 guest run list 钉到
   视图自己的长度，Vulkan rail 的 `resolve_stage_buffers` 把每个 binding 解析成 bind
   自己的窗口，descriptor 写的是 `vk::WHOLE_SIZE`；所以"pass 绑定的那段"在两侧是同一个
   事实，`BindingRange` 下**不做** `required <= view.length` 比较（没有 required）。

## 能力位与 wire

| 面 | 内容 |
|---|---|
| `ProviderCapabilities` | 新字段 `supports_render_stage_buffer_binding_range` + 谓词 `declares_render_stage_buffer_binding_range()` |
| capability frame | 第二 tag 族新增 `0x00 0x0c <bool>`；旧 frame 读作 `false`（fail-closed），未知家族 tag 仍 `UnknownCapabilityTail` |
| footprint wire | `put_footprint` / `get_footprint` 新增 code `3`（单字节、无 payload）；code `2`（`Unbounded`）语义不变 |
| Vulkan rail | 设备报 `robustBufferAccess` **且** 建 device 时已开启 → 发布该位；否则注册按名拒（`render_stage_buffer_binding_range_unsupported`） |
| native rail | 位保持 `false`，注册按名拒（同 slug、自己的句子） |

wire 的方向性：旧解码器遇到 `0x0c` 会以 `UnknownCapabilityTail` 拒整帧，而不是把它读成
别的段；不带该段的旧 frame 在新解码器里读作 `false`。两个方向都是 fail-closed。

## 设备事实

`robustBufferAccess` 是 Vulkan 1.0 的 core feature，本机两条 rail 的读数：

| 设备 | `robustBufferAccess` | 说明 |
|---|---|---|
| Lavapipe（WSL，`lvp_icd.json`） | 报 `true`，建 device 时开启 | 本增量 e2e 的执行面 |
| NVIDIA RTX 5060（Windows 宿主驱动） | 见 `evidence/sb-binding-range-*-2026-09-20/` | 与 census 主机同一驱动路径 |

`capabilities_from_limits` 本身**不**声明该位（limit 说不出 feature），
`VulkanExecutor::provider_capabilities()` 用设备读数覆盖它，和 depth/stencil resolve 两个
位的写法同形。

## 两条 rail 的执行语义

- **Vulkan**：翻译模块的反射说"没有任何 reach"（`has_unbounded_access`，或既无 static
  range 又无 strided access）时，契约的 `BindingRange` 与它配对；pass 绑定的视图整段
  `WHOLE_SIZE` 绑进 descriptor，越界访问落在 `robustBufferAccess` 上。反射**说出了**界
  （static/affine）而声明写 `BindingRange` 时按 `render_stage_reflection_mismatch` 拒——
  这一臂配对的是"两边都没说"。
- **native**：reviewed 模块按自己 pinned 源码的 extent 读每个 `[[buffer(N)]]`，没有执行
  "没有述说 reach 的声明"的路线，所以注册按名拒，与它能力位保持 `false` 一致。
- **compute 面**：`FootprintProof` 是共用的枚举，但声明这条臂的能力位是 render 位，
  所以 compute 声明按名拒（`buffer_footprint_binding_range_unsupported`）——
  `Unbounded` 的 `buffer_footprint_unbounded` 不变。

## 证据与命令

```sh
# rail 级 e2e（fixture：索引来自内存的 [[buffer(1)]]）
cargo test -p metal-api-vulkan --test render_stage_buffer_binding_range_e2e -- --nocapture
# → selector 0/2 与改表三组读数，见 tests/render_stage_buffer_binding_range_e2e.rs

# 契约与 wire
cargo test -p metal-api-core --lib whole_binding
cargo test -p metal-api-ipc --lib whole_binding
cargo test -p metal-api-native --lib stage_buffer_registration

# 全量门
REPO=$PWD bash /home/hiliang/hackintosh/tools/gates-local.sh
REPO=$PWD bash /home/hiliang/hackintosh/tools/lavapipe-smoke.sh
```

本增量的证据归档在 `evidence/sb-binding-range-<sha>-2026-09-20/`（门日志、e2e 输出、
能力帧字节断言、`sha256.txt`）。
