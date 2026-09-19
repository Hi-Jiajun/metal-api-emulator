# 同 pass 自采样的 canonical 臂：pass 入口快照（E-TX15）

本文件记录一次 provider 增量：**一条 draw 采样它自己正在写的附件**时，canonical 契约
给出一条**显式声明臂**，语义写死为"读 = 该附件在 pass 入口那一刻的字节"，Vulkan 轨用
**device-side 镜像拷贝**在 render pass 打开之前取这份快照。它对应 census 的冻结边界
`render_provider_out_of_class_texture_source_order`（v43：5 847/轮）与 engine 自己的
fallback 臂——`crates/reims-vgpu/src/backend/vulkan/engine/exec.rs`：
"capture the prior resident content into a same-format GPU image before changing the
attachment"。

## 1. 缺口

engine 对这条形状有两个臂（`exec.rs:4019-4048`）：设备带
`VK_EXT_attachment_feedback_loop_layout` 且视图是 plain 2D identity 时走 **feedback**
（pass 内读活帧，竞态区由 spec 的 data-race 定义裁定为未定义），否则走 **snapshot
fallback**（pass 打开前把 resident 内容拷进同格式图像）。canonical 侧今天连**声明面**
都没有：`RenderPassDescriptor::validate`（`provider.rs:4832-4840` 一带）对"采样的 view
与附件同身份"**无条件**按 `RenderTextureAttachmentConflict` 拒，理由是"rails 的执行次序
只能把这条读表达成竞态"。

而"pass 入口快照"不是竞态：它的生产者只有**附件在 pass 打开时的内容**，与 pass 自己的
任何写都没有次序关系（拷贝发生在 `vkCmdBeginRenderPass` 之前）。这就是本增量给出的
第三个臂。

## 2. 契约

```rust
pub enum TextureSource {
    OwnedBytes(Vec<u8>),
    StagedLease(LeaseId),
    BorrowedNoCopy(LeaseId),
    TraceView,
    PassEntrySnapshot,   // 本增量
}
```

* **身份**：声明必须命名**本 pass 的某个颜色附件**，按 `(allocation_id, view_id)` 整对
  匹配；不匹配是 `RenderPassEntrySnapshotUnattached`。
* **形状**：声明必须**逐字重述**该附件的 format 与 extent（快照就是那块 texel 网格），
  否则 `RenderPassEntrySnapshotShapeMismatch`；视图必须是 plain 单样本 2D
  （`RenderPassEntrySnapshotShapeUnsupported`），这与采样面本身的窗口一致。
* **load 臂**：附件必须**建立**这份先验内容——`LoadOp::Load` 或 `LoadOp::Resident`。
  `Clear` 的入口内容是 clear 色（不是"附件持有过的字节"），`DontCare` 什么也没建立，
  两者按名拒（`RenderPassEntrySnapshotLoadUnsupported`）。
* **其余臂不变**：任何*其它* source 出现在附件自己的 view 上，仍是
  `RenderTextureAttachmentConflict` —— 新臂是唯一能命名"同 pass 读"的声明，它不扩宽
  别的臂。
* **footprint/一致性条款**：读的足迹就是该附件紧密排布的 extent（采样面既有的
  `WholeView` 单位），读的次序由构造保证在 pass 的**所有写之前**（rail 的拷贝在
  `vkCmdBeginRenderPass` 前），因此这个臂不可能隐藏 same-pass read-after-write。
* **admission**：新能力位 `ProviderCapabilities::supports_render_pass_entry_snapshot`
  （默认 `false`）；快照没声明它时，带该声明的 pass 在 admission 按名拒
  （`render_pass_entry_snapshot_unsupported`）。

## 3. wire

* **texture source tag `4`**（TraceView 的 `3` 之后）：和 trace-produced 臂一样不带
  bytes、不带 lease——view 的 `(allocation, view_id)` 对就是附件的身份。旧解码器读到
  `4` 是 `UnknownEnumValue{field: "texture source"}`，整帧拒，不会把声明读成别的臂。
* **能力位**：capability tail 第二 tag 族 `0x00 0x08 <bool>`（E-SB2 的 `0x00 0x07` 之后）。
  族是闭集：旧解码器读到 `0x08` 是 `UnknownCapabilityTail(0x08)`（typed refusal），
  不会把 bool 读成 `0x06` 的 kept-frame 位。
* 只声明这一位的快照**仍然写扩展载荷**（三处 guard 都加了谓词）。

## 4. 两条 rail

| rail | 行为 |
|---|---|
| Vulkan（trace 轨） | 执行：`resolve_render_textures` 把声明解析成附件的**位置**（四问各自按名拒），`create_attachment` 为该附件加 `TRANSFER_SRC` usage，`create_render_textures` 为采样图像建 device-local `OPTIMAL`（`SAMPLED|TRANSFER_DST`，`UNDEFINED`），`record` 在 render pass **之前**录一条 `vkCmdCopyImage`（attachment → 采样图像）并把采样图像发到 `GENERAL`、把附件恢复成 pass 声明的 `initialLayout`；resident 目标的布局守卫照旧 |
| native | 按名拒（`render_texture_source_unsupported` + `source=pass_entry_snapshot`），快照位保持 `false`（Apple 对该形状没有 oracle） |
| 对象 API | 没有入口：对象轨的纹理输入自带 bytes，而这条声明不带 bytes |

**provenance 读数**：`VulkanContext` 新增 `attachment_snapshots` /
`attachment_snapshot_bytes` 两个计数器，在录拷贝处累加（via
`VulkanExecutor::attachment_snapshot_counts()`）。帧本身分不出"采样图像被附件入口内容
填过"与"采样图像是别的东西"，计数器可以——fixture 因此同时读帧与这对计数。

## 5. 证据

* **Rust e2e**（Lavapipe）：`crates/metal-api-vulkan/tests/render_pass_entry_snapshot_e2e.rs`，
  4 条读数：
  * **Load 臂**：pass 从 caller 的 gradient bytes 打开附件，采样声明走新臂 ⇒ 帧等于
    `f(entry)`（= 字节臂同一份 bytes 落的帧，逐字节相同），且计数 delta = `(1, 64)`；
  * **跟随**：只换入口 bytes（升/降 gradient）⇒ 帧跟着换；
  * **Resident 臂**（census 的形状）：先一个 `StoreOp::Resident` pass 把帧留在 provider
    图像里，再一个 `LoadOp::Resident` + 新臂的 pass 采样它 ⇒ 帧 = kept frame（左半列
    是画的、右半列是 clear），计数 delta 同样 `(1, 64)`；
  * **反例**：不在本 pass 附件上的声明（`render_pass_entry_snapshot_unattached`）、
    重述另一个 extent（`..._shape_mismatch`）、`Clear`/`DontCare` load
    （`..._load_unsupported`）、provider 不持有图像的 resident 臂
    （`resident_target_unavailable`）、以及附件自己 view 上的字节臂仍是
    `trace_contract_invalid`（attachment conflict）。
* **契约 / wire 单测**：core 的四问与 admission 门（位假 ⇒
  `render_pass_entry_snapshot_unsupported`）、ipc 的 source tag `4`（与 `3` 只差一个
  字节）与能力位 `0x00 0x08`（族序、只声明它仍写扩展载荷、缺失读作 false）、native
  的按名拒。
* **conformance 五路 fixture**：`conformance/suite-v43.json`。declaring case 是
  `copy_word` 在 4×4 附件自己的 64 字节 view 上；render case
  `pass_entry_snapshot_4x4` 是**两条翻译 AIR**（`render_offscreen_2x2.vert.ll` +
  `render_sample_texture_2d_nearest_clamp.frag.ll`）——模块的两个固定采样点读入口
  texel (3,0) 与 (1,0) 的红通道 ⇒ 画到的一半落 `c04000ff`，被 scissor 留下的另一半
  保持入口 bytes。frame 因此**一帧两个读数**："快照内容"与"pass 自己的光栅"。
  comparator 另外要求 `attachment_snapshots == 1`、`attachment_snapshot_bytes == 64`
  （"这张快照真的被取过"），并拒绝"帧 = 入口 bytes"或"drawn 颜色等于某个入口 texel"
  的编排。marker 只声明 `vulkan`（native 两轨按名拒、对象轨无入口）。
* **门**：`cargo fmt`、`cargo clippy -D warnings`、`tools/gates-local.sh`（`GATES_OK`）、
  `tools/lavapipe-smoke.sh`（suites/captures 只增）、`openspec validate --strict`、
  RTX 5060 读数按既有做法。

## 6. 边界

1. **Clear / DontCare 的 load 臂按名拒**：本臂声明的是"附件的先验内容"，clear 建立的
   是 clear 色而不是它持有过的字节，`DontCare` 什么都没建立。engine 的 fallback 在
   clear 形状上拷的是 pass 前的 resident 内容，与 canonical 的入口语义不同——这类子形状
   因此**继续按名拒**，不猜。
2. **depth/stencil 附件不在这条臂里**：canonical 的采样面今天只读 8-bit unorm 四条
   lane，depth 的 `D32_SFLOAT` 不是其中一条，所以"采样 depth 附件"没有可达的声明。
   engine 的 fallback 覆盖 depth，这一半留给未来的增量。
3. **native / Apple 无读数**：该轨按名拒，快照位保持 `false`。
4. **对象 API 无入口**：这条声明不带 bytes，对象轨的纹理输入没有对应形状。
5. **不做真机 census 的"桶下降"**：本增量只给 canonical 出口；census 里的桶要下降，需要
   R 侧随动（读 `0x00 0x08`、把 `writes_attachment` 形状以新臂声明、其余保持逐字拒句），
   那份清单见交付报告。本刀的真机轮（tag `b8`）因此预期**桶不变**、红线与四个零读保持 0。
6. **不等于完整 Metal conformance**：一台 Lavapipe 与一次 RTX 读数不替代 Apple 真机，
   本文件只说明这条通道在契约、wire、Vulkan 执行上一致且可证伪。

## 7. conformance 五路 fixture：`suite-v43.json`

* **`conformance/suite-v43.json`**：declaring case `render_declaring_pass_entry_snapshot`
  （`copy_word`，binding 0 = 4×4 附件的 64 字节 view、binding 1 = 4 字节 scratch），
  render case `pass_entry_snapshot_4x4`：`fragment_textures[0].source =
  "pass_entry_snapshot"`（identity 与附件相同、不带 `initial_hex`），attachment
  `load = "load"` + `initial_hex` = 入口 gradient，`scissor = [0,0,2,4]`，
  `expected_hex` = 画到的两列 `c04000ff` + 留下的两列入口 bytes，marker 只有 `vulkan`。
* **harness（`examples/metal-smoke/src/bin/provider-capture.rs`）**：`RenderGeometry`
  新增 `PassEntrySnapshot`（翻译臂，默认 set 0，声明一个 `[[texture(0)]]` 采样槽），
  `FragmentTextureDefinition.source` 取值 `bytes` / `pass_entry_snapshot`（闭集），
  render case 结果新增 `attachment_snapshots` / `attachment_snapshot_bytes` 两字段。
* **comparator（`conformance/compare.py`）**：`RenderExpectation.snapshots` 携带
  `(1, extent*4)`；case 侧校验 identity/format/extent/load/scissor 与两半帧规则，capture
  侧校验这对计数（其余 case 必须不带）。
* **native oracle（`conformance/NativeOracle.swift`）**：登记 `compute-buffer-v43` 并执行
  declaring case；render case 因新臂按名拒（该轨没有 Apple oracle）。版本区间推进到 v43。
* **窄类登记（`conformance/narrow-class.json`）**：render case 落在 render 窄类之外，
  规则 `render-covered-attachment`。
* **守卫（`conformance/test_suite_v43.py`）**：14 条读数——suite 身份、两份源码哈希、
  declaring case 的 view、render case 的声明与两半帧、plan 的 `snapshots` 读数，
  以及一族"改编排必须红"的负例（重定向 identity、改 extent、给声明带 bytes、把 load
  改成 clear、帧 = 入口 bytes、drawn 颜色等于入口 texel、去掉 scissor、marker 越界、
  去掉 `source` 的旧形状）。
* **CI（`.github/workflows/ci.yml`）**：v43 加入四条版本环与五路 capture/compare。
