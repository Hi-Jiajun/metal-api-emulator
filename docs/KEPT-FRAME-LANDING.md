# 延迟 Store 的兑现：kept frame 的落地条目（E-TX14/R4b）

本文件记录一次 provider 增量：**帧已经留在 provider 图像里之后，由一次后续条目把它落进
owner 的注册窗口**。它是 `research/docs/23` §115（E-TX8）与 §115 之后的增量（E-TX13，
`StoreOp::BorrowedLanding`）的**延迟送达**续作，对应 fp3 拆账里两条延迟 Store 的尾记录
（`provider_held_store_gva` 9 464/轮、`provider_held_store_surface` 547/轮）。

## 1. 缺口

E 侧今天的两条落地臂都绑在**同一个 pass 的完成**上：`StoreOp::Borrowed` 从附件自己的 view
声明取窗口，`StoreOp::BorrowedLanding` 从 store 臂携带的第二份声明取窗口。两者说的都是
"这一拍的帧落进那个窗口"。

而 R 的意图是**延迟**：这一拍先把帧留在 provider 图像里（`StoreOp::Resident`，R 已经在用），
真正的读者出现时再搬。契约里没有"后续条目把它搬出去"的形状，于是延迟只能退化成"整包把
字节交回 engine"（census v35 的 `relay_surface` 每窗 19 条、`provider_held_store_*`）。

一处现状决定了形状：`StoreOp::Resident` 与 owner-window store **不能同台**（`render.rs` 的
落地解析在 `resident.is_some()` 时按名拒，理由写着"一个帧两个家"）。所以延迟送达需要的是
**第二个条目**，而不是第二个 store 臂。

## 2. 契约

```rust
/// provider 留在自己图像里的那一帧的身份与形状。
pub struct KeptFrame {
    pub allocation_id: AllocationId,
    pub view_id: ViewId,
    pub format: AttachmentFormat,
    pub width: u64,
    pub height: u64,
}

/// 一次只做落地、不重画的提交条目。
pub struct KeptFrameLanding {
    pub frame: KeptFrame,
    pub landing: AttachmentLandingView,   // E-TX13 的同一个类型
}

pub enum TracePass {
    Compute(ComputePass),
    Render(RenderPassDescriptor),
    Landing(KeptFrameLanding),            // 本增量
}
```

* **条目里没有可重画的东西**：没有 pipeline、没有 draw、没有 load 源、没有 store 决策。
  "这次提交只落地"因此是类型层的事实，而不是执行器要遵守的约定。
* **形状规则**：四个身份 id 各自非零、extent 非零、format 是颜色附件格式、字节数不溢出
  （与前缀同一个 `InvalidIdentity` / `ZeroDimension` / `UnsupportedAttachmentFormat` /
  `ArithmeticOverflow` 词表）。`frame` 与 `landing` **可以**是同一个 `(allocation, view)`：
  那是"把留在 provider 里的帧落回它自己声明的窗口"，与 E-TX13 那条"第二份声明不得等于附件
  自己"的规则不同——那条拒的是"同一份声明既当 load 源又当落点"，本条目没有 load 源。
* **消费语义**：一次成功落地**消费**这个身份，之后同一身份的第二次落地按名拒
  （`kept_frame_already_landed`），直到一次新的、已完成的 `StoreOp::Resident` pass 把它重新
  留下（该 pass 完成时清消费位）。这是"不许把旧帧当权威写第二次"的可执行形式。
* **窗口规则**照 E-TX8/E-TX13：只接受 owner 的注册窗口（`BufferSource::BorrowedNoCopy`
  单窗口，或 `BufferSource::GuestRuns` 拼接），窗口字节总量必须等于 kept frame 的紧密字节数，
  拷贝臂与 extent 不符各自按名拒。
* **条目永远不独自出现**：窗口的声明是 **compute pass 的视图**（`serial_resources()` 只收
  compute pass 的视图 + 附件身份），所以带 landing 条目的 trace 一定带着那条声明 pass，它的
  pipeline 表因此非空且每条都被使用。只带 landing 条目的 trace 会被契约按
  `EmptyPipelineTable` 拒——这不是额外规则，而是声明通道的直接后果。

## 3. wire

* **pass kind tag `0x19`** + 定长载荷：`frame.view_id`、`frame.allocation_id`、`frame.format`、
  `frame.width`、`frame.height`、`landing.view_id`、`landing.allocation_id`。旧解码器读到
  `0x19` 就是 `UnknownPassTag(0x19)`——本仓既有的未知 tag 行为逐字相同，**不会跳过载荷**去
  读后面的条目。
* 带 landing 条目的 trace 进入 tagged 布局（`put_trace` 的 `tagged` 判定把
  `has_landing_entries()` 算进去）；不带它的 trace 字节逐字不变。
* **能力位** `ProviderCapabilities::supports_render_kept_frame_landing`（默认 `false`）+
  谓词 `declares_render_kept_frame_landing_support()`，在 capability tail 的第二 tag 族里以
  `0x00 0x06 <bool>` 编码（E-TX13 的 `0x00 0x05` 之后）。**不扩** E-TX13 的位：那一位说的是
  "一次 pass 的帧落到哪份声明"（当拍），本位说的是"一条新条目形状的准入 + 跨提交的 kept
  图像"，一个 rail 可以执行前者而没有任何 kept 注册表，两件事各有独立真假。

## 4. 两条 rail

| rail | 行为 |
|---|---|
| Vulkan（trace 轨） | 执行：`compute_provider.rs` 从 resident 注册表解析 kept 身份（未持有 / 墓碑三条 / 未定义 / 形状变了 / 已被消费，各按名拒），从 trace 的 serial view 列表解析落点声明，`render.rs` 用布局守卫 + `vkCmdCopyImageToBuffer` 把 provider 图像读回、用 E-TX8 的同一段窗口代码写进 owner 页，成功时消费该身份 |
| native | 按名拒（`kept_frame_landing_unsupported` + `source=native_rail`）：该轨有 resident 注册表，但 owner 窗口是**输入**通道，没有写回路由；快照保持该位 `false` |
| 对象 API | 本增量没有 landing 条目的入口（trace 轨的形状），对象轨的读数因此不含这条 case |

按名拒的名字表（一个事实一个名字）：

| 事实 | slug |
|---|---|
| 注册表没有这个身份、也没有墓碑 | `kept_frame_not_held` |
| 身份被 LRU 预算退役 | `kept_frame_evicted` |
| 身份的 lease 被释放 | `kept_frame_released` |
| 设备 epoch 前进（整表作废） | `kept_frame_stale` |
| 图像存在但没有已完成的 pass 定义过它 | `kept_frame_undefined` |
| 同一身份的形状与条目声明不符 | `kept_frame_shape_changed` |
| 身份已被一次成功落地消费 | `kept_frame_already_landed` |
| 落点身份在 trace 里没有声明 | `kept_frame_landing_undeclared` |
| 落点声明是拷贝臂 / 该 rail 不执行条目 | `kept_frame_landing_unsupported`（带 `source`） |
| 窗口字节数 ≠ kept frame 的紧密 extent | `kept_frame_landing_mismatch` |
| 快照没有声明这个能力位（admission） | `kept_frame_landing_unsupported` |

## 5. 证据

* **Rust e2e**（Lavapipe）：
  `crates/metal-api-vulkan/tests/render_kept_frame_landing_e2e.rs`，11 条读数：
  * 正向：一个提交里"先留帧的 pass + landing 条目"，owner 窗口逐字节等于该 pass 的帧
    （左列是片元输出、右列是 pass 的 clear，窗口原字节是 `11223344…`）；事后一次
    `LoadOp::Resident` 读回证明**条目没有重画**、kept 图像仍是那一帧；
  * 条目不为帧发布 writeback（帧的新家是 owner 页）；
  * 反例：同一身份第二次落地（`kept_frame_already_landed`）、条目在留帧 pass 之前 /
    身份从未被留（`kept_frame_not_held`）、身份被预算挤出（`kept_frame_evicted` +
    `retired_by`）、落点未声明（`kept_frame_landing_undeclared`）、落点是拷贝臂
    （`kept_frame_landing_unsupported` + `source=owned_bytes`）、窗口长度不符
    （`kept_frame_landing_mismatch`）、快照位假（admission 的
    `kept_frame_landing_unsupported`）；
  * 只换窗口原字节 ⇒ 帧不动（条目是写方向）；`GuestRuns` 列表臂落进同一窗口。
* **契约 / wire 单测**：core 的形状规则与 admission 门、ipc 的 tag `0x19` 载荷、能力位
  `0x00 0x06`（含"只声明它仍写扩展载荷"与未知 tag 的闭集读数）、native 的按名拒。
* **conformance 五路 fixture**：`conformance/suite-v41.json`（见 §7）。render case
  `kept_frame_landing_quad_2x2` 让这一个 pass 把帧留在 provider 图像里
  （`store: "resident"`），`kept_frame_landing` 一节同时点出 kept 身份与 owner 窗口，
  `expected_landing_hex` 是被观察的那份帧。它在 Vulkan trace 轨上跑通：窗口读到
  `4080c0fffefefefe4080c0fffefefe`（左列片元输出、右列是 pass 的 load 字节），
  `writebacks`/`allocations` 都是空的，而 `copy_out` 比 v40 的当拍臂少一次发布的
  回读、多一次条目自己的回读——"pass 没发布"与"条目确实送达"因此是两个可分辨的读数。
* **门**：`cargo fmt`、`cargo clippy -D warnings`、`tools/gates-local.sh`（`GATES_OK`）、
  `tools/lavapipe-smoke.sh`（suites/captures 计数不减）、`openspec validate --strict`、
  RTX 5060 读数。

## 6. 边界

1. **conformance 五路 fixture 已落地，但只有 Vulkan trace 轨执行它**：见 §7。这条形状
   的"五路"里，两条 native 轨按名拒（不持帧、也没有写 owner 窗口的路由）、两条对象轨
   无入口，所以它的 `capture_rails` 只声明 `vulkan`——**这条 fixture 证不了 native /
   Apple 侧的行为**，它证的是"契约 + wire + Vulkan 执行 + comparator 的期望"这一串。
2. **对象 API 无入口**：landing 条目是 trace 形状，对象轨没有对应入口（与 E-TX13 §115.5.4
   同一处边界）。
3. **不做真机 census**：本增量不宣称 `relay_surface` 已收窄；预期读数与 R 侧随动写在交付
   报告里。
4. **padded 窗口、跨 registration run 列表、GVA 窗口**仍是类外（两条既定 change）。
5. **native / Apple 无读数**：该轨按名拒，快照位保持 `false`。
6. **不等于完整 Metal conformance**：一台 Lavapipe 与一次 Windows 真机读数不替代 Apple
   真机，本文件只说明这一条通道在契约、wire、Vulkan 执行上一致且可证伪。

## 7. conformance 五路 fixture：`suite-v41.json`

这条形状在 harness 里需要的是一串互相咬合的登记，本文件记录它们各自钉住了什么：

* **`conformance/suite-v41.json`**：declaring case 复用 v40 的形状（`render_declaring_landing_view`
  ——同一个 witness kernel、附件自己的 16 字节 view、copy landing，以及它第三绑定声明的
  owner 窗口 `borrowed_no_copy`），render case 是 `kept_frame_landing_quad_2x2`：
  `attachment.store = "resident"`（**没有** `expected_hex`）、一节
  `kept_frame_landing {frame:{900,910}, landing:{940,950}}`、以及唯一的那份期望
  `expected_landing_hex = 4080c0fffefefefe4080c0fffefefefe`。marker 只声明 `["vulkan"]`。
* **harness（`examples/metal-smoke/src/bin/provider-capture.rs`）**：`RenderCase` 新增
  `kept_frame_landing` 一节与校验（kept 身份必须是 resident 附件自己的、窗口仍按
  `borrowed_no_copy` + 附件 extent 解析）；trace 组装处在 render pass **之后**追加
  `TracePass::Landing`（窗口声明来自 declaring compute pass，所以条目永不独自出现）；
  `render_case_landing()` 对两条落地臂统一解析，落地窗口的读回与
  `expected_landing_hex` 比对整段复用 E-TX13 的实现；expectation 规则仍按 resident 臂
  用 `expected_landing_hex` 那份字节做逐 texel 校验（"load 混合了 drawn 与 kept texel"
  一类规则因此原样生效）；写回观测循环跳过 resident store（它不发布 writeback）。
* **comparator（`conformance/compare.py`）**：resident 臂的 `writes`/`allocations` 为空、
  `written` 含 kept 帧自己的 allocation（条目送达时的那一次设备回读正是 `copy_out` 的
  来源），落地窗口规则与 §3.1 共用一段实现；并新增"期望不得等于窗口运行前的字节"这条
  证伪规则——在没有 writeback 可比对时，它是"条目根本没跑"的唯一落点。
* **native oracle（`conformance/NativeOracle.swift`）**：`compute-buffer-v41` 登记
  declaring case 并执行它；render case 因 `kept_frame_landing`/`resident` 按名拒
  （该轨不持帧、也没有写 owner 窗口的路由），CI 上这一轨因此只报 declaring case。
* **窄类登记（`conformance/narrow-class.json`）**：这条 case 落在 render 窄类之外，
  规则 `render-covered-attachment`。
* **守卫（`conformance/test_suite_v41.py`）**：钉住 suite 身份、源码哈希、declaring
  pass 形状、render case 的三处关键字段与计划读数，以及一族"改编排必须红"的负例
  （缺 `kept_frame_landing`、有 section 没有 resident store、resident 臂旁出现
  `expected_hex`/texel rule、kept 身份不是附件自己、窗口未声明/是拷贝臂/extent 不符、
  期望等于窗口原字节）。
