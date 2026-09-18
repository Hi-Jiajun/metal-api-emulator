# 颜色附件的「落地窗」声明面（E-TX13）

本文件记录一次 provider 增量：**一个颜色附件的 load 源与 store 落地点可以是两份不同的
view 声明**。它是 `research/docs/23` §115（E-TX8，`StoreOp::Borrowed`）的 store 侧续作，
对应 census v31/v32 `guest_backing` 桶里 A 家族（683 行/轮）那一类记录。

## 1. 缺口

`StoreOp::Borrowed` 的窗口是附件**自己**的 view 声明：pass 从哪读、帧落到哪，两者是同一个
`(allocation, view)`。这在 A 家族上不成立：

* 记录的 **load 源是 exec walk 交过来的链值**（调用者字节），而它 guest 页里躺的是 packet
  之前的帧；
* 它需要把这一拍的帧**落回 guest 页**（链路后面的 scanout / 前台要读那里）。

一个附件一份声明时，两者只能二选一：声明窗口 = 把 pass 的地基从链值换成旧页（bucket、
canonical、landing 计数全部正常，像素来自错帧）；声明调用者字节 = 帧落不到 guest 页。
R 侧 B3（reims `0488e78`）已经把 admission 臂接好、等生产端，缺的正是这条声明面。

## 2. 契约

```rust
pub struct AttachmentLandingView {   // 第二份 view 声明的身份
    pub allocation_id: AllocationId,
    pub view_id: ViewId,
}

pub enum StoreOp {
    Store,
    Borrowed,                        // E-TX8：窗口 = 附件自己的声明
    BorrowedLanding(AttachmentLandingView), // E-TX13：窗口 = 这份声明
    Resident,
    DontCare,
}
```

* **只在 store 侧生效**：`load`（`Load`/`Clear`/`DontCare`/`Resident`）的语义一行不改，
  `Load` 仍然从附件自己的 view 声明取前序字节。
* **身份规则**：landing view 的两个 id 必须非零；且**不得等于附件自己的
  `(allocation, view)`**——那正是 `Borrowed` 已经说过的同一件事，两种拼法会让 rail 有两个
  地方解析同一个窗口（`AttachmentLandingViewSameIdentity`）。
* **窗口规则**照 E-TX8：只接受 owner 的注册窗口（`BufferSource::BorrowedNoCopy` 单窗口，或
  `BufferSource::GuestRuns` 拼接），窗口字节总量必须等于附件的紧密字节数
  （`RenderAttachment::expected_bytes()`），拷贝臂与 extent 不符各自按名拒。
* **writeback 通道照旧**：`RenderAttachment::publishes_bytes()` 对该臂为真，所以 trace 自己的
  production 与调用方的比对读到的还是同一份帧，既有读者逐字节不变。

谓词：`StoreOp::lands_in_owner_window()`、`StoreOp::landing_view()`、
`RenderAttachment::landing_identity()`（两条 owner-window 臂各自的答案）。

## 3. wire

* **store tag `4`** + 16 字节载荷（landing view 的 `view_id`、`allocation_id`，与附件自身
  字段同序）。旧解码器读到 tag `4` 就是
  `UnknownEnumValue{field:"attachment store op"}`——`research/docs/23` §115.4 早已把 `4`
  登记为未知 tag，所以这一臂天然 fail-closed，而不是靠一段尾部可选段（那会被旧解码器读成
  下一个附件的字段）。
* **能力位** `ProviderCapabilities::supports_render_attachment_landing_view`（默认 `false`）
  + 谓词 `declares_render_attachment_landing_view_support()`，在 capability tail 的第二
  tag 族里以 `0x00 0x05 <bool>` 编码（E-TX12 的 `0x00 0x04` 之后）。它与
  `supports_render_texture_gathered_extent*` 两个位不是同一个轴：那两个位说的是**被采样
  源**的 extent，本位说的是**附件帧的落地点**，因此各自成位、互不代读。

## 4. 两条 rail

| rail | 行为 |
|---|---|
| Vulkan（trace 轨） | 执行：`compute_provider.rs` 按 store 臂携带的身份在同一份 serial view 列表里解析第二份声明，`render.rs` 用 E-TX8 的 retain/retire 落地把帧写进那个窗口 |
| Vulkan（present 轨） | 按名拒（`render_present_borrowed_store_unsupported`）：present target 是 provider 自己的图，"一帧两个家" |
| native | 按名拒（`render_attachment_landing_unsupported` + `source=landing_view`）：该轨的 owner 窗口是**输入**通道，没有写回路由；快照保持该位 `false` |
| 对象 API | 本增量没有落地点入口（§115.5.4 的边界），对象轨只把新臂归到按名拒一侧 |

Vulkan 侧的失败面（每条一个名字）：`landing_view_undeclared`（trace 没声明那份 view）、
`owned_bytes` / `staged_lease`（第二份声明的 source 是拷贝臂）、
`render_attachment_landing_mismatch`（窗口拼接 ≠ 附件 extent）、`resident_target`
（与 resident store 并置）。

## 5. 证据

* **Rust e2e**（Lavapipe）：`crates/metal-api-vulkan/tests/render_attachment_landing_view_e2e.rs`
  ——7 条用例，读数：
  * 帧 `4080c0fffefefefe4080c0fffefefefe`：左列是片元输出，右列是**调用者字节**；
  * owner 窗口 pass 后逐字节等于该帧（窗口原字节是 `11223344…`，所以"没落地"或"读错了源"都能被抓）；
  * 只换窗口原字节 ⇒ 帧不动；只换调用者字节 ⇒ 帧变成
    `4080c0ff010203044080c0ff01020304`；
  * `Borrowed` 臂在同一 fixture 上保持自己的 `owned_bytes` 拒答；未声明 / 拷贝臂 / extent 不符
    三条按名拒；`GuestRuns` 列表臂按声明顺序落进两个 run。
* **conformance**：`conformance/suite-v40.json` 一条 fixture（load = 调用者字节、land = owner
  窗口），declaring pass 用 reviewed witness kernel 的第三个绑定声明窗口；捕获新增
  `landing` 观测（窗口字节），comparator 校验它等于声明的帧。
* **门**：`cargo fmt`、`tools/gates-local.sh`（`GATES_OK`）、`tools/lavapipe-smoke.sh`
  （suites/captures 只增不减）、`openspec validate --strict`、RTX 5060 读数。

## 6. 边界

1. **对象 API 无落地点入口**：本增量的读数在 trace 轨与 conformance 的 Vulkan trace rail；
   对象 rail 的入口是后续增量（§115.5.4）。
2. **padded 窗口与跨 registration 的 run 列表**仍是类外（`render-attachment-window-stride` /
   `render-guest-run-multi-allocation` 两个既定 change）。
3. **native / Apple 无读数**：该轨按名拒，快照位保持 `false`；Apple 侧没有该形状的 oracle。
4. **不做真机 census**：本增量不宣称 `guest_backing` 已收窄；预期读数与 R 侧随动写在交付
   报告里。
5. **不等于完整 Metal conformance**：本文件只说明这一条 store 臂在契约、wire、Vulkan 执行
   与 conformance fixture 上一致且可证伪。
