# 没有 index buffer 的那条 draw 臂：受审覆盖面（E，v39）

本文记录本仓一个增量：契约里 `RenderPassDescriptor::indices == None` 的那条 draw 臂（顶点
`0..vertices`、没有 index buffer）怎么从"从未被五路观测过的形状"变成 render narrow class 的
**受审覆盖面**。叙事全貌在 `research/docs/23` §3.3 的交接条目里；这篇只写 E 仓自己的覆盖面、
边界与可复跑的验证命令。**本增量不改 provider、不改 wire、不新增 capability 位。**

## 形状与今天的失败面

gate-3 census v26（reims `4bd90c1` × E `9fbbfab`，`evidence/gate3-census-v26-2026-09-18/`）里
`render_provider_out_of_class_nonindexed` 是 **447 条首败**（13,067 条的 3.4%）、**16 种形状**，
16/16 行的 `attrs` 是 2/3/4、`store=1`、`skip=resident`：单颜色附件、三角形、其余每一条都已进类、
只差一个 index buffer。v27b（`evidence/gate3-census-v27b-2026-09-18/`）同桶 520 条 / 8 形状。

R 侧已经在 `07b68ca`（R39，证据 `evidence/nonindexed-07b68ca-2026-09-18/`）把这条臂放行，并补了
fail-closed 的 `render_provider_out_of_class_vertex_span` 门：类内失败在 R 侧是**打字化 decline**
而不是回退 engine，所以"顶点流短于 `vertices * stride`"必须在 R 侧就被按名挡住。

E 侧要改的不是代码：

| 面 | 现状 | 位置 |
|---|---|---|
| 契约 | `indices: Option<IndexBufferBinding>`，`None` 读作顶点 `0..vertices` | `crates/metal-api-core/src/provider.rs` |
| Vulkan rail | `DrawShape::Vertices { vertex_count }` 选臂并 `cmd_draw` | `crates/metal-api-vulkan/src/render.rs` |
| native rail | `plan_vertex_input` 的 `None` 臂 + `draw_primitives` | `crates/metal-api-native/src/render.rs` |
| wire | `has_vertex_input` 判 `vertex_buffers` 非空或 `indices` 存在，非索引写 `0` | `crates/metal-api-ipc/src/command_codec.rs` |

两条 rail 的覆盖率证明对非索引臂**更严**：索引臂只要求覆盖 `base_vertex + 最高索引 + 1`，非索引臂
要求每条 per-vertex 流覆盖整个 `vertices * stride`（Vulkan slug
`render_vertex_buffer_footprint_unsupported`，native slug `render_vertex_footprint_unsupported`）。

真正的缺口是**覆盖面声明**：`conformance/narrow_class.py` 的 `render-covered-draw` 与
`conformance/narrow-class.json` 至今把这个形状登记为类外，也没有任何"带顶点流的非索引"fixture，
于是这族 draw 的帧从未被五路 parity 观测过。

## 为什么不需要新 capability 位

位是用来让消费方区分"这个 snapshot 会不会执行某形状"的。这条臂不需要区分：契约一直是
`Option`，两条 rail 一直执行它，wire 一直用既有的 `0` 标记承载它，reims 的 R39 也已经在既有字段
（`req.indexed.is_none()`）上放行。缺的只是 E 侧"被声明、被五路观测"的覆盖面。新增一个位会造出
一个不存在的分歧面，并迫使 reims 再读一次位；因此本增量**不写 capability frame 的任何字节**，
也不改任何既有 slug、字段名与句子。

## 覆盖面的规则（`render-covered-draw` 的第二条臂）

```text
非索引 draw 被 covered 的条件：
  - 声明 vertices；
  - base_vertex == 0（契约没有"非索引起始顶点"，BaseVertexRequiresIndices 保持原样）；
  - 没有 vertex_layout 时：vertices == 3（milestone 的 vertex_id 三角）；
  - 有 layout 时：vertices >= 3，且每条 per-vertex 流覆盖 vertices * stride；
  - instance_count == 1、attachment/load/store 等其它 covered 规则不变。
```

`vertices >= 3` 的下限来自语义：三角形列表下 `vertices` 就是 `floor(vertices / 3)` 条三角形，
少于三个顶点的 draw 一条都画不出，落出的必然是纯 Clear 帧——那是"没有执行"的读数，不是 fixture。

## fixture（`conformance/suite-v39.json`，`compute-buffer-v39`）

一个 declaring compute pass + 四个 render case，四者都登记五条 rail：

| case | 形状 | 可证伪的声明 |
|---|---|---|
| `offscreen_triangle_clear_2x2` | milestone 的 `vertex_id` 三角（无流、无索引） | 加宽后它从 refusal 变成 covered 邻居 |
| `quad_indexed_clear_2x2` | 索引臂（v16 的原样副本） | 索引臂仍在同一张 suite 里可观测 |
| `nonindexed_quad_clear_2x2` | 把索引展开成六条记录的**非索引** draw | 帧 MUST 与索引臂**逐字节相同**——这就是 class 声明的 parity |
| `nonindexed_triangle_clear_2x2` | 同一 layout 上只画一条自己的三角形 | 部分覆盖；帧 MUST 与 quad 不同（顶点字节决定帧），在 `narrow-class.json` 里按 `render-covered-state` 登记为类外邻居 |

后两个 case 的顶点流是**推导**而不是拷贝：`conformance/test_suite_v39.py` 自己按索引展开
`quad_indexed_clear_2x2` 的四条记录，再与 `nonindexed_quad_clear_2x2` 的字节逐字节比对；邻居的
三角形则按"落在某一个 texel 的格子里"选点，避免对角线正好穿过 texel 中心的填充规则平局。

## 三条 harness 面

reviewed 顶点输入形状在三个地方各是一张闭集，必须同时加宽，否则"一条 rail 拒绝、另一条执行"的
suite 会被自己的门挡下：

| 面 | 改动 |
|---|---|
| `conformance/compare.py::_vertex_input_declaration` | 非索引 arm：`_integer(vertices)`、`vertices >= 3`、流覆盖 `vertices * stride`；case 的 `vertices` 与 draw 自己的计数比对（索引臂读索引数） |
| `conformance/NativeOracle.swift::validateRenderCase` | 新的 `(layout?, bindings?, nil)` arm：同一 `reviewedBuffers` 校验、`base_vertex == 0`、`vertices >= 3`、每条 per-vertex 流覆盖 `vertices * stride`；对 depth/stencil/cull/blend/per-instance **fail closed** |
| `examples/metal-smoke/src/bin/provider-capture.rs` | 新的 `RenderGeometry::NonIndexedQuad`：判决在 `render_geometry`，索引在 `render_inputs`（返回 `None`），计数在 `validate_render_case`；object 轨走 `draw_primitives_with_attachments`（流照绑、不绑 index） |

**按名拒桩**：`vertices < 3`、流短于 `vertices * stride`、`base_vertex != 0` 三者各有一条钉住的
用例（Rust 单元测试 + `conformance/test_suite_v39.py`），`test_oracle_coverage.py` 继续把 suite
表、`loadSuite` 表、`validate_suite` 表与 CI 的逐 suite 行对在一起。

**已知边界**：`NativeOracle.swift` 只在 Apple CI 上编译（Linux 侧没有 Swift），所以本机只能跑
元数据一致性检查；新 arm 的语法与语义由 Apple Paravirtual 设备上的抓图暴露。RTX 5060 与
Lavapipe 的三轨抓图对照见本文末的证据目录。

## 证据与可复跑命令

```sh
mkdir -p conformance/captures
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
cargo run --locked -p metal-smoke --bin provider-capture -- \
  --suite conformance/suite-v39.json --output conformance/captures/vulkan-v39.json
python3 conformance/compare.py --suite conformance/suite-v39.json \
  --check conformance/captures/vulkan-v39.json
python3 -m unittest discover -s conformance -p 'test_*.py'
python3 conformance/narrow_class.py --class render-narrow \
  --rail vulkan=conformance/captures/vulkan-v39.json --preview
```

RTX 5060 的三轨抓图与 Lavapipe 的逐字节对照在
`evidence/nonindexed-coverage-<sha>-2026-09-19/`；R 侧的 `07b68ca` 读数在
`evidence/nonindexed-07b68ca-2026-09-18/`。

## 消费方（另一仓，本 change 不含）

reims 侧**无需随动**：R39（`07b68ca`）已经放行这条臂、带着 `render_provider_out_of_class_vertex_span`
门，索引臂的门序逐字未变。E 侧这次只是把 R 早就能执行的形状写进受审覆盖面。

## 未决

- census 的 447/520 都是**首败桶的上界**：放行后这些 draw 会继续被
  `texture_extent` / `texture_bind` / `texture_source_order` 与 views 预算再筛一遍，真实搬动量只能
  靠下一轮 census 读。本增量不声称任何桶的搬运量。
- 两条 rail 的顶点覆盖率 slug 不同名（`render_vertex_buffer_footprint_unsupported` /
  `render_vertex_footprint_unsupported`）；要按名读 decline 分布需要先对齐或先登记，属独立小刀。
- `base_vertex != 0` 的非索引形状仍是类外，且契约层就拒绝（`BaseVertexRequiresIndices`）：这条边界
  是既有的、正确的，本增量不放宽。
