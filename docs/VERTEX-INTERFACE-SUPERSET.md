# 声明的 vertex layout 比模块读的多：按方向声明的形状位（E-TX11）

本文记录本仓一个增量：一次 draw 的 contract 里 vertex layout 声明的 attribute location 集合
**严格覆盖**模块实际读取的 location 集合时，canonical Vulkan rail 今天已经执行的那一半怎么变成
消费方（reims）可读的帧事实，以及它**不**覆盖的反方向为什么继续按名拒。叙事全貌在
`research/docs/23` §3.3 的交接条目里；这篇只写 E 仓自己的契约、边界与可复跑的验证命令。

## 形状与今天的失败面

gate-3 census v27b（reims `cc51f9d` ≈ `7540d05` × E `aa6cfa3`）的最大首败桶是
`vertex_interface`：**5,241 行**，占全部被拒 draw 的 **50.2%**。R-VI1 新加的方向路由把整轮求和
读成：

| route | 行数 |
|---|---|
| `vertex_interface_declared_superset` | **5,241** |
| `vertex_interface_reflected_superset` | 0 |
| `vertex_interface_location_mismatch` | 0 |

也就是 **100% 是"契约/描述符多声明了 attribute location，模块并不读"**。
`MTLVertexDescriptor` 命名一个 vertex function 从不读的 location 在 Metal 里是合法形状：那条流
**绑定但忽略**。E 侧的执行面本来就按 declared 建——`render.rs` 的 `create_pipeline` 用
`resolve_vertex_streams` 把 `pass.vertex_buffers` 与 `contract.vertex_layout.buffers()` 配对，为
每条 declared attribute 发一个 `VkVertexInputAttributeDescription`，SPIR-V 模块只声明它读的
location——所以这个形状**天然能执行**。

今天拦住它的只有两条"必须逐位置相等"的规则：E 侧
`validate_translated_vertex_attributes`（先比两侧**数量**、再逐条 declared 找 reflected），
R 侧 R-VI1 的门。于是 5,241 行留在 engine。

## 位的按方向语义（唯一真源）

```rust
// crates/metal-api-core/src/provider.rs
pub supports_render_vertex_interface_superset: bool          // 默认 false
pub fn declares_render_vertex_interface_superset_support(&self) -> bool
```

它是 vertex-input 面的**第四个问题**，与既有三个字段各答一问：

| 字段 | 回答的问题 | 本增量 |
|---|---|---|
| `max_vertex_buffers` | 一个 pass 能声明几条流 | 不变 |
| `supported_vertex_formats` | 哪些 attribute 格式 | 不变 |
| `supported_index_formats` | 哪些 index 宽度 | 不变 |
| `supports_render_vertex_interface_superset`（新） | declared **严格覆盖** reflected 的 layout 会不会被执行（多声明的流绑定并忽略） | 新增，默认 `false` |

语义**按方向**限定，两个方向不是同一句话：

| 方向 | 含义 | 本增量的回答 |
|---|---|---|
| `declared ⊇ reflected` | 模块读的每个 location 都有 declared 覆盖，且 component 形状一致 | 可执行（该位为真时）；多余的 declared attribute 只出现在 vertex input state 与 bind 步骤，不参与帧 |
| `reflected ⊄ declared` | 模块读的某个 location 没有任何 declared attribute 覆盖 | **按名拒**（`render_stage_reflection_mismatch`，`field=vertex_attributes` + 缺的 `location`）：vertex input state 里没有那一条，Vulkan 给该输入的值是未定义的，rail 不能替调用方猜一个流 |

位是**逐 snapshot** 的，不能由"Vulkan 会执行"推出"这个 provider 会执行"。

## 两条轨的取值是事实

| 轨 | 取值 | 依据 |
|---|---|---|
| Vulkan | `true` | `create_pipeline` 按 contract 的 layout 建 vertex input state（`render.rs`）；证据是 `tests/render_vertex_superset_e2e.rs` 的读数 |
| native | `false` | `reviewed_module` 按 layout 的**精确形状**查表选 MSL 模块（一条流两条 attribute 的 depth/instanced 形状等），4 条 attribute 的 layout 命中不了任何一条，rail 按名拒；Apple 侧没有该形状的 oracle |

## capability frame

| 面 | 内容 |
|---|---|
| 位与声明 helper | `crates/metal-api-core/src/provider.rs` |
| 族内 tag | `CAPABILITY_RENDER_VERTEX_INTERFACE_SUPERSET_TAIL = 0x03`（`command_codec.rs`） |
| 编码 | `0x00`（转义）+ `0x03`（族内 tag）+ 一个 bool，写在 E-TX10 的 `0x00 0x02 <bool>` **之后** |
| 不声明该位 | 一个字节都不多写 |

tail 原有的 8 个 presence tag 是 `0x01..=0x80`，已被八个块占满；E-TX9 引入 `0x00` 转义族，
E-TX10 用了族内 `0x02`，本增量的族内 tag 是 `0x03`。族内 section 是**连续的一段**，每个 section
自带转义字节，按 encoder 的固定顺序写、也按同一顺序读：decoder 的
`decode_capability_extended_tail` 的闭集与分支扩到 `0x03`（未知族内 tag / 乱序 / 重复 / 族内
section 之后出现非转义字节，一律 `UnknownCapabilityTail` 或该帧的 `TrailingPayload`），因此
各调用点的"返回 `true` = 帧到此为止"契约不变。

方向性：

| 帧 | 本增量后的解码器 | E-TX10 期的解码器 |
|---|---|---|
| 旧帧（无新块） | 新位 `false`，不报错 | 行为不变 |
| 只声明新位 | 新位 `true` | 拒绝：`UnknownCapabilityTail(0x03)` |
| 三位都声明 | 三位均按声明读出 | 读完成 `00 02 <bool>` 后剩 3 字节 ⇒ `TrailingPayload{extra: 3}` |

两条诚实边界：(1) 新帧对旧解码器是**整帧被拒**，不是"静默忽略尾块"——保持 fail-closed；(2)
本增量**不**声称完整 Metal conformance，也不动采样、MSAA、深度、展示、heaps/ICB 等其它形状。

## 证据与可复跑命令

| 面 | 位置 |
|---|---|
| translated e2e（4 条 declared / 2 条被读进入 provider、被忽略字节不动帧、被读流换字节换帧、object 轨与 trace 轨逐字节相同、反方向按名拒） | `crates/metal-api-vulkan/tests/render_vertex_superset_e2e.rs` |
| 注册门的两个方向 | `crates/metal-api-vulkan/tests/render_translated_stage.rs` |
| conformance 声明与 comparator/schema 检查 | `conformance/suite-v37.json`、`conformance/test_suite_v37.py` |
| wire 往返 / 只声明新位 / 旧帧 / 族内拒绝 | `crates/metal-api-ipc/src/command.rs` 的单测 |
| 默认快照与既有 vertex-input 读数不变 | `crates/metal-api-core/src/provider.rs` 的单测 |
| 两条轨快照声明（Vulkan `true` / native `false`） | `crates/metal-api-vulkan/src/provider.rs`、`crates/metal-api-native/src/render.rs` |

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  cargo test -p metal-api-vulkan --test render_vertex_superset_e2e

cargo run -p metal-smoke --bin provider-capture -- \
  --suite conformance/suite-v37.json --output /tmp/vulkan-capture-v37.json
python3 conformance/compare.py --suite conformance/suite-v37.json \
  --check /tmp/vulkan-capture-v37.json        # 另加 --api objects 一轮
```

v37 的 fixture 是三顶点的覆盖三角形：模块读 location 0 的 `float2` 位置与 location 1 的
`float2` 偏移，契约在同一条 32 字节记录上多声明 location 2、3（各 `float2`，值远在 clip space
之外）。偏移把三角形挪到只覆盖 2×2 的三个 texel，所以帧是 `4080c0ff fefefefe 4080c0ff 4080c0ff`
（v37 的 comparator 用 `coverage=partial` 钉住"画的 texel 与 clear texel 都在"）。这把读数钉在
可证伪的边界上：**改被忽略的 16 字节 ⇒ 帧逐字节不变**（它们被绑定但没进帧），**改 location 1
的偏移 ⇒ 帧跟着变**；一个忽略整条流的 rail 会落"整屏红"的 `4080c0ff × 4`，v37 的 comparator
用例拒绝这一帧。

## 消费方（另一仓，本 change 不含）

reims R-VI1 随动的任务书由主代理另派：读新位，位真时只放行
`vertex_interface_declared_superset` 这一方向，位为假或反方向（`reflected_superset` /
`location_mismatch`）逐字保留今天的 slug / 句子 / route；随后一轮 census 读
`vertex_interface` 桶的下降幅度，以及
`chain_resident_land_fail` / `load_target_content_not_ready` /
`draws_skipped_after_engine_refusal` / `vk_engine_target_read` 四个零读保持 0。

## 未决

1. **多流拼写**：被忽略的 attribute 位于**第二条 stream**（另一个 binding）时是同一条规则的另一种
   拼写，本增量只由单流的 fixture 覆盖（同一条记录里的两条多余 attribute）。是否补一条多流用例
   取决于 census 合流后是否真的出现该行——它只加用例，不改契约或实现。
2. **native 轨**：该形状在 native 侧继续按名拒；若将来 Apple 侧要为它出一个 oracle，那是一次
   独立的增量（新 reviewed 模块族 + 新的 selftest 读数），本增量不假装它会执行。
