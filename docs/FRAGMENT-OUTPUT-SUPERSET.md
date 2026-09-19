# 模块声明的颜色 location 比挂载的附件多：按方向声明的形状位（E-FOS1）

本文记录本仓一个增量：一次 draw 的 fragment 模块**声明的颜色 location 多于** pass 挂载的
颜色附件时，canonical Vulkan rail 今天已经执行的那一半怎么变成消费方（reims）可读的帧事实，
以及它**不**覆盖的反方向为什么继续按名拒。它是 `docs/VERTEX-INTERFACE-SUPERSET.md`
（E-TX11）的姊妹篇：同一个"声明面 ⊇ 使用面"的概念，方向相反。

## 形状与今天的失败面

gate-3 census v46 的残桶 `stage_buffer_footprint`（58 = 29 条记录 × 2 次 charge）真身是
**一条** pipeline：`pipe=57` = `fixed_vert_lpf_gen` × `fixed_frag_lpf_cpf`（壁纸/图层 LPF）。
只读侦察（`.agents/tasks/root/e_stage_buffer_unbounded_reach-report.md`）把门序读全：
stage-buffer reach → D3 采样 → **fragment 输出多于附件**。第三道门的形状是确定的：

- 模块**无条件**写 Location 0/1/2（`OpStore`，反射 `render_targets` = 3 项）；
- 这批 draw 的 latch 是 `slots=1`（`req.colors.len()`），即**一个**颜色附件；
- R 侧**没有** render-target 计数门（全仓 `grep render_targets` 只命中测试桩），
  `register_render_pipeline` 给 provider 的契约是 `color_formats: vec![pass.format]`；
- E 侧 `validate_translated_fragment`（`metal-api-vulkan/src/render.rs`）要求
  `declared.len() == reflection.render_targets.len()` ⇒ **provider 在注册这一步拒**。

这不是"少一个桶"：class 已放行、provider 仍拒 ⇒ **丢画**，正是 census 红线
`draws_skipped_after_engine_refusal` 的形状（v33 的 245 条、v39 的 4 条同源）。

Vulkan 的定义没有歧义：fragment shader 写到**没有对应附件**的 location 时，该写入被
**丢弃**（不是错误）；只要管线布局与渲染通道的附件数、blend 状态合法，这个形状就是可执行
的。本 rail 的管线本来就按**契约**的 `color_formats` 建，所以执行面天然存在。

## 位的按方向语义（唯一真源）

```rust
// crates/metal-api-core/src/provider.rs
pub supports_render_fragment_output_superset: bool          // 默认 false
pub fn declares_render_fragment_output_superset_support(&self) -> bool
```

| 方向 | 含义 | 本增量的回答 |
|---|---|---|
| `reflected ⊇ attached` | 模块声明（并写入）的颜色 location **多于**契约挂载的附件 | 可执行（该位为真时）；多出来的 store 被丢弃，不进帧 |
| `attached ⊋ reflected` | 契约挂载了一个模块**从不写**的附件 | **按名拒**：该附件读回的字节没有任何写者 |

位是**逐 snapshot** 的，不能由"Vulkan 会执行"推出"这个 provider 会执行"。位缺席时注册门
逐字保留今天的相等规则：同一 slug（`render_stage_reflection_mismatch`）、同一字段组
（`field=render_targets`、`declared_targets`、`reflected_targets`）与同一句 detail。

## 两条轨的取值是事实

| 轨 | 取值 | 依据 |
|---|---|---|
| Vulkan | `true` | 管线按契约的 `color_formats` 建，模块保留自己声明的全部 store；证据是 `tests/render_fragment_output_superset_e2e.rs` 的读数 |
| native | `false` | `reviewed_module` 按 layout 与颜色格式列表的**精确形状**选 MSL 模块，一个"写了 pass 没有挂载的 location"的模块命中不了任何一条臂，rail 按名拒；Apple 侧没有该形状的 oracle |

## capability frame

| 面 | 内容 |
|---|---|
| 位与谓词 | `crates/metal-api-core/src/provider.rs` |
| 族内 tag | `CAPABILITY_RENDER_FRAGMENT_OUTPUT_SUPERSET_TAIL = 0x0E`（`command_codec.rs`） |
| 编码 | `0x00`（转义）+ `0x0E`（族内 tag）+ 一个 bool，写在 layout-free count 的 `0x00 0x0B <bool>` **之后** |
| 不声明该位 | 一个字节都不多写 |

tail 原有的 8 个 presence tag 是 `0x01..=0x80`；E-TX9 引入 `0x00` 转义族，本增量的族内 tag
是 `0x0E`（`0x0C` / `0x0D` 归同日并行两条 track）。族内 section 是连续的一段，每段自带
转义字节，按 encoder 的固定顺序写、也按同一顺序读；未知族内 tag / 乱序 / 重复一律
`UnknownCapabilityTail`。方向性：

| 帧 | 本增量后的解码器 | 旧解码器 |
|---|---|---|
| 旧帧（无新段） | 新位 `false`，不报错 | 行为不变 |
| 只声明新位 | 新位 `true` | 拒绝：`UnknownCapabilityTail(0x0e)` |

## 证据与可复跑命令

| 面 | 位置 |
|---|---|
| translated e2e（单附件落被挂载 texel、孪生同帧、双附件真写第二个 texel、object 轨逐字节相同、反方向按名拒、形状规则不被放宽、快照声明） | `crates/metal-api-vulkan/tests/render_fragment_output_superset_e2e.rs` |
| 计数的两个臂（严格臂逐字保留今天的句子） | `crates/metal-api-vulkan/src/render.rs` 的单测 `the_fragment_count_rule_widens_one_direction_on_the_rails_own_answer` |
| 快照声明与 rail 常量的一致性 | `crates/metal-api-vulkan/src/provider.rs` 的能力测试 |
| fixture | `tests/fixtures/render_two_output_rgba8.frag.ll`（+ `_alt` 孪生） |
| conformance 声明与 comparator/schema 检查 | `conformance/suite-v45.json`、`conformance/test_suite_v45.py` |
| wire 往返 / 只声明新位 / 旧帧 / 族内拒绝 | `crates/metal-api-ipc/src/command.rs` 的单测 |
| 默认快照与附件面读数不变 | `crates/metal-api-core/src/provider.rs` 的单测 |
| 两条轨快照声明（Vulkan `true` / native `false`） | `crates/metal-api-vulkan/src/provider.rs`、`crates/metal-api-native/src/render.rs` |

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  cargo test -p metal-api-vulkan --test render_fragment_output_superset_e2e

cargo run -p metal-smoke --bin provider-capture -- \
  --suite conformance/suite-v45.json --output /tmp/vulkan-capture-v45.json
python3 conformance/compare.py --suite conformance/suite-v45.json \
  --check /tmp/vulkan-capture-v45.json          # 另加 --api objects / --api objects --async
```

v45 的 fixture 是 `[[vertex_id]]` 三顶点的覆盖三角形：模块写 Location 0
`(64/255, 128/255, 192/255, 1)`（8 位 UNORM 下 `40 80 c0 ff`）与 Location 1
`(1, 0, 0, 1)`（`ff 00 00 ff`），契约只挂载一个附件。所以帧是
`4080c0ff`×4，而孪生（只把被丢弃的 Location 1 改成 `(0, 1, 0, 1)`）落**同一串字节**——
这把读数钉在可证伪的边界上：改被丢弃的 location ⇒ 帧逐字节不变；把同一模块挂到两个附件上
⇒ 第二个附件真的读出 `ff0000ff`×4。

## 消费方（另一仓，本 change 不含）

reims 侧随动的任务书由主代理另派：读新位，位真时放行"反射 `render_targets` 多于请求附件"
的形状并把**实际挂载**的那几路如实交给 provider；位缺席时逐字保留今天的 slug / 句子。
R 侧今天**没有**该形状的计数门（class 会把它交给 provider），所以顺序纪律是硬的：
**E 先 R 后** —— 反过来就是"class 放行、provider 拒"的丢画形状。

## 未决

1. **native 轨**：该形状在 native 侧继续按名拒；若将来 Apple 侧要为它出一个 oracle，那是一次
   独立的增量（新 reviewed 模块族 + 新的 selftest 读数），本增量不假装它会执行。
2. **请求侧 MRT**：`req.secondary_targets` 非空（请求自己声明多个附件）仍然按名留 engine，
   本增量只说"模块声明多于请求附件"这一条形状。
