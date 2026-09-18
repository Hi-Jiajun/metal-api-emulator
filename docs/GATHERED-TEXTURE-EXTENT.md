# 另一 extent 的采样源：按臂声明的形状位（E-TX10）

本文记录本仓一个增量：一次 render pass 里被采样的纹理其 extent 与 render area 不一致时，
canonical Vulkan rail 今天已经执行的那一半怎么变成消费方（reims）可读的帧事实，以及它**不**
覆盖的那一半为什么继续按名拒。叙事全貌在 `research/docs/23` §3.3 的交接条目里；这篇只写 E
仓自己的契约、边界与可复跑的验证命令。

## 形状与今天的失败面

gate-3 census v26（reims `4bd90c1` × E `9fbbfab`）的最大首败桶是 `texture_extent`：6,095 行，
占全部被拒 draw 的 46.6%。R35 新加的两条 route 把它按"源有没有 host 字节"劈成
`texture_extent_host_bytes` **5,679** 行（93.2%）与 `texture_extent_borrowed_no_copy` **416**
行（6.8%）。

E 侧的事实是：Vulkan rail 在 E-TX5 之后**已经执行**前一半——reviewed 模块的采样点是
fragment 自己的中心，rail 把源 gather 进 render area 自己的整数网格；translated 模块自己声明
绝对采样坐标，rail 按**源自身**的 extent 绑定。两条路都只要求源的字节在 host 侧可读。唯一
的按名拒是"reviewed + owner 的零拷贝窗口"（`render_texture_extent_unsupported`，字段
`source=borrowed_no_copy`）。native rail 对**任何**另一 extent 的源都按名拒。

难点不在执行，而在**声明**：`ProviderCapabilities` 里原本没有哪一位让消费方区分"会执行
host-bytes 臂的 rail"与"两条臂都拒的 rail"，于是 R 侧只能对整个形状 fail-closed——5,679 行
继续留在 engine 里。

## 位的按臂语义（唯一真源）

```rust
// crates/metal-api-core/src/provider.rs
pub supports_render_texture_gathered_extent: bool          // 默认 false
pub fn declares_render_texture_gathered_extent_support(&self) -> bool
```

它是 render-sampler 面的**第四个问题**，与既有三个字段各答一问：

| 字段 | 回答的问题 | 本增量 |
|---|---|---|
| `supports_render_texture_sampling` | 会不会采样 render pass 的纹理（同 extent 窗口） | 不变 |
| `max_render_textures` | 一个 pass 能绑几个采样纹理 | 不变 |
| `supported_render_texture_formats` | 哪些格式能被采样 | 不变 |
| `supports_render_texture_gathered_extent`（新） | 另一 extent 的源、且源字节在 host 侧，会不会被执行 | 新增，默认 `false` |

语义**按臂限定**：为真 = "这个 snapshot 会执行"另一 extent + host 字节"这一半"。它 MUST NOT
被读成"会把任意源拷到 host"，也 MUST NOT 被读成"会执行 owner 的零拷贝窗口"——那一臂的整个
陈述是"device 直接读 owner 的映射"，另一 extent 在那里需要一次 host 拷贝，本增量不提供该
通道，它继续按名拒（同一个 slug、同一组字段、同一条 detail）。

两条轨的取值是**事实**而不是缺陷：Vulkan 声明 `true`（E-TX5 的 gather / translated 绑定 +
`render_texture_extent_e2e.rs` 的读数），native 保持 `false`（该轨每一份源都按名拒，Apple
侧没有该形状的 oracle）。位是**逐 snapshot** 的，不能由"Vulkan 会执行"推出"这个 provider 会
执行"。

## capability frame

| 面 | 内容 |
|---|---|
| 位与声明 helper | `crates/metal-api-core/src/provider.rs` |
| 族内 tag | `CAPABILITY_RENDER_TEXTURE_GATHERED_EXTENT_TAIL = 0x02`（`command_codec.rs`） |
| 编码 | `0x00`（转义）+ `0x02`（族内 tag）+ 一个 bool，写在 E-TX9 的 `0x00 0x01 <bool>` **之后** |
| 不声明该位 | 一个字节都不多写 |

tail 原有的 8 个 presence tag 是 `0x01..=0x80`，已被八个块占满；E-TX9 引入 `0x00` 转义族，
本增量用**族内下一个 tag**。族内 section 是**连续的一段**，每个 section 自带转义字节，按
encoder 的固定顺序写、也按同一顺序读：decoder 的 `decode_capability_extended_tail` 改成消费
整段（未知族内 tag / 乱序 / 重复 / 族内 section 之后出现非转义字节，一律
`UnknownCapabilityTail`），因此八个调用点的"返回 `true` = 帧到此为止"契约不变。

方向性：

| 帧 | 本增量后的解码器 | E-TX9 期的解码器 |
|---|---|---|
| 旧帧（无新块） | 新位 `false`，不报错 | 行为不变 |
| 只声明新位 | 新位 `true` | 拒绝：`UnknownCapabilityTail(0x02)` |
| 两位都声明 | 两位 `true` | 读完成 `00 01 <bool>` 后剩 3 字节 ⇒ `TrailingPayload{extra: 3}` |

两条诚实边界：(1) 新帧对旧解码器是**整帧被拒**，不是"静默忽略尾块"——保持 fail-closed，但
与"旧解码器跳过新块"不是同一句话；(2) 本增量**不**声称完整 Metal conformance，也不放宽纹理
格式表、binding 上限、MSAA/深度/展示等其它形状。

## 证据与可复跑命令

| 面 | 位置 |
|---|---|
| translated e2e（另一 extent 的 host-bytes 源进入 provider、换源字节换帧、未读 texel 不动帧、object 轨与 trace 轨逐字节相同、零拷贝臂仍按名拒） | `crates/metal-api-vulkan/tests/render_texture_extent_e2e.rs` |
| conformance 声明与 comparator/schema 检查 | `conformance/suite-v36.json`、`conformance/test_suite_v36.py` |
| wire 往返 / 只声明新位 / 旧帧 / 族内拒绝 | `crates/metal-api-ipc/src/command.rs` 的单测 |
| 默认快照与既有三个 render-sampler 读数不变 | `crates/metal-api-core/src/provider.rs` 的单测 |
| 两条轨快照声明（Vulkan `true` / native `false`） | `crates/metal-api-vulkan/src/provider.rs`、`crates/metal-api-native/src/render.rs` |

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  cargo test -p metal-api-vulkan --test render_texture_extent_e2e

cargo run -p metal-smoke --bin provider-capture -- \
  --suite conformance/suite-v36.json --output /tmp/vulkan-capture-v36.json
python3 conformance/compare.py --suite conformance/suite-v36.json \
  --check /tmp/vulkan-capture-v36.json        # 另加 --api objects 一轮
```

v36 的源纹理是 6×4、render area 是 4×4，translated fragment 模块的两个绝对采样落在源列 5 与
列 1；源网格里第 `x` 列的红色通道是 `16 * x`，所以每个 fragment 落 `50 10 00 ff`。这把读数
钉在定义式 oracle 上：**gather 成目标网格**会回答 `50 20 00 ff`（v36 的 comparator 用例拒绝
这一帧），而**只换源字节**（列 5 或列 1 的红色）必须只换对应的输出通道。

## 消费方（另一仓，本 change 不含）

reims R36 随动的任务书是 `~/.agents/tasks/root/r37_extent_bit_wire.md`：读新位、位真时只放行 R35 的
`texture_extent_host_bytes` 臂，位为假或 `texture_extent_borrowed_no_copy` 臂逐字保留今天的
slug / 句子 / route；随后一轮 census 读 `texture_extent_host_bytes` 转 canonical、
`texture_extent_borrowed_no_copy` 仍按名、四个零读保持 0。

## 未决

`conformance/narrow-class.json`（Gate 2 G2-f 的窄类声明）**本轮不加宽**：该类的 covered
rules 目前把 `fragment_textures` 归为"类未覆盖的轴"（`conformance/narrow_class.py` 的
`RENDER_INERT_KEYS`），把一个另一 extent 的源写进类声明等于给窄类加一条新的 covered rule，
这件事属于 G2-f 自己的声明演化，应由 R36 合流后的一轮 census 读数来支撑，而不是随本增量顺带
改。E 侧因此只发布"这个 provider 会执行该臂"的帧事实。
