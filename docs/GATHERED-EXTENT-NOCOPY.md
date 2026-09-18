# 零拷贝窗口的另一 extent：设备侧整数索引（E-TX12）

本文记录本仓一个增量：一次 render pass 里被采样的纹理是 **owner 的零拷贝窗口**
（`TextureSource::BorrowedNoCopy`）且其 extent 不等于 render area 时，canonical Vulkan
rail 不再按名拒——它用 reviewed 采样对的第二个片元模块，把"目的地网格"的 texel 索引在
**设备侧**用整数算出，直接读 owner 的映射。叙事全貌在 `research/docs/23` §111 的交接条目
里；这篇只写 E 仓自己的契约、边界与可复跑的验证命令。

## 形状与今天的失败面

gate-3 census v29/v30b 的 `texture_extent` 桶（1,096 / 5,893 行）**全部**是 R35 的
`texture_extent_borrowed_no_copy` 那一臂：源的字节在 owner 的映射里，host 侧没有一份可
gather 的拷贝。E 侧此前的事实有两条：

* Vulkan rail 对 **host 字节可读**的源已经执行（E-TX5 的整数网格 gather、translated
  模块按源自身 extent 绑定），并在 E-TX10 用 `supports_render_texture_gathered_extent`
  发布了那一半；
* 同一形状里 **owner 零拷贝窗口**这一半按名拒（`render_texture_extent_unsupported`，
  字段 `source=borrowed_no_copy`），因为采样器只会把 texel 的选择交给驱动的插值与过滤
  精度，而这一臂既没有 host 拷贝通道，也不该把定义式读数变成实现相关的读数。

难点仍在**声明**：`supports_render_texture_gathered_extent` 的语义按臂限定，MUST NOT 被
读成"会执行零拷贝窗口"，于是 R 侧只能继续 fail-closed 这一半。

## 决策：设备侧整数索引，不拷贝、不交给采样器

"把 owner 的窗口 gather 进 render area 网格"在本 rail 上只有一条既忠实又精确的路径，
另外两条都被排除：

| 方案 | 结论 |
|---|---|
| host 侧拷贝 owner 的映射后 gather | 排除。这一臂的陈述是"设备读 owner 的映射"，host 拷贝是它没有的通道；且拷贝发生在 resolve 期，读到的字节与设备执行期读到的不是同一件事 |
| `vkCmdBlitImage`（NEAREST）重采样 | 排除。texel 的选择落回驱动的过滤/插值精度，而 census 的形状里正好有"目的地中心落在源 texel 边界上"的比例（如 `64→80`），边界舍入会让读数与定义式不等 |
| shader 侧整数索引（本增量） | 采用。索引 `floor((2 * index + 1) * source / (2 * destination))` 在设备侧用 `u32` 精确算出，`OpImageFetch` 读同一枚 texel——与 rail 的 host gather（`gather_texel_index`）逐 texel 同一定义，且 owner 的映射仍由设备读取 |

索引的两个 extent 通过 **specialization constant** 注入：pipeline 本来就是按 pass 建的，
所以这条算术是这次 pass 自己的形状，而不是一个可能没被写过的 uniform。四个 `SpecId`
依次是 `source_width`、`source_height`、`destination_width`、`destination_height`。

## 声明面：reviewed 采样对的第二个片元模块

reviewed 采样对（`SAMPLED_QUAD_VERT_SPV` + 采样片元模块）现在有一个**兄弟模块**
`render_spv/gathered_fetch.frag.spv`（入口同为 `fragment_main`）。选哪个由**注册声明的
那个模块**决定，两侧都按同一张表校验：

| 注册声明 | 这一臂的行为 |
|---|---|
| 采样兄弟（`solid_unorm8_sampled.frag.spv`） | 保持今天的按名拒：`render_texture_extent_unsupported`，字段与 detail 与前一个增量逐条相同 |
| 收集兄弟（`gathered_fetch.frag.spv`） | 执行：owner 的窗口按源自身 extent 建镜像、不建 `VkSampler`（槽位是 `SAMPLED_IMAGE`），片元模块按目的地网格的整数索引 `OpImageFetch` |

于是"执行的模块"仍然是"声明的模块"，这是本 rail 从 v70 起对 reviewed 对的一贯规则。
translated 模块那一臂不受影响：它自己声明采样坐标，rail 按源自身 extent 绑定 owner 的
窗口——census 的形状正是从这条路进来的（fork 注册 guest 的 translated 阶段）。

两条**新**的按名拒把边界写清楚，而不是静默兜底：

| slug | 何时出现 |
|---|---|
| `render_texture_gathered_fetch_arm_unsupported` | 注册声明了收集兄弟，但这次 pass 的采样源不是"另一 extent + 零拷贝窗口"（字段带 `source`：`owned_bytes`/`staged_lease`/`trace_view`） |
| `render_texture_gathered_fetch_window` | 形状对，但 `(2 * render_extent - 1) * source_extent` 超过 `u32`（模块的算术窗口；任何真实设备的最大图像维度都远在窗口内，这是**声明的界**而不是会遇到的形状） |

## 能力位与 wire

```rust
// crates/metal-api-core/src/provider.rs
pub supports_render_texture_gathered_extent_no_copy: bool          // 默认 false
pub fn declares_render_texture_gathered_extent_no_copy_support(&self) -> bool
```

它是 E-TX10 那一位的**另一半**，而不是同一位的第二种读法：两半由不同的代码回答，一个
snapshot 可以执行其中一半而继续按名拒另一半。位的陈述是"会执行另一 extent 的源、且该源的
字节就是 owner 的映射"，**不是**拷贝通道，也**不是**关于 host-bytes 那一半的声明。

它在 capability frame 尾部第二个 tag 族里以**第四个** tag 编码：`0x00 0x04 <bool>`，写在
E-TX11（superset vertex interface）的 `0x00 0x03 <bool>` 之后。不声明该位的 snapshot 一个
字节都不多写；旧帧解出的读数是 `false`；只声明这一位的 snapshot 仍写扩展载荷（否则声明
会在 wire 上被丢掉）。

两条轨的取值是**事实**：Vulkan 声明 `true`（收集兄弟 + translated 绑定，两者都读 owner
的映射），native 保持 `false`（该轨对任何另一 extent 的源都按名拒，Apple 侧没有这个形状的
oracle）。

## 证据与可复跑命令

| 面 | 位置 |
|---|---|
| 零拷贝臂 e2e：收集兄弟的帧与 host gather 逐字节相同、换源字节则换帧、**没被读到的 texel 不动帧**、translated 臂读源自身 extent、两条新的按名拒 | `crates/metal-api-vulkan/tests/render_texture_extent_nocopy_e2e.rs` |
| 采样兄弟仍按名拒（字段与 detail 不变） | `crates/metal-api-vulkan/tests/render_texture_extent_e2e.rs` |
| 算术窗口与"收集兄弟只认一个臂"的 host 侧读数 | `crates/metal-api-vulkan/src/render.rs` 的单测 |
| wire 往返 / 只声明该位 / 旧帧 / 族内 tag 闭集 | `crates/metal-api-ipc/src/command.rs`、`command_codec.rs` 的单测 |
| 两条轨的 snapshot 声明（Vulkan `true` / native `false`） | `crates/metal-api-vulkan/src/provider.rs`、`crates/metal-api-native/src/{render.rs,native.rs}` |

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  cargo test -p metal-api-vulkan --test render_texture_extent_nocopy_e2e

VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  cargo test -p metal-api-vulkan --test render_texture_extent_e2e

cargo test -p metal-api-ipc --lib command
```

## 边界（本增量不做的）

* **不是 host 拷贝通道**：provider 不把 owner 的窗口拷到 host 再 gather，这一臂的字节始终
  从 owner 的映射读；
* **不是所有模块都能读**：采样兄弟仍然按名拒这一臂（这是它自己的语义），要执行就得在注册
  里声明收集兄弟；
* **不改同 extent 窗口**：同 extent 的零拷贝采样、host-bytes gather、translated 绑定、
  四种 UNORM 格式表、binding 上限、MSAA/深度/展示都不动；
* **不声称完整 Metal conformance**：native 轨继续按名拒，Apple 侧没有该形状的 oracle；
* `conformance/narrow-class.json`（Gate 2 G2-f 的窄类声明）本轮不加宽：新覆盖规则属于
  R 侧随动后的 census 轮，而不是随本增量顺带改。
