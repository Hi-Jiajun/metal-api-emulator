# 折叠的 stage-buffer 形状：命名空间分离布局（E-TX9）

本文记录本仓一个增量：一个 pass 的两个 *translated* stage 各自读同一个
`[[buffer(n)]]` 索引时，canonical Vulkan rail 怎么执行它，以及消费方（reims）要按哪一位
放行。叙事全貌在 `research/docs/23` §3.3 的交接条目里；这篇只写 E 仓自己的契约、边界与
可复跑的验证命令。

## 形状与今天的失败面

Metal 的 `setVertexBuffer(_:offset:index:)` 与 `setFragmentBuffer(_:offset:index:)` 是两套
独立的索引空间：同一个 `n` 在两个 stage 上是两个 slot。`metal2vulkan` 的默认 descriptor
layout 把任一 stage 的每个 `[[buffer(n)]]` 都放在 `(set 0, binding n)`，所以这样一个 pair
的两份声明会折到同一个 descriptor 上——rail 按名拒绝
（`render_stage_buffer_layout_unsupported`），而不是让一个 stage 读到另一个 stage 的字节。

## 布局（唯一真源）

```rust
// crates/metal-api-vulkan/src/lib.rs
pub fn stage_buffer_namespace_layout() -> DescriptorLayout
pub use render::STAGE_BUFFER_NAMESPACE_SET;   // = STAGE_BUFFER_REVIEWED_SET_BASE = 1
```

布局把**一个 stage 的整段 layout** 挪到 set 1；其余 band 保持翻译器自己的取值。折叠形状下
移动的是 **vertex** stage：rail 的纹理通路是 fragment 专属且写死 set 0
（`translated_texture_pairs` 对非 fragment 直接返回空），所以搬 vertex 的 layout 在效果上
只会搬它自己的 buffer descriptor，fragment 的纹理/采样器仍在 canonical band 上。set 1 也
正是 reviewed pair 的 vertex 槽位，未超过 rail 已命名的 set ceiling（=2）。

选择归调用方：rail 拿不到 AIR，只能按 module 自己的 reflection 执行；只要两段 slot 不相交
就执行，折到同一 slot 就按名拒绝（拒绝文本会指出改用上面的 layout）。E 侧不发布"是否折叠"
的判定，那是消费方 class gate 已经在算的同一件事，两仓各留一份规则只会漂移。

## 能力位与 wire

| 面 | 内容 |
|---|---|
| `ProviderCapabilities` | 新位 `supports_render_stage_buffer_namespace_split`（默认 `false`）与 `declares_render_stage_buffer_namespace_split()` |
| Vulkan rail 快照 | `true`（本增量的证据面） |
| native rail 快照 | `true`（reviewed pair 本来就绑 set 1 / set 2，Apple 读数即证据，不新增实现） |
| capability frame | 独立 presence-tag 块，payload 一个 bool；旧 frame 解码为 `false` |

frame 细节（`crates/metal-api-ipc/src/command_codec.rs`）：tail 原有的 8 个 presence tag 是
`0x01..=0x80`，八个既有块已经全部占用，所以第九块用**转义字节**引入——`0x00`（"后面是本族
自己的 tag"）再跟族内 tag `0x01`，最后是 bool。`0x00` 在任何旧版本里都是
`UnknownCapabilityTail`，因此没有哪一版 frame 会被读错；既有块的字节与顺序一字未动，新块
只出现在尾部。

两个诚实的边界：

1. 旧版本解码器读新 frame 会以 `TrailingPayload` 拒绝（`decode_response_payload` 末尾有
   `decoder.finish()`），不是"静默忽略尾块"。方向是 fail-closed 的，但它与"旧解码器忽略
   新块"不是同一句话：向后兼容成立的是"旧 frame → 新位 false"这一侧（这也是规格要求的那
   一侧）。
2. 本增量**不**声称完整 Metal conformance，也不承诺 rail 会自动分离命名空间：不选布局的
   调用方仍然得到按名拒绝。

## 证据与可复跑命令

| 面 | 位置 |
|---|---|
| translated e2e（4 个用例：落帧、两条 payload 反例、默认布局仍拒绝、与 reviewed pair 同设备） | `crates/metal-api-vulkan/tests/render_stage_buffer_namespace_e2e.rs` |
| v112 合并 set 0 的新旧对照（默认布局拒绝 + namespace 布局执行并落字节） | `crates/metal-api-vulkan/tests/render_set0_layout_e2e.rs` |
| conformance 声明与 comparator/schema 检查 | `conformance/suite-v35.json`、`conformance/test_suite_v35.py` |
| wire 往返 / 旧 frame / 未知 tag 拒绝 | `crates/metal-api-ipc/src/command.rs` 的单测 |
| 布局真源与字节不变性（无 buffer 模块两布局 SPIR-V 相同） | `crates/metal-api-vulkan/src/render.rs` 的单测 |

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  cargo test -p metal-api-vulkan --test render_stage_buffer_namespace_e2e \
             --test render_set0_layout_e2e

cargo run -p metal-smoke --bin provider-capture -- \
  --suite conformance/suite-v35.json --output /tmp/vulkan-capture-v35.json
python3 conformance/compare.py --suite conformance/suite-v35.json \
  --check /tmp/vulkan-capture-v35.json        # 另加 --api objects / --async 两轮
```

期望帧钉在 reviewed `stage_buffer_borrowed_tint_2x2`（`conformance/suite-v31.json`，Apple
设备已测）上：同一份 position/tint 字节落 `40 80 c0 ff` + 三格 clear（2×2 `rgba8_unorm`），
所以"两段各自读到自己字节"是逐 texel 可证伪的，而不是只断言 pass 被接受。

## 消费方（另一仓，本 change 不含）

reims R32 随动清单写在 `~/.agents/tasks/root/r32_reims_namespace_split.md`：读该位、折叠时给
vertex 传 `stage_buffer_namespace_layout()`、未声明位时保持 R31 的按名 fail-closed，并用一轮
census 验证 `stage_buffer_shape_folded` 归零且四个零读数不掉。
