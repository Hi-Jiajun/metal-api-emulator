# 窄通道采样格式（`r8_unorm` / `rg8_unorm`）

本文件记录 `render-sampled-narrow-lanes` 这一增量：契约、wire、两条 provider rail 与
conformance 上"被采样的纹理只有一个或两个 8-bit 通道"这件事。与
`docs/STAGE-BUFFER-NAMESPACE.md` 同形：先写形状与边界，再写证据与可复跑命令，最后写未决面。

## 1. 形状与动机

census v25b 的 `texture_bind` 桶 1,785 行被逐行归桶后只剩**一个**理由（v26 的 889 行同型）：
被采样的纹理其 bind/view 五轴（view 形状、extent、layers、descriptor 数、多重采样）全部合规，
唯一越窗的是 texel 格式 —— `R8_UNORM`（1,778 / 850 行）与 `R8G8_UNORM`（7 / 40 行）。拒绝句出自
消费方（reims `provider_render.rs`），它读的是 E 快照的 `supported_render_texture_formats`；
而 E 的 `TextureFormat` 里**根本没有这两个变体**，所以这是契约级缺口而不是执行缺口：Vulkan 的
`VK_FORMAT_R8_UNORM` / `VK_FORMAT_R8G8_UNORM` 与 Metal 的 `.r8Unorm` / `.rg8Unorm` 都能表达，
采样语义两侧一致。

| 契约格式 | Vulkan | Metal | 采样读出的四分量 |
|---|---|---|---|
| `r8_unorm` | `VK_FORMAT_R8_UNORM` | `.r8Unorm` | `(r, 0, 0, 1)` |
| `rg8_unorm` | `VK_FORMAT_R8G8_UNORM` | `.rg8Unorm` | `(r, g, 0, 1)` |

`0` 与 `1` 是**采样规则**的规定语义（缺失分量补 0、alpha 补 1），不是本项目的选择；`r`、`g` 是
该字节除以 255 的归一化值。所以"按整数 texel 中心采样进 `rgba8_unorm` 附件"的期望是**定义式**
的：源 byte `v` 在附件里就是 `[v, 0x00, 0x00, 0xff]`，不需要设备读数来定义。

## 2. 契约与 wire

- `TextureFormat` 追加 `R8Unorm` / `R8G8Unorm`（`crates/metal-api-core/src/provider.rs`），
  `bytes_per_texel` 分别是 1 与 2；**既有五个变体的语义一个都不动**。
- `TextureFormat::RENDER_SAMPLED` 从两个成员扩到四个：
  `[Rgba8Unorm, Bgra8Unorm, R8Unorm, R8G8Unorm]`。它回答的是"render pass 可以采样哪些格式"，
  不是"哪些格式共享四分量布局"，所以窄通道属于它；capability 快照的
  `supported_render_texture_formats` 就是它的镜像。
- wire 码表（`crates/metal-api-ipc/src/command_codec.rs`）把两个新变体**追加**为 code 5 / 6，
  既有 0–4 的映射逐字节不变。旧解码器遇到 5/6 走 `get_texture_format` 的兜底，报
  `UnknownEnumValue { field: "texture format", value }` 整帧拒绝，而不会把它读成别的格式。

## 3. 执行与边界：两条轨的分歧

**Vulkan rail 执行这两个 lane。** 三处格式闸门（pipeline 声明、pass 视图、
`render_texture_vk_format`）都读 `RENDER_SAMPLED`，扩成员后一起放宽；`render_texture_vk_format`
把两种格式映射到对应的 `VkFormat`，图像按视图自己的格式创建、上传、绑定，没有 component
mapping。

上传路径上有一处**不是闸门但必须改**的地方：`upload_render_texture` 原来把"每行紧致字节数"
写成 `width * 4`。窄 lane 的行宽是 `width * 1` 或 `width * 2`，旧写法会把第二行起的所有行写进
第一行之后的 driver 空隙里 —— 症状是帧的第一行正确、其余行全零。修复方式是把
`OffscreenRenderTexture.texel_bytes`（= `view.format.bytes_per_texel()`）传进上传，行宽由它算。

**native rail 不假装支持。** `SUPPORTED_RENDER_TEXTURE_FORMATS` 保持 `[Rgba8Unorm]`，两种窄格式
继续按名拒（`render_texture_format_unsupported`，在任何 Metal 对象之前）。Metal 确实能表达
`.r8Unorm` / `.rg8Unorm`，但该 rail 执行的是**reviewed MSL 模块**，它的窗口是一个字节序的四分量
表面；快照里的格式列表是"我真正执行"的声明，不是"设备支持"的声明。Apple 侧的自测读数落地前，
把窄格式写进这张表等于用一个没有读数的表去声明能力，所以窄表不动 —— 这是**事实的分歧**，
不是遗漏：`cargo test -p metal-api-native` 里有一条断言把这条边界钉住。

**R8 作为 render target 不在本增量。** `AttachmentFormat` 不动，R8 视图作为颜色附件仍按既有名
拒绝。本增量只解"采样源"这一面。

## 4. 证据

| 面 | 位置 | 读什么 |
|---|---|---|
| 契约 | `crates/metal-api-core/src/provider.rs` 的 `the_narrow_sampled_formats_are_one_and_two_byte_texels` 与 `render_texture_bits_gate_the_pass_the_count_and_the_format` | 1/2 字节、`RENDER_SAMPLED` 的成员与顺序、窄格式被四字节窗口按名拒、被四个成员的窗口放行 |
| wire | `crates/metal-api-ipc/src/command.rs` 的 `the_narrow_texture_format_codes_are_appended_and_read_back`、`a_legacy_decoder_refuses_the_narrow_codes_by_name` | 0–4 的字节不变、5/6 的往返、未知 code 的 `UnknownEnumValue` |
| Vulkan 执行 | `crates/metal-api-vulkan/tests/render_narrow_texture_e2e.rs`（三条） | R8 源的帧 == `[v, 0, 0, 0xff]`、换源字节就换帧、object rail 与 trace rail 逐字节相同、与 `rgba8_unorm` 兄弟共享红通道、`rgba16_float` 仍按名拒 |
| native 边界 | `crates/metal-api-native/src/render.rs` 的 `the_narrow_sampled_formats_stay_refused_by_name`、`render_texture_capability_bits_name_the_reviewed_window` | 两种窄格式按名拒且字段不变、快照列表仍是一个格式 |
| conformance | `conformance/suite-v37.json` 的 `sampled_texel_r8_4x4` 与 `conformance/test_suite_v37.py`（12 条） | 定义式期望、`capture_rails` 只含 Vulkan 两条轨、四分量上传被长度规则拒、错 lane 的帧被派生拒 |

可复跑命令（把 `REPO` 指到本仓 checkout）：

```sh
# 契约与 wire
cargo test -p metal-api-core
cargo test -p metal-api-ipc

# 两条 rail（Lavapipe）
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  cargo test -p metal-api-vulkan --test render_narrow_texture_e2e
cargo test -p metal-api-native
cargo test -p metal-api-vulkan

# conformance 与全套门
python3 -m unittest discover -s conformance -p 'test_*.py'
REPO=$REPO bash /home/hiliang/hackintosh/tools/gates-local.sh
REPO=$REPO bash /home/hiliang/hackintosh/tools/lavapipe-smoke.sh
```

本增量落地的读数：`GATES_OK`；`LAVAPIPE_SMOKE_OK suites=37 captures=111`（基线 36 / 108，新增
v37 在 trace、objects、objects-async 三条 capture 上都是 `PASS`）。证据目录见
`evidence/narrow-lanes-<sha>-2026-09-18/`。

## 5. 未决

- **native rail 的 widen**：Apple 侧的 `.r8Unorm` / `.rg8Unorm` 自测（reviewed MSL 模块 +
  `NativeOracle` 的读数）落地后，`SUPPORTED_RENDER_TEXTURE_FORMATS` 才能加这两个成员。那是一件
  自己的增量，本文件只记录边界。
- **R8 作为 render target**：需要 `AttachmentFormat` 加成员、`attachment_vk_format` 加映射，
  以及一个有 oracle 的 R8 附件形状（谁来读它的字节）。census 目前没有该形状的读数支撑。
- **`conformance/narrow-class.json` 不加宽**：`fragment_textures` 在该类的
  `RENDER_INERT_KEYS` 里，属"类未覆盖的轴"。把它写成一条 covered rule 是 G2-f 声明自身的演化，
  应由写入 `texture_bind` 桶之后的 census 读数支撑（与 v36 时同一口径）。
- **`texture_source` 的 stride**：census 里这些 R8 视图的行 padding 是另一个桶
  （`PaddedRows`，`row_length_texels ∈ {32, 144, 240, 1920}`），缺的是"窗口带 stride"的契约
  能力（建议独立 change `render-texture-strided-lease`），不在本增量的范围内 —— 本增量只解
  bind/view 侧的格式闸门。
- **消费方随动（另一仓 reims）**：R 侧读 provider 快照的格式列表放行 `texture_bind` 桶里的
  R8 / R8G8 形状；本增量 E 仓一行 reims 代码未动。
